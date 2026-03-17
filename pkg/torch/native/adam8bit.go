// adam8bit.go implements 8-bit Adam optimizer for 75% less optimizer memory.
//
// WHAT: Standard Adam stores two FP32 state tensors per parameter:
//   - First moment (m): running mean of gradients
//   - Second moment (v): running mean of squared gradients
// For a 7B model, this is ~28GB of optimizer state alone.
//
// 8-bit Adam quantizes m and v to INT8 with dynamic scaling, cutting
// optimizer memory from 8 bytes/param to 2 bytes/param (75% reduction).
// The parameter updates themselves are computed in FP32 for accuracy.
//
// HOW: Block-wise quantization with 64-element blocks. Each block has
// its own absmax scale factor, providing better precision than global scaling.
//
// ACCURACY: Equivalent to FP32 Adam in practice. Research (Dettmers 2022)
// shows zero degradation on LLM training at 7B-65B scale.
package native

import (
	"math"
)

// BlockSize is the quantization block size for 8-bit states.
// 64 elements per block provides a good precision/overhead tradeoff.
const Adam8BitBlockSize = 64

// Adam8BitState holds quantized optimizer state for one parameter.
type Adam8BitState struct {
	// Quantized first moment (mean of gradients).
	M      []int8    // [paramSize]
	MScale []float32 // [paramSize / blockSize] absmax per block

	// Quantized second moment (mean of squared gradients).
	V      []int8    // [paramSize]
	VScale []float32 // [paramSize / blockSize] absmax per block

	// Step counter for bias correction.
	Step int
}

// Adam8Bit is a memory-efficient Adam optimizer using INT8 state quantization.
type Adam8Bit struct {
	lr    float32
	beta1 float32
	beta2 float32
	eps   float32

	states map[string]*Adam8BitState
}

// NewAdam8Bit creates an 8-bit Adam optimizer.
func NewAdam8Bit(lr, beta1, beta2, eps float32) *Adam8Bit {
	if beta1 == 0 {
		beta1 = 0.9
	}
	if beta2 == 0 {
		beta2 = 0.999
	}
	if eps == 0 {
		eps = 1e-8
	}
	return &Adam8Bit{
		lr:     lr,
		beta1:  beta1,
		beta2:  beta2,
		eps:    eps,
		states: make(map[string]*Adam8BitState),
	}
}

// Step performs one optimizer update for a parameter.
// params and grads are modified in-place.
func (opt *Adam8Bit) Step(name string, params, grads []float32) {
	n := len(params)
	numBlocks := (n + Adam8BitBlockSize - 1) / Adam8BitBlockSize

	// Initialize state if first step.
	state, ok := opt.states[name]
	if !ok {
		state = &Adam8BitState{
			M:      make([]int8, n),
			MScale: make([]float32, numBlocks),
			V:      make([]int8, n),
			VScale: make([]float32, numBlocks),
		}
		opt.states[name] = state
	}
	state.Step++

	// Bias correction factors.
	bc1 := 1.0 - math.Pow(float64(opt.beta1), float64(state.Step))
	bc2 := 1.0 - math.Pow(float64(opt.beta2), float64(state.Step))

	// Process in blocks for quantization precision.
	for b := 0; b < numBlocks; b++ {
		start := b * Adam8BitBlockSize
		end := start + Adam8BitBlockSize
		if end > n {
			end = n
		}

		// Dequantize m and v for this block.
		mScale := state.MScale[b]
		vScale := state.VScale[b]

		var newMMax, newVMax float32

		for i := start; i < end; i++ {
			// Dequantize current state.
			mVal := dequant8(state.M[i], mScale)
			vVal := dequant8(state.V[i], vScale)

			g := grads[i]

			// Standard Adam update in FP32.
			mVal = opt.beta1*mVal + (1-opt.beta1)*g
			vVal = opt.beta2*vVal + (1-opt.beta2)*g*g

			// Bias-corrected estimates.
			mHat := float64(mVal) / bc1
			vHat := float64(vVal) / bc2

			// Parameter update.
			params[i] -= opt.lr * float32(mHat/(math.Sqrt(vHat)+float64(opt.eps)))

			// Track block max for re-quantization.
			absM := float32(math.Abs(float64(mVal)))
			absV := float32(math.Abs(float64(vVal)))
			if absM > newMMax {
				newMMax = absM
			}
			if absV > newVMax {
				newVMax = absV
			}

			// Re-quantize with new scale (computed after loop).
			// Store FP32 temporarily via grads buffer (we're done with grads[i]).
			grads[i] = mVal // Reuse grads as temp buffer for m.
			state.V[i] = 0  // Will be set below.

			// Store v separately.
			if i < end {
				// Use a trick: encode vVal in the sign-extended range.
				state.V[i] = quant8(vVal, newVMax)
			}
		}

		// Update scales and re-quantize m.
		if newMMax < 1e-10 {
			newMMax = 1e-10
		}
		if newVMax < 1e-10 {
			newVMax = 1e-10
		}
		state.MScale[b] = newMMax
		state.VScale[b] = newVMax

		for i := start; i < end; i++ {
			state.M[i] = quant8(grads[i], newMMax)     // grads[i] has mVal from above
			state.V[i] = quant8(dequant8(state.V[i], vScale), newVMax) // re-scale v
		}
	}
}

// quant8 quantizes a float to INT8 given the block's absmax.
func quant8(val, absmax float32) int8 {
	if absmax < 1e-10 {
		return 0
	}
	scaled := val / absmax * 127.0
	if scaled > 127 {
		return 127
	}
	if scaled < -127 {
		return -127
	}
	return int8(scaled)
}

// dequant8 restores a float from INT8 given the block's absmax.
func dequant8(val int8, absmax float32) float32 {
	return float32(val) / 127.0 * absmax
}

// MemoryUsage returns bytes used by optimizer states.
func (opt *Adam8Bit) MemoryUsage() int64 {
	var total int64
	for _, state := range opt.states {
		// INT8 m + INT8 v + FP32 scales for both.
		total += int64(len(state.M))                   // 1 byte per param (m)
		total += int64(len(state.V))                   // 1 byte per param (v)
		total += int64(len(state.MScale) * 4)          // 4 bytes per block (m scale)
		total += int64(len(state.VScale) * 4)          // 4 bytes per block (v scale)
	}
	return total
}

// FP32Equivalent returns what standard Adam would cost for the same params.
func (opt *Adam8Bit) FP32Equivalent() int64 {
	var total int64
	for _, state := range opt.states {
		total += int64(len(state.M)) * 4 * 2 // FP32 m + FP32 v = 8 bytes/param
	}
	return total
}

// CompressionRatio returns memory savings vs standard Adam.
func (opt *Adam8Bit) CompressionRatio() float64 {
	used := opt.MemoryUsage()
	if used == 0 {
		return 0
	}
	return float64(opt.FP32Equivalent()) / float64(used)
}

// Stats returns optimizer metrics.
func (opt *Adam8Bit) Stats() map[string]interface{} {
	return map[string]interface{}{
		"lr":               opt.lr,
		"beta1":            opt.beta1,
		"beta2":            opt.beta2,
		"params_tracked":   len(opt.states),
		"memory_bytes":     opt.MemoryUsage(),
		"fp32_equivalent":  opt.FP32Equivalent(),
		"compression":      opt.CompressionRatio(),
	}
}
