// quantization_aware_training.go implements fake-quantize operations
// for quantization-aware training (QAT).
//
// WHAT: Standard quantization (after training) approximates FP32 weights
// with INT4/INT8. This introduces quantization error that degrades accuracy.
// QAT simulates this quantization DURING training so the model learns to
// compensate for the reduced precision.
//
// HOW: Insert "fake quantize" ops in the forward pass:
//   x_fake = dequant(quant(x))  // Round-trip through quantization
// The forward pass sees quantized values; the backward pass uses straight-
// through estimator (STE) to flow gradients through the non-differentiable
// rounding operation.
//
// GAIN: Models quantized with QAT retain 95-99% of full-precision accuracy,
// vs 90-95% for post-training quantization. Critical for INT4 deployment.
//
// REFERENCE: Unsloth, LlamaFactory support QAT workflows.
package native

import (
	"math"
)

// QuantConfig specifies the target quantization parameters.
type QuantConfig struct {
	Bits         int     // Target bit width (4 or 8)
	Symmetric    bool    // Symmetric (-max..max) vs asymmetric (min..max)
	PerChannel   bool    // Per-channel vs per-tensor quantization
	GroupSize    int     // Group quantization size (0 = no groups, e.g., 128 for GPTQ-style)
}

// DefaultQATConfig returns settings for INT4 group quantization (GPTQ-style).
func DefaultQATConfig() QuantConfig {
	return QuantConfig{
		Bits:       4,
		Symmetric:  true,
		PerChannel: false,
		GroupSize:  128,
	}
}

// FakeQuantize simulates quantization in the forward pass for QAT.
// Input values are quantized to the target precision and immediately
// dequantized back to FP32. The rounding error trains the model to be
// robust to quantization.
//
// During backward pass, gradients flow through unchanged (STE).
func FakeQuantize(data []float32, config QuantConfig) []float32 {
	if config.GroupSize > 0 {
		return fakeQuantizeGrouped(data, config)
	}
	return fakeQuantizeTensor(data, config)
}

// fakeQuantizeTensor applies per-tensor fake quantization.
func fakeQuantizeTensor(data []float32, config QuantConfig) []float32 {
	out := make([]float32, len(data))
	qMin, qMax := quantRange(config.Bits, config.Symmetric)

	// Compute scale and zero point.
	dataMin, dataMax := minMaxF32(data)
	scale, zeroPoint := computeScaleZP(dataMin, dataMax, qMin, qMax, config.Symmetric)

	// Fake quantize: round-trip through quantization.
	for i, v := range data {
		q := math.Round(float64(v)/float64(scale)) + float64(zeroPoint)
		q = math.Max(float64(qMin), math.Min(float64(qMax), q))
		out[i] = (float32(q) - float32(zeroPoint)) * scale
	}

	return out
}

// fakeQuantizeGrouped applies group-wise fake quantization.
// Each group of GroupSize elements has its own scale/zero point.
func fakeQuantizeGrouped(data []float32, config QuantConfig) []float32 {
	out := make([]float32, len(data))
	groupSize := config.GroupSize
	qMin, qMax := quantRange(config.Bits, config.Symmetric)

	for g := 0; g < len(data); g += groupSize {
		end := g + groupSize
		if end > len(data) {
			end = len(data)
		}
		group := data[g:end]

		dataMin, dataMax := minMaxF32(group)
		scale, zeroPoint := computeScaleZP(dataMin, dataMax, qMin, qMax, config.Symmetric)

		for i, v := range group {
			q := math.Round(float64(v)/float64(scale)) + float64(zeroPoint)
			q = math.Max(float64(qMin), math.Min(float64(qMax), q))
			out[g+i] = (float32(q) - float32(zeroPoint)) * scale
		}
	}

	return out
}

// StraightThroughGrad implements the straight-through estimator for backward.
// During backward, gradients pass through fake-quantize unchanged, but we
// zero out gradients for values that were clamped (outside quantization range).
func StraightThroughGrad(grad, originalData []float32, config QuantConfig) []float32 {
	out := make([]float32, len(grad))
	qMin, qMax := quantRange(config.Bits, config.Symmetric)

	dataMin, dataMax := minMaxF32(originalData)
	scale, _ := computeScaleZP(dataMin, dataMax, qMin, qMax, config.Symmetric)

	loBound := float32(qMin) * scale
	hiBound := float32(qMax) * scale

	for i, v := range originalData {
		if v >= loBound && v <= hiBound {
			out[i] = grad[i] // Pass through.
		}
		// Otherwise: gradient is 0 (clamped region).
	}

	return out
}

// QuantizeWeights applies actual quantization (not fake) for deployment.
// Returns quantized INT8 values and scale factors.
func QuantizeWeights(data []float32, config QuantConfig) ([]int8, []float32) {
	groupSize := config.GroupSize
	if groupSize <= 0 {
		groupSize = len(data)
	}
	numGroups := (len(data) + groupSize - 1) / groupSize

	quantized := make([]int8, len(data))
	scales := make([]float32, numGroups)
	qMin, qMax := quantRange(config.Bits, config.Symmetric)

	for g := 0; g < numGroups; g++ {
		start := g * groupSize
		end := start + groupSize
		if end > len(data) {
			end = len(data)
		}

		_, dataMax := minMaxF32(data[start:end])
		absMax := float32(math.Max(math.Abs(float64(dataMax)), 1e-10))
		scale := absMax / float32(qMax)
		scales[g] = scale

		for i := start; i < end; i++ {
			q := int(math.Round(float64(data[i] / scale)))
			if q < int(qMin) {
				q = int(qMin)
			}
			if q > int(qMax) {
				q = int(qMax)
			}
			quantized[i] = int8(q)
		}
	}

	return quantized, scales
}

// quantRange returns the min and max quantized values for a given bit width.
func quantRange(bits int, symmetric bool) (float64, float64) {
	if symmetric {
		maxVal := math.Pow(2, float64(bits-1)) - 1
		return -maxVal, maxVal
	}
	return 0, math.Pow(2, float64(bits)) - 1
}

// computeScaleZP computes quantization scale and zero point.
func computeScaleZP(dataMin, dataMax float32, qMin, qMax float64, symmetric bool) (float32, float32) {
	if symmetric {
		absMax := float32(math.Max(math.Abs(float64(dataMin)), math.Abs(float64(dataMax))))
		if absMax < 1e-10 {
			absMax = 1e-10
		}
		scale := absMax / float32(qMax)
		return scale, 0
	}

	dataRange := dataMax - dataMin
	if dataRange < 1e-10 {
		dataRange = 1e-10
	}
	scale := dataRange / float32(qMax-qMin)
	zeroPoint := float32(qMin) - dataMin/scale
	return scale, zeroPoint
}

// minMaxF32 returns the min and max of a float32 slice.
func minMaxF32(data []float32) (float32, float32) {
	if len(data) == 0 {
		return 0, 0
	}
	min, max := data[0], data[0]
	for _, v := range data[1:] {
		if v < min {
			min = v
		}
		if v > max {
			max = v
		}
	}
	return min, max
}
