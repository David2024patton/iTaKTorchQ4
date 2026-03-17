// Package native provides a pure Go tensor engine for tiny transformer models.
//
// WHY THIS EXISTS:
// The main TorchEngine uses llama.cpp via FFI (purego), which requires
// platform-specific shared libraries. GOTensor eliminates ALL external
// dependencies: no DLLs, no .so files, no CGo. Just `go build` and run.
//
// TRADE-OFFS:
//   - MUCH slower than llama.cpp (no SIMD intrinsics, no GPU offload)
//   - Only viable for tiny models (<1B parameters, ideally <500M)
//   - No advanced features: no flash attention, no quantized KV cache
//   - Primary value: zero-dependency portability + educational clarity
//
// ARCHITECTURE:
//   tensor.go  - N-dimensional tensor with basic math ops
//   matmul.go  - Matrix multiplication (naive + cache-blocked)
//   attention.go - Self-attention mechanism
//   model.go   - Transformer forward pass + GGUF loading stub
//   engine.go  - Implements torch.Engine interface
package native

import (
	"fmt"
	"math"
)

// Tensor is an N-dimensional array of float32 values.
//
// Layout:
//   - Shape: dimensions, e.g. [batch=1, seq=128, hidden=768]
//   - Strides: how many elements to skip for each dimension
//   - Data: flat backing array (row-major order)
//
// Example:
//
//	t := NewTensor([]int{2, 3}) // 2x3 matrix
//	t.Set([]int{0, 1}, 3.14)
//	val := t.Get([]int{0, 1})  // -> 3.14
type Tensor struct {
	Data    []float32 // flat backing array (row-major)
	Shape   []int     // dimensions
	Strides []int     // elements to skip per dimension
	pooled  bool      // if true, Data was allocated from tensor pool
}

// NewTensor creates a zero-initialized tensor with the given shape.
func NewTensor(shape []int) *Tensor {
	size := 1
	for _, s := range shape {
		size *= s
	}

	strides := make([]int, len(shape))
	strides[len(shape)-1] = 1
	for i := len(shape) - 2; i >= 0; i-- {
		strides[i] = strides[i+1] * shape[i+1]
	}

	return &Tensor{
		Data:    make([]float32, size),
		Shape:   shape,
		Strides: strides,
	}
}

// NewTensorFrom creates a tensor from existing data (no copy).
func NewTensorFrom(shape []int, data []float32) *Tensor {
	strides := make([]int, len(shape))
	strides[len(shape)-1] = 1
	for i := len(shape) - 2; i >= 0; i-- {
		strides[i] = strides[i+1] * shape[i+1]
	}
	return &Tensor{Data: data, Shape: shape, Strides: strides}
}

// Size returns the total number of elements.
func (t *Tensor) Size() int {
	return len(t.Data)
}

// Get returns the element at the given indices.
func (t *Tensor) Get(indices []int) float32 {
	offset := 0
	for i, idx := range indices {
		offset += idx * t.Strides[i]
	}
	return t.Data[offset]
}

// Set writes a value at the given indices.
func (t *Tensor) Set(indices []int, val float32) {
	offset := 0
	for i, idx := range indices {
		offset += idx * t.Strides[i]
	}
	t.Data[offset] = val
}

// ---------- Element-wise Operations ----------

// Add performs element-wise addition: result = a + b.
// a and b must have the same shape.
func Add(a, b *Tensor) *Tensor {
	if len(a.Data) != len(b.Data) {
		panic(fmt.Sprintf("Add: shape mismatch: %v vs %v", a.Shape, b.Shape))
	}
	result := NewTensor(a.Shape)
	for i := range result.Data {
		result.Data[i] = a.Data[i] + b.Data[i]
	}
	return result
}

// Mul performs element-wise multiplication: result = a * b.
func Mul(a, b *Tensor) *Tensor {
	if len(a.Data) != len(b.Data) {
		panic(fmt.Sprintf("Mul: shape mismatch: %v vs %v", a.Shape, b.Shape))
	}
	result := NewTensor(a.Shape)
	for i := range result.Data {
		result.Data[i] = a.Data[i] * b.Data[i]
	}
	return result
}

// Scale multiplies every element by a scalar.
func Scale(t *Tensor, scalar float32) *Tensor {
	result := NewTensor(t.Shape)
	for i := range result.Data {
		result.Data[i] = t.Data[i] * scalar
	}
	return result
}

// ---------- Activation Functions ----------

// ReLU applies rectified linear unit: max(0, x).
func ReLU(t *Tensor) *Tensor {
	result := NewTensor(t.Shape)
	for i, v := range t.Data {
		if v > 0 {
			result.Data[i] = v
		}
	}
	return result
}

// GELU applies Gaussian Error Linear Unit (approximate).
// GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
func GELU(t *Tensor) *Tensor {
	result := NewTensor(t.Shape)
	sqrt2pi := float32(math.Sqrt(2.0 / math.Pi))
	for i, x := range t.Data {
		inner := sqrt2pi * (x + 0.044715*x*x*x)
		result.Data[i] = 0.5 * x * (1.0 + float32(math.Tanh(float64(inner))))
	}
	return result
}

// SiLU applies Sigmoid Linear Unit: x * sigmoid(x).
// Used in Qwen3's FFN layers.
func SiLU(t *Tensor) *Tensor {
	result := NewTensor(t.Shape)
	for i, x := range t.Data {
		sigmoid := float32(1.0 / (1.0 + math.Exp(-float64(x))))
		result.Data[i] = x * sigmoid
	}
	return result
}

// ---------- Normalization ----------

// RMSNorm applies Root Mean Square Layer Normalization.
// Used in modern transformers (Llama, Qwen) instead of LayerNorm.
//
// Formula: x_normalized = (x / sqrt(mean(x^2) + eps)) * weight
//
// Parameters:
//   - input: [seq, hidden] tensor
//   - weight: [hidden] tensor (learned gamma parameter)
//   - eps: small value for numerical stability (typically 1e-6)
func RMSNorm(input, weight *Tensor, eps float32) *Tensor {
	hidden := input.Shape[len(input.Shape)-1]
	result := NewTensor(input.Shape)

	numVectors := len(input.Data) / hidden
	for v := 0; v < numVectors; v++ {
		offset := v * hidden

		// Compute mean(x^2).
		var sumSq float32
		for i := 0; i < hidden; i++ {
			val := input.Data[offset+i]
			sumSq += val * val
		}
		rms := float32(math.Sqrt(float64(sumSq/float32(hidden) + eps)))

		// Normalize and apply weight.
		for i := 0; i < hidden; i++ {
			result.Data[offset+i] = (input.Data[offset+i] / rms) * weight.Data[i]
		}
	}

	return result
}

// Softmax applies softmax along the last dimension.
// softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
func Softmax(t *Tensor) *Tensor {
	lastDim := t.Shape[len(t.Shape)-1]
	result := NewTensor(t.Shape)

	numVectors := len(t.Data) / lastDim
	for v := 0; v < numVectors; v++ {
		offset := v * lastDim

		// Find max for numerical stability.
		maxVal := t.Data[offset]
		for i := 1; i < lastDim; i++ {
			if t.Data[offset+i] > maxVal {
				maxVal = t.Data[offset+i]
			}
		}

		// Compute exp(x - max) and sum.
		var sum float32
		for i := 0; i < lastDim; i++ {
			result.Data[offset+i] = float32(math.Exp(float64(t.Data[offset+i] - maxVal)))
			sum += result.Data[offset+i]
		}

		// Normalize.
		for i := 0; i < lastDim; i++ {
			result.Data[offset+i] /= sum
		}
	}

	return result
}
