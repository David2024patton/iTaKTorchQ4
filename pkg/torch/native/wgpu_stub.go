//go:build !wgpu && !cuda && !metal

// wgpu_stub.go provides CPU-only fallback when the wgpu build tag is not set.
//
// WHY: WebGPU support requires the cogentcore.org/webgpu/wgpu dependency,
// which pulls in platform-specific GPU drivers. By default, GOTensor uses
// CPU-only computation. To enable GPU acceleration, build with:
//
//   go build -tags wgpu ./...
//
// This stub ensures `go build ./...` always works without GPU dependencies.
package native

// GPUBackend represents a GPU compute backend for accelerated matrix operations.
// When the wgpu build tag is not set, this is a no-op stub.
type GPUBackend struct {
	available bool
}

// NewGPUBackend creates a stub GPU backend (always CPU fallback).
func NewGPUBackend() *GPUBackend {
	return &GPUBackend{available: false}
}

// IsAvailable returns false for the stub backend (no GPU).
func (g *GPUBackend) IsAvailable() bool {
	return false
}

// MatMulGPU falls back to CPU MatMul when no GPU backend is compiled.
func (g *GPUBackend) MatMulGPU(a, b *Tensor) *Tensor {
	return MatMul(a, b)
}

// SiLUGPU falls back to CPU SiLU.
func (g *GPUBackend) SiLUGPU(t *Tensor) *Tensor {
	return SiLU(t)
}

// AddGPU falls back to CPU Add.
func (g *GPUBackend) AddGPU(a, b *Tensor) *Tensor {
	return Add(a, b)
}

// MulGPU falls back to CPU Mul.
func (g *GPUBackend) MulGPU(a, b *Tensor) *Tensor {
	return Mul(a, b)
}

// RMSNormGPU falls back to CPU RMSNorm.
func (g *GPUBackend) RMSNormGPU(input, weight *Tensor, eps float32) *Tensor {
	return RMSNorm(input, weight, eps)
}

// SoftmaxGPU falls back to CPU Softmax.
func (g *GPUBackend) SoftmaxGPU(t *Tensor) *Tensor {
	return Softmax(t)
}

// MatVecMulGPU falls back to CPU MatVecMul.
func (g *GPUBackend) MatVecMulGPU(a, v *Tensor) *Tensor {
	return MatVecMul(a, v)
}

// SiLUMulGPU falls back to CPU SiLU + Mul.
func (g *GPUBackend) SiLUMulGPU(gate, up *Tensor) *Tensor {
	return Mul(SiLU(gate), up)
}

// Close is a no-op for the stub backend.
func (g *GPUBackend) Close() {}

// UploadWeight is a no-op for the stub backend (no GPU to upload to).
func (g *GPUBackend) UploadWeight(t *Tensor) {}
