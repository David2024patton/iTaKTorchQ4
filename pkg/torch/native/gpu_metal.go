//go:build metal

// gpu_metal.go implements GPU-accelerated tensor operations on macOS using
// Apple's Accelerate framework (cblas) via runtime dynamic loading.
//
// ZERO CGO: Uses ebitengine/purego for dlopen on macOS.
//
// WHY ACCELERATE INSTEAD OF METAL COMPUTE SHADERS:
//   Apple's Accelerate framework routes cblas calls through the AMX coprocessor
//   on Apple Silicon (M1/M2/M3/M4), which is a dedicated matrix math unit.
//   For BLAS operations (matmul, matvec), AMX is as fast as Metal compute
//   shaders and doesn't require Objective-C runtime or Metal API plumbing.
//
// UNIFIED MEMORY ADVANTAGE:
//   Apple Silicon shares CPU and GPU memory. Weight tensors in Go slices
//   are already accessible to the AMX/GPU with zero copies. The "upload"
//   step is essentially free - we just pass the pointer.
//
// BUILD: go build -tags metal ./...
//
// When built with this tag, the stub (wgpu_stub.go) and other GPU backends
// are excluded. Accelerate provides the GPUBackend implementation.
package native

import (
	"fmt"
	"runtime"
	"sync"
	"unsafe"

	"github.com/ebitengine/purego"
)

// CBLAS constants matching Apple's Accelerate framework headers.
const (
	cblasRowMajor  = 101
	cblasNoTrans   = 111
	cblasTrans     = 112
)

// GPUBackend wraps Apple's Accelerate framework for AMX-accelerated BLAS.
// When built with -tags metal, this replaces the Vulkan, CUDA, and stub backends.
type GPUBackend struct {
	available bool
	mu        sync.Mutex

	// Resolved function pointers from Accelerate framework.
	fnCblasSgemm func(
		order, transA, transB int32,
		M, N, K int32,
		alpha float32,
		A uintptr, lda int32,
		B uintptr, ldb int32,
		beta float32,
		C uintptr, ldc int32,
	)
	fnCblasSgemv func(
		order, trans int32,
		M, N int32,
		alpha float32,
		A uintptr, lda int32,
		X uintptr, incX int32,
		beta float32,
		Y uintptr, incY int32,
	)

	// Device info.
	DeviceName string
	IsAppleSi  bool // true for M1/M2/M3/M4 (AMX available)

	// Library handle.
	accelLib uintptr
}

// NewGPUBackend initializes the Accelerate framework backend for macOS.
func NewGPUBackend() *GPUBackend {
	b := &GPUBackend{}

	if runtime.GOOS != "darwin" {
		fmt.Println("[Metal] Not macOS, falling back to CPU")
		return b
	}

	// Load Apple's Accelerate framework.
	lib, err := purego.Dlopen(
		"/System/Library/Frameworks/Accelerate.framework/Accelerate",
		purego.RTLD_LAZY|purego.RTLD_GLOBAL,
	)
	if err != nil {
		fmt.Printf("[Metal] Failed to load Accelerate.framework: %v\n", err)
		return b
	}
	b.accelLib = lib

	// Resolve cblas function pointers.
	purego.RegisterLibFunc(&b.fnCblasSgemm, lib, "cblas_sgemm")
	purego.RegisterLibFunc(&b.fnCblasSgemv, lib, "cblas_sgemv")

	if b.fnCblasSgemm == nil || b.fnCblasSgemv == nil {
		fmt.Println("[Metal] Failed to resolve cblas symbols")
		return b
	}

	// Detect Apple Silicon vs Intel.
	b.IsAppleSi = runtime.GOARCH == "arm64"
	if b.IsAppleSi {
		b.DeviceName = "Apple Silicon (AMX)"
	} else {
		b.DeviceName = "Intel Mac (Accelerate)"
	}

	b.available = true
	fmt.Printf("[Metal] %s - Accelerate framework loaded (unified memory, zero-copy weights)\n", b.DeviceName)
	return b
}

// IsAvailable returns true if Accelerate loaded successfully.
func (b *GPUBackend) IsAvailable() bool { return b.available }

// UploadWeight is a no-op on Apple Silicon.
// Unified memory means Go slice data is already accessible to the AMX
// coprocessor without any copy. The pointer passed to cblas_sgemm
// directly reads from the Go heap.
func (b *GPUBackend) UploadWeight(t *Tensor) {}

// ---------- Accelerate-Accelerated Operations ----------

// MatMulGPU performs C = A * B using cblas_sgemm (AMX-accelerated on Apple Silicon).
// Zero-copy: passes Go slice pointers directly to Accelerate.
func (b *GPUBackend) MatMulGPU(a, bTensor *Tensor) *Tensor {
	if !b.available {
		return MatMul(a, bTensor)
	}

	M := int32(a.Shape[0])
	K := int32(a.Shape[1])
	N := int32(bTensor.Shape[1])

	result := NewTensor([]int{int(M), int(N)})

	b.mu.Lock()
	defer b.mu.Unlock()

	alpha := float32(1.0)
	beta := float32(0.0)

	// cblas_sgemm with row-major order - no transposition tricks needed.
	// The AMX coprocessor handles the actual multiplication in hardware.
	b.fnCblasSgemm(
		int32(cblasRowMajor), int32(cblasNoTrans), int32(cblasNoTrans),
		M, N, K,
		alpha,
		uintptr(unsafe.Pointer(&a.Data[0])), K,
		uintptr(unsafe.Pointer(&bTensor.Data[0])), N,
		beta,
		uintptr(unsafe.Pointer(&result.Data[0])), N,
	)

	return result
}

// MatVecMulGPU performs out = A * v using cblas_sgemv (AMX-accelerated).
func (b *GPUBackend) MatVecMulGPU(a, v *Tensor) *Tensor {
	if !b.available {
		return MatVecMul(a, v)
	}

	M := int32(a.Shape[0])
	K := int32(a.Shape[1])

	result := NewTensor([]int{int(M)})

	b.mu.Lock()
	defer b.mu.Unlock()

	alpha := float32(1.0)
	beta := float32(0.0)

	// Row-major A (M x K) * v (K x 1) = result (M x 1).
	b.fnCblasSgemv(
		int32(cblasRowMajor), int32(cblasNoTrans),
		M, K,
		alpha,
		uintptr(unsafe.Pointer(&a.Data[0])), K,
		uintptr(unsafe.Pointer(&v.Data[0])), 1,
		beta,
		uintptr(unsafe.Pointer(&result.Data[0])), 1,
	)

	return result
}

// ---------- CPU Fallback Operations ----------
// Element-wise ops stay on CPU. On Apple Silicon, these still benefit
// from the high-bandwidth unified memory and NEON SIMD.

func (b *GPUBackend) SiLUGPU(t *Tensor) *Tensor                    { return SiLU(t) }
func (b *GPUBackend) AddGPU(a, bT *Tensor) *Tensor                 { return Add(a, bT) }
func (b *GPUBackend) MulGPU(a, bT *Tensor) *Tensor                 { return Mul(a, bT) }
func (b *GPUBackend) RMSNormGPU(in, w *Tensor, eps float32) *Tensor { return RMSNorm(in, w, eps) }
func (b *GPUBackend) SoftmaxGPU(t *Tensor) *Tensor                  { return Softmax(t) }
func (b *GPUBackend) SiLUMulGPU(gate, up *Tensor) *Tensor           { return Mul(SiLU(gate), up) }

// Close releases Accelerate framework resources.
func (b *GPUBackend) Close() {
	// purego doesn't require explicit dlclose.
	b.available = false
}
