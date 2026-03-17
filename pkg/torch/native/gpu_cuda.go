//go:build cuda

// gpu_cuda.go implements GPU-accelerated tensor operations using NVIDIA
// CUDA/cuBLAS via runtime dynamic library loading.
//
// ZERO CGO: Uses ebitengine/purego for platform-agnostic dlopen/LoadLibrary.
// Works on Linux, WSL, and Windows without requiring CGO.
//
// RUNTIME REQUIREMENTS: NVIDIA GPU driver only. No CUDA Toolkit needed.
//   Linux:   libcudart.so.12 and libcublas.so.12 (from nvidia-driver)
//   Windows: cudart64_12.dll and cublas64_12.dll (from GPU driver)
//   WSL:     /usr/lib/wsl/lib/libcuda*.so (mapped by NVIDIA WSL driver)
//
// BUILD: go build -tags cuda ./...
//
// When built with this tag, the stub (wgpu_stub.go) and Vulkan (wgpu.go)
// backends are excluded. CUDA provides the GPUBackend implementation.
package native

import (
	"fmt"
	"runtime"
	"strings"
	"sync"
	"unsafe"

	"github.com/ebitengine/purego"
)

// cuBLAS / CUDA constants.
const (
	cublasStatusSuccess  = 0
	cublasOpN            = 0 // No transpose
	cublasOpT            = 1 // Transpose
	cublasTensorOpMath   = 99

	cudaSuccess      = 0
	cudaMemcpyH2D    = 1
	cudaMemcpyD2H    = 2
)

// GPUBackend wraps dynamically-loaded CUDA/cuBLAS for tensor core acceleration.
// When built with -tags cuda, this replaces the Vulkan and stub GPUBackend.
type GPUBackend struct {
	available bool
	mu        sync.Mutex

	// cuBLAS handle (opaque pointer from cublasCreate).
	handle uintptr

	// Resolved function pointers (set via purego.RegisterLibFunc).
	fnCudaMalloc   func(devPtr *uintptr, size uintptr) int32
	fnCudaFree     func(devPtr uintptr) int32
	fnCudaMemcpy   func(dst, src uintptr, count uintptr, kind int32) int32

	fnCublasCreate    func(handle *uintptr) int32
	fnCublasDestroy   func(handle uintptr) int32
	fnCublasSetMath   func(handle uintptr, mode int32) int32
	fnCublasSgemm     func(handle uintptr, transa, transb int32, m, n, k int32, alpha *float32, A uintptr, lda int32, B uintptr, ldb int32, beta *float32, C uintptr, ldc int32) int32
	fnCublasSgemv     func(handle uintptr, trans int32, m, n int32, alpha *float32, A uintptr, lda int32, x uintptr, incx int32, beta *float32, y uintptr, incy int32) int32

	// Cached device info.
	DeviceName     string
	ComputeMajor   int
	ComputeMinor   int
	TotalVRAM      int64
	HasTensorCores bool

	// Library handles for cleanup.
	cudartLib uintptr
	cublasLib uintptr

	// Persistent weight cache: tensor data pointer -> device pointer.
	// Weights are uploaded once via UploadWeight() and reused across calls.
	// Only activations (which change every call) get copied per-inference.
	weightCache map[uintptr]deviceBuf
}

// deviceBuf holds a GPU allocation for a cached weight tensor.
type deviceBuf struct {
	ptr  uintptr // device pointer from cudaMalloc
	size uintptr // size in bytes
}

// cudaLibNames returns the library filenames to try for each platform.
func cudaLibNames() (cudartNames, cublasNames []string) {
	if runtime.GOOS == "windows" {
		return []string{"cudart64_12.dll", "cudart64_11.dll"},
			[]string{"cublas64_12.dll", "cublas64_11.dll"}
	}
	// Linux and WSL.
	return []string{"libcudart.so.12", "libcudart.so.11.0", "libcudart.so"},
		[]string{"libcublas.so.12", "libcublas.so.11", "libcublas.so"}
}

// loadLib is defined in platform-specific files:
//   gpu_cuda_unix.go    - purego.Dlopen (Linux, WSL, macOS)
//   gpu_cuda_windows.go - syscall.LoadDLL (Windows)


// NewGPUBackend attempts to load CUDA libraries from the NVIDIA GPU driver.
// If the libraries aren't found, returns a backend with available=false
// and the engine falls back to CPU computation.
func NewGPUBackend() *GPUBackend {
	b := &GPUBackend{}

	cudartNames, cublasNames := cudaLibNames()

	var err error
	b.cudartLib, err = loadLib(cudartNames)
	if err != nil {
		fmt.Printf("[CUDA] cudart not found: %v (falling back to CPU)\n", err)
		return b
	}

	b.cublasLib, err = loadLib(cublasNames)
	if err != nil {
		fmt.Printf("[CUDA] cublas not found: %v (falling back to CPU)\n", err)
		return b
	}

	// Resolve CUDA runtime functions.
	purego.RegisterLibFunc(&b.fnCudaMalloc, b.cudartLib, "cudaMalloc")
	purego.RegisterLibFunc(&b.fnCudaFree, b.cudartLib, "cudaFree")
	purego.RegisterLibFunc(&b.fnCudaMemcpy, b.cudartLib, "cudaMemcpy")

	if b.fnCudaMalloc == nil || b.fnCudaFree == nil || b.fnCudaMemcpy == nil {
		fmt.Println("[CUDA] Failed to resolve cudart functions (falling back to CPU)")
		return b
	}

	// Resolve cuBLAS functions.
	purego.RegisterLibFunc(&b.fnCublasCreate, b.cublasLib, "cublasCreate_v2")
	purego.RegisterLibFunc(&b.fnCublasDestroy, b.cublasLib, "cublasDestroy_v2")
	purego.RegisterLibFunc(&b.fnCublasSgemm, b.cublasLib, "cublasSgemm_v2")
	purego.RegisterLibFunc(&b.fnCublasSgemv, b.cublasLib, "cublasSgemv_v2")
	purego.RegisterLibFunc(&b.fnCublasSetMath, b.cublasLib, "cublasSetMathMode")

	if b.fnCublasCreate == nil || b.fnCublasSgemm == nil {
		fmt.Println("[CUDA] Failed to resolve cuBLAS functions (falling back to CPU)")
		return b
	}

	// Create cuBLAS handle.
	var handle uintptr
	if ret := b.fnCublasCreate(&handle); ret != cublasStatusSuccess {
		fmt.Printf("[CUDA] cublasCreate failed: %d (falling back to CPU)\n", ret)
		return b
	}
	b.handle = handle

	// Enable tensor core math.
	if b.fnCublasSetMath != nil {
		b.fnCublasSetMath(handle, cublasTensorOpMath)
	}

	// Query device properties.
	b.probeDevice()

	b.available = true
	b.weightCache = make(map[uintptr]deviceBuf)
	fmt.Printf("[CUDA] GPU ready: %s (%d MB VRAM, compute %d.%d, tensor_cores=%v)\n",
		b.DeviceName, b.TotalVRAM/(1024*1024), b.ComputeMajor, b.ComputeMinor, b.HasTensorCores)
	return b
}

// probeDevice reads GPU name, VRAM, and compute capability.
func (b *GPUBackend) probeDevice() {
	// Try to resolve cudaGetDeviceProperties.
	var fnGetProps func(props uintptr, device int32) int32
	purego.RegisterLibFunc(&fnGetProps, b.cudartLib, "cudaGetDeviceProperties")
	if fnGetProps == nil {
		return
	}

	// cudaDeviceProp is ~900 bytes. Allocate enough.
	var props [1024]byte
	if ret := fnGetProps(uintptr(unsafe.Pointer(&props[0])), 0); ret != cudaSuccess {
		return
	}

	// First 256 bytes = device name (null-terminated string).
	for i := 0; i < 256; i++ {
		if props[i] == 0 {
			b.DeviceName = string(props[:i])
			break
		}
	}

	// Bytes 256-263 = totalGlobalMem (size_t / uint64).
	b.TotalVRAM = int64(*(*uint64)(unsafe.Pointer(&props[256])))

	// Infer compute capability from device name (reliable fallback).
	switch {
	case strings.Contains(b.DeviceName, "RTX 40") || strings.Contains(b.DeviceName, "RTX 50"):
		b.ComputeMajor, b.ComputeMinor, b.HasTensorCores = 8, 9, true
	case strings.Contains(b.DeviceName, "RTX 30"):
		b.ComputeMajor, b.ComputeMinor, b.HasTensorCores = 8, 6, true
	case strings.Contains(b.DeviceName, "RTX 20") || strings.Contains(b.DeviceName, "TITAN"):
		b.ComputeMajor, b.ComputeMinor, b.HasTensorCores = 7, 5, true
	case strings.Contains(b.DeviceName, "GTX"):
		b.ComputeMajor, b.HasTensorCores = 6, false
	default:
		b.ComputeMajor, b.HasTensorCores = 5, false
	}
}

// IsAvailable returns true if CUDA/cuBLAS loaded successfully.
func (b *GPUBackend) IsAvailable() bool { return b.available }

// UploadWeight uploads a tensor to GPU memory and caches it by data pointer.
// Subsequent MatMul/MatVec calls with this tensor skip the H2D copy.
// Call this once per weight tensor during model load.
func (b *GPUBackend) UploadWeight(t *Tensor) {
	if !b.available || len(t.Data) == 0 {
		return
	}

	key := uintptr(unsafe.Pointer(&t.Data[0]))
	b.mu.Lock()
	defer b.mu.Unlock()

	if _, exists := b.weightCache[key]; exists {
		return // already cached
	}

	size := uintptr(len(t.Data) * 4)
	var dPtr uintptr
	if b.fnCudaMalloc(&dPtr, size) != 0 {
		return // allocation failed, will use per-call upload
	}

	b.fnCudaMemcpy(dPtr, uintptr(unsafe.Pointer(&t.Data[0])), size, cudaMemcpyH2D)
	b.weightCache[key] = deviceBuf{ptr: dPtr, size: size}
}

// getOrUpload returns a device pointer for a tensor. If cached, returns the
// cached pointer and false (caller should NOT free). If not cached, allocates
// and copies, returning the pointer and true (caller MUST free).
func (b *GPUBackend) getOrUpload(t *Tensor) (uintptr, bool) {
	key := uintptr(unsafe.Pointer(&t.Data[0]))
	if cached, ok := b.weightCache[key]; ok {
		return cached.ptr, false // cached, don't free
	}

	// Not cached, allocate and copy (activation tensor).
	size := uintptr(len(t.Data) * 4)
	var dPtr uintptr
	if b.fnCudaMalloc(&dPtr, size) != 0 {
		return 0, false
	}
	b.fnCudaMemcpy(dPtr, uintptr(unsafe.Pointer(&t.Data[0])), size, cudaMemcpyH2D)
	return dPtr, true // caller must free
}

// MatMulGPU performs C = A * B using cuBLAS sgemm (tensor core accelerated).
// Uses cached device pointers for weight tensors to skip H2D copies.
func (b *GPUBackend) MatMulGPU(a, bTensor *Tensor) *Tensor {
	if !b.available {
		return MatMul(a, bTensor)
	}

	M, K, N := a.Shape[0], a.Shape[1], bTensor.Shape[1]

	b.mu.Lock()
	defer b.mu.Unlock()

	// Get or upload A (often an activation - temporary).
	dA, freeA := b.getOrUpload(a)
	if dA == 0 {
		return MatMul(a, bTensor)
	}
	if freeA {
		defer b.fnCudaFree(dA)
	}

	// Get or upload B (often a weight - cached).
	dB, freeB := b.getOrUpload(bTensor)
	if dB == 0 {
		return MatMul(a, bTensor)
	}
	if freeB {
		defer b.fnCudaFree(dB)
	}

	// Allocate output (always temporary).
	sC := uintptr(M * N * 4)
	var dC uintptr
	if b.fnCudaMalloc(&dC, sC) != 0 {
		return MatMul(a, bTensor)
	}
	defer b.fnCudaFree(dC)

	// cuBLAS is column-major. For row-major: swap A/B.
	alpha, beta := float32(1.0), float32(0.0)
	b.fnCublasSgemm(b.handle,
		int32(cublasOpN), int32(cublasOpN),
		int32(N), int32(M), int32(K),
		&alpha, dB, int32(N), dA, int32(K),
		&beta, dC, int32(N),
	)

	result := NewTensor([]int{M, N})
	b.fnCudaMemcpy(uintptr(unsafe.Pointer(&result.Data[0])), dC, sC, cudaMemcpyD2H)
	return result
}

// MatVecMulGPU performs out = A * v using cuBLAS sgemv.
// Uses cached device pointer for A (weight matrix) when available.
func (b *GPUBackend) MatVecMulGPU(a, v *Tensor) *Tensor {
	if !b.available {
		return MatVecMul(a, v)
	}

	M, K := a.Shape[0], a.Shape[1]

	b.mu.Lock()
	defer b.mu.Unlock()

	// Get or upload A (matrix - often a cached weight).
	dA, freeA := b.getOrUpload(a)
	if dA == 0 {
		return MatVecMul(a, v)
	}
	if freeA {
		defer b.fnCudaFree(dA)
	}

	// Upload v (vector - always an activation).
	sV := uintptr(K * 4)
	var dV uintptr
	if b.fnCudaMalloc(&dV, sV) != 0 {
		return MatVecMul(a, v)
	}
	defer b.fnCudaFree(dV)
	b.fnCudaMemcpy(dV, uintptr(unsafe.Pointer(&v.Data[0])), sV, cudaMemcpyH2D)

	// Output.
	sO := uintptr(M * 4)
	var dOut uintptr
	if b.fnCudaMalloc(&dOut, sO) != 0 {
		return MatVecMul(a, v)
	}
	defer b.fnCudaFree(dOut)

	alpha, beta := float32(1.0), float32(0.0)
	b.fnCublasSgemv(b.handle,
		int32(cublasOpT), int32(K), int32(M),
		&alpha, dA, int32(K),
		dV, 1,
		&beta, dOut, 1,
	)

	result := NewTensor([]int{M})
	b.fnCudaMemcpy(uintptr(unsafe.Pointer(&result.Data[0])), dOut, sO, cudaMemcpyD2H)
	return result
}

// ---------- Element-wise Ops (CPU fallback, memory-bound) ----------

func (b *GPUBackend) SiLUMulGPU(gate, up *Tensor) *Tensor          { return Mul(SiLU(gate), up) }
func (b *GPUBackend) SiLUGPU(t *Tensor) *Tensor                     { return SiLU(t) }
func (b *GPUBackend) AddGPU(a, bT *Tensor) *Tensor                  { return Add(a, bT) }
func (b *GPUBackend) MulGPU(a, bT *Tensor) *Tensor                  { return Mul(a, bT) }
func (b *GPUBackend) RMSNormGPU(in, w *Tensor, eps float32) *Tensor { return RMSNorm(in, w, eps) }
func (b *GPUBackend) SoftmaxGPU(t *Tensor) *Tensor                  { return Softmax(t) }

// Close releases CUDA resources and frees all cached GPU buffers.
func (b *GPUBackend) Close() {
	b.mu.Lock()
	// Free all cached weight buffers.
	for key, buf := range b.weightCache {
		b.fnCudaFree(buf.ptr)
		delete(b.weightCache, key)
	}
	b.mu.Unlock()

	if b.handle != 0 && b.fnCublasDestroy != nil {
		b.fnCublasDestroy(b.handle)
		b.handle = 0
	}
	// purego doesn't require explicit library close.
	b.available = false
}

// ---------- CUDA Streams (async pipelined execution) ----------
//
// Without streams, CUDA operations are synchronous:
//   H2D copy -> compute -> D2H copy (each waits for the previous)
// With streams, we can overlap these stages:
//   Stream 1: H2D copy for next batch
//   Stream 2: compute on current batch
//   Stream 3: D2H copy for previous batch
//
// This hides transfer latency and can improve throughput by 30-50%.

// CUDAStream wraps a CUDA stream handle for async operations.
type CUDAStream struct {
	handle uintptr // cudaStream_t

	// Function pointers for stream operations.
	fnStreamCreate  func(stream *uintptr, flags uint32) int32
	fnStreamDestroy func(stream uintptr) int32
	fnStreamSync    func(stream uintptr) int32
	fnMemcpyAsync   func(dst, src uintptr, count uintptr, kind int32, stream uintptr) int32
}

// NewCUDAStream creates a new async CUDA stream.
// The cudart library handle must have been previously loaded.
func NewCUDAStream(cudartLib uintptr) (*CUDAStream, error) {
	s := &CUDAStream{}

	purego.RegisterLibFunc(&s.fnStreamCreate, cudartLib, "cudaStreamCreate")
	purego.RegisterLibFunc(&s.fnStreamDestroy, cudartLib, "cudaStreamDestroy")
	purego.RegisterLibFunc(&s.fnStreamSync, cudartLib, "cudaStreamSynchronize")
	purego.RegisterLibFunc(&s.fnMemcpyAsync, cudartLib, "cudaMemcpyAsync")

	if s.fnStreamCreate == nil {
		return nil, fmt.Errorf("cudaStreamCreate not found")
	}

	if ret := s.fnStreamCreate(&s.handle, 0); ret != 0 {
		return nil, fmt.Errorf("cudaStreamCreate failed: %d", ret)
	}

	return s, nil
}

// AsyncH2D copies data from host to device asynchronously on this stream.
func (s *CUDAStream) AsyncH2D(dstDevice uintptr, srcHost []float32) {
	if s.fnMemcpyAsync == nil || len(srcHost) == 0 {
		return
	}
	size := uintptr(len(srcHost) * 4)
	s.fnMemcpyAsync(dstDevice, uintptr(unsafe.Pointer(&srcHost[0])), size, cudaMemcpyH2D, s.handle)
}

// AsyncD2H copies data from device to host asynchronously on this stream.
func (s *CUDAStream) AsyncD2H(dstHost []float32, srcDevice uintptr) {
	if s.fnMemcpyAsync == nil || len(dstHost) == 0 {
		return
	}
	size := uintptr(len(dstHost) * 4)
	s.fnMemcpyAsync(uintptr(unsafe.Pointer(&dstHost[0])), srcDevice, size, cudaMemcpyD2H, s.handle)
}

// Sync waits for all operations on this stream to complete.
func (s *CUDAStream) Sync() {
	if s.fnStreamSync != nil {
		s.fnStreamSync(s.handle)
	}
}

// Close destroys the CUDA stream.
func (s *CUDAStream) Close() {
	if s.fnStreamDestroy != nil && s.handle != 0 {
		s.fnStreamDestroy(s.handle)
		s.handle = 0
	}
}

// StreamPool manages a set of CUDA streams for pipelined operations.
type StreamPool struct {
	streams []*CUDAStream
	current int
}

// NewStreamPool creates N async CUDA streams for pipelined execution.
func NewStreamPool(cudartLib uintptr, count int) *StreamPool {
	pool := &StreamPool{
		streams: make([]*CUDAStream, 0, count),
	}

	for i := 0; i < count; i++ {
		stream, err := NewCUDAStream(cudartLib)
		if err != nil {
			fmt.Printf("[CUDA] Failed to create stream %d: %v\n", i, err)
			break
		}
		pool.streams = append(pool.streams, stream)
	}

	if len(pool.streams) > 0 {
		fmt.Printf("[CUDA] Created %d async streams for pipelined execution\n", len(pool.streams))
	}
	return pool
}

// Next returns the next stream in round-robin order.
func (p *StreamPool) Next() *CUDAStream {
	if len(p.streams) == 0 {
		return nil
	}
	s := p.streams[p.current]
	p.current = (p.current + 1) % len(p.streams)
	return s
}

// SyncAll waits for all streams to complete.
func (p *StreamPool) SyncAll() {
	for _, s := range p.streams {
		s.Sync()
	}
}

// Close destroys all streams.
func (p *StreamPool) Close() {
	for _, s := range p.streams {
		s.Close()
	}
	p.streams = nil
}

