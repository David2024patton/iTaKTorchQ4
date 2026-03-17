package llama

import (
	"github.com/David2024patton/iTaKTorch/pkg/torch/utils"
	"github.com/ebitengine/purego"
)

// purego direct-call function pointers (zero FFI overhead)
var (
	backendInitFn       func()
	backendFreeFn       func()
	numaInitFn          func(numaStrategy int32)
	maxDevicesFn        func() uint64
	maxParallelSeqFn    func() uint64
	maxTensorBuftOvFn   func() uint64
	supportsMmapFn      func() uint8
	supportsMlockFn     func() uint8
	supportsGpuOffFn    func() uint8
	supportsRpcFn       func() uint8
	timeUsFn            func() int64
	flashAttnTypeNameFn func(flashAttnType int32) *byte
	printSystemInfoFn   func() *byte
)

// Typed Go function variable for optional ggml_backend_load_all
var (
	backendLoadAllFn  func()
	hasBackendLoadAll bool
)

func loadBackendPuregoFuncs(lib uintptr) {
	purego.RegisterLibFunc(&backendInitFn, lib, "llama_backend_init")
	purego.RegisterLibFunc(&backendFreeFn, lib, "llama_backend_free")
	purego.RegisterLibFunc(&numaInitFn, lib, "llama_numa_init")
	purego.RegisterLibFunc(&maxDevicesFn, lib, "llama_max_devices")
	purego.RegisterLibFunc(&maxParallelSeqFn, lib, "llama_max_parallel_sequences")
	purego.RegisterLibFunc(&maxTensorBuftOvFn, lib, "llama_max_tensor_buft_overrides")
	purego.RegisterLibFunc(&supportsMmapFn, lib, "llama_supports_mmap")
	purego.RegisterLibFunc(&supportsMlockFn, lib, "llama_supports_mlock")
	purego.RegisterLibFunc(&supportsGpuOffFn, lib, "llama_supports_gpu_offload")
	purego.RegisterLibFunc(&supportsRpcFn, lib, "llama_supports_rpc")
	purego.RegisterLibFunc(&timeUsFn, lib, "llama_time_us")
	purego.RegisterLibFunc(&flashAttnTypeNameFn, lib, "llama_flash_attn_type_name")
	purego.RegisterLibFunc(&printSystemInfoFn, lib, "llama_print_system_info")
}

// BackendInit initializes the llama.cpp back-end.
func BackendInit() {
	backendInitFn()
}

// BackendFree frees the llama.cpp back-end.
func BackendFree() {
	backendFreeFn()
}

// NumaInit initializes NUMA with the given strategy.
func NumaInit(numaStrategy NumaStrategy) {
	numaInitFn(int32(numaStrategy))
}

// MaxDevices returns the maximum number of devices supported.
func MaxDevices() uint64 {
	return maxDevicesFn()
}

// MaxParallelSequences returns the maximum number of parallel sequences supported.
func MaxParallelSequences() uint64 {
	return maxParallelSeqFn()
}

// MaxTensorBuftOverrides returns the maximum number of tensor buffer overrides supported.
func MaxTensorBuftOverrides() uint64 {
	return maxTensorBuftOvFn()
}

// SupportsMmap checks if memory-mapped files are supported.
func SupportsMmap() bool {
	return supportsMmapFn() != 0
}

// SupportsMlock checks if memory locking is supported.
func SupportsMlock() bool {
	return supportsMlockFn() != 0
}

// SupportsGpuOffload checks if GPU offloading is supported.
func SupportsGpuOffload() bool {
	return supportsGpuOffFn() != 0
}

// SupportsRpc checks if RPC is supported.
func SupportsRpc() bool {
	return supportsRpcFn() != 0
}

// TimeUs returns the current time in microseconds.
func TimeUs() int64 {
	return timeUsFn()
}

// FlashAttnTypeName returns the name for a given flash attention type.
func FlashAttnTypeName(flashAttnType FlashAttentionType) string {
	result := flashAttnTypeNameFn(int32(flashAttnType))
	if result == nil {
		return ""
	}
	return utils.BytePtrToString(result)
}

// PrintSystemInfo returns system information as a string.
func PrintSystemInfo() string {
	result := printSystemInfoFn()
	if result == nil {
		return ""
	}
	return utils.BytePtrToString(result)
}

// BackendLoadAll loads all available backends (CPU, CUDA, etc.).
func BackendLoadAll() {
	if hasBackendLoadAll {
		backendLoadAllFn()
	}
}
