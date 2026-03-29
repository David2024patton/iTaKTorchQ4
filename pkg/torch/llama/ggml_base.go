package llama

import (
	"fmt"

	"github.com/David2024patton/iTaKTorchQ4/pkg/torch/utils"
	"github.com/ebitengine/purego"
)

// Opaque types (represented as pointers)
type GGMLBackendBufferType uintptr

// --- purego direct-call function pointers ---
var (
	ggmlBackendCpuBufferTypeFn func() GGMLBackendBufferType
	ggmlBackendDevNameFn       func(device GGMLBackendDevice) *byte
)

func loadGGMLBase(lib uintptr) error {
	purego.RegisterLibFunc(&ggmlBackendCpuBufferTypeFn, lib, "ggml_backend_cpu_buffer_type")
	purego.RegisterLibFunc(&ggmlBackendDevNameFn, lib, "ggml_backend_dev_name")
	return nil
}

// GGMLBackendCpuBufferType returns the buffer type used for CPU backends.
func GGMLBackendCpuBufferType() GGMLBackendBufferType {
	return ggmlBackendCpuBufferTypeFn()
}

const ffnExprsRegex = `\.ffn_(up|down|gate)_(ch|)exps`

func ffnExprBlockRegex(index int) string {
	return fmt.Sprintf("blk\\.%d%s", index, ffnExprsRegex)
}

// NewTensorBuftBlockOverride creates a TensorBuftOverride for a specific FFN block index to execute in the CPU.
func NewTensorBuftBlockOverride(index int) TensorBuftOverride {
	return NewTensorBuftOverride(ffnExprBlockRegex(index))
}

// NewTensorBuftAllFFNExprsOverride creates a TensorBuftOverride for all FFN expression tensors to execute in the CPU.
func NewTensorBuftAllFFNExprsOverride() TensorBuftOverride {
	return NewTensorBuftOverride(ffnExprsRegex)
}

// NewTensorBuftOverride creates a TensorBuftOverride for a custom pattern to execute in the CPU.
func NewTensorBuftOverride(pattern string) TensorBuftOverride {
	data, err := utils.BytePtrFromString(pattern)
	if err != nil {
		return TensorBuftOverride{}
	}
	return TensorBuftOverride{
		Pattern: data,
		Type:    GGMLBackendCpuBufferType(),
	}
}

// GGMLBackendDeviceName returns the name of the given backend device.
func GGMLBackendDeviceName(device GGMLBackendDevice) string {
	ret := ggmlBackendDevNameFn(device)
	return utils.BytePtrToString(ret)
}
