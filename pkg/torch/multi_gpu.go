// multi_gpu.go provides multi-GPU tensor parallelism and layer splitting.
//
// HOW IT WORKS:
//   Layer split (SplitModeLayer):
//     Layers 0-15 on GPU 0, Layers 16-31 on GPU 1.
//     Simple, works on any multi-GPU setup. Sequential execution.
//
//   Row split / Tensor Parallelism (SplitModeRow):
//     Each layer's weight matrix is SPLIT across GPUs.
//     Both GPUs compute simultaneously, then sync results.
//     Requires NCCL or equivalent for inter-GPU communication.
//     3-4x speedup on 4 GPUs vs single GPU.
//
// WHEN TO USE:
//   - Model too large for single GPU (e.g. 70B on 2x 24GB GPUs)
//   - Want maximum throughput with multiple identical GPUs
//   - Layer split: any GPU combo. Row split: same-architecture GPUs only.
//
// WHEN NOT TO USE:
//   - PCIe bandwidth bottleneck (row split needs fast interconnect)
//   - Mixed GPU architectures (row split may hang or crash)
//   - Models that fit on a single GPU (overhead negates benefit)
package torch

import (
	"fmt"
	"strings"

	"github.com/David2024patton/iTaKTorch/pkg/torch/llama"
)

// MultiGPUConfig holds the multi-GPU splitting configuration.
type MultiGPUConfig struct {
	// SplitMode determines how the model is distributed.
	// "none" = single GPU, "layer" = layer split, "row" = tensor parallel.
	SplitMode string `json:"split_mode"`

	// MainGPU is the primary GPU index for compute scheduling.
	MainGPU int `json:"main_gpu"`

	// TensorSplit defines the proportion of model layers on each GPU.
	// Example: [0.5, 0.5] = 50% on GPU 0, 50% on GPU 1.
	// Example: [0.7, 0.3] = 70% on GPU 0, 30% on GPU 1.
	TensorSplit []float32 `json:"tensor_split,omitempty"`

	// AutoBalance auto-calculates TensorSplit based on VRAM ratios.
	AutoBalance bool `json:"auto_balance"`
}

// DetectMultiGPU probes for multiple GPUs and builds an optimal configuration.
// Returns nil if only one GPU (or no GPUs) are detected.
func DetectMultiGPU() *MultiGPUConfig {
	gpus := DetectGPUs()

	// Count discrete GPUs (ignore iGPUs).
	discreteGPUs := make([]GPUInfo, 0)
	for _, gpu := range gpus.GPUs {
		if !gpu.IsShared && gpu.VRAMMiB > 0 {
			discreteGPUs = append(discreteGPUs, gpu)
		}
	}

	if len(discreteGPUs) < 2 {
		return nil // single GPU or no GPU -- multi-GPU not applicable
	}

	config := &MultiGPUConfig{
		MainGPU:     0,
		AutoBalance: true,
	}

	// Check if GPUs are same architecture (required for row/tensor parallel).
	sameArch := true
	firstVendor := discreteGPUs[0].Vendor
	for _, gpu := range discreteGPUs[1:] {
		if gpu.Vendor != firstVendor {
			sameArch = false
			break
		}
	}

	if sameArch {
		// Same architecture: use row split (tensor parallel) for max speed.
		config.SplitMode = "row"
		fmt.Printf("[iTaK Torch] Multi-GPU: %d identical %s GPUs detected, using tensor parallelism\n",
			len(discreteGPUs), firstVendor)
	} else {
		// Mixed architectures: use layer split (safer, always works).
		config.SplitMode = "layer"
		fmt.Printf("[iTaK Torch] Multi-GPU: %d mixed GPUs detected, using layer split\n", len(discreteGPUs))
	}

	// Auto-balance: split proportionally to VRAM.
	totalVRAM := int64(0)
	for _, gpu := range discreteGPUs {
		totalVRAM += gpu.VRAMMiB
	}

	config.TensorSplit = make([]float32, len(discreteGPUs))
	for i, gpu := range discreteGPUs {
		config.TensorSplit[i] = float32(gpu.VRAMMiB) / float32(totalVRAM)
	}

	// Log the split.
	splitStrs := make([]string, len(config.TensorSplit))
	for i, s := range config.TensorSplit {
		splitStrs[i] = fmt.Sprintf("GPU%d=%.0f%%", discreteGPUs[i].Index, s*100)
	}
	fmt.Printf("[iTaK Torch] Tensor split: %s\n", strings.Join(splitStrs, ", "))

	return config
}

// ApplyMultiGPU applies multi-GPU configuration to ModelParams.
// Only modifies params if multi-GPU is configured and user hasn't overridden.
func ApplyMultiGPU(params *llama.ModelParams, config *MultiGPUConfig) {
	if config == nil {
		return
	}

	switch config.SplitMode {
	case "none":
		params.SplitMode = llama.SplitModeNone
	case "layer":
		params.SplitMode = llama.SplitModeLayer
	case "row":
		params.SplitMode = llama.SplitModeRow
	default:
		params.SplitMode = llama.SplitModeLayer // safe default
	}

	params.MainGpu = int32(config.MainGPU)

	// Set tensor split if provided.
	if len(config.TensorSplit) > 0 {
		params.TensorSplit = &config.TensorSplit[0]
	}
}

// MultiGPUStatus returns a human-readable summary of the multi-GPU config.
func MultiGPUStatus(config *MultiGPUConfig) string {
	if config == nil {
		return "single GPU (multi-GPU not active)"
	}
	splitStrs := make([]string, len(config.TensorSplit))
	for i, s := range config.TensorSplit {
		splitStrs[i] = fmt.Sprintf("GPU%d=%.0f%%", i, s*100)
	}
	return fmt.Sprintf("mode=%s main_gpu=%d split=[%s]",
		config.SplitMode, config.MainGPU, strings.Join(splitStrs, " "))
}
