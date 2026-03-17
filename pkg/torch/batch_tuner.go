// batch_tuner.go auto-tunes batch size based on available VRAM and model size.
//
// WHY: Batch size directly controls GPU utilization during prompt processing.
// Too small = GPU pipeline bubbles (GPU waits for each tiny batch).
// Too large = OOM or excessive VRAM for compute buffers.
//
// The optimal batch size depends on:
//   - Available VRAM after model loading
//   - Context size (larger ctx = more KV cache = less room for batches)
//   - GPU architecture (consumer cards handle 512-2048, server GPUs 2048-8192)
//
// This auto-tuner sizes the batch to fill GPU pipelines without OOM risk.
package torch

import "fmt"

// TuneBatchSize returns an optimized batch size based on available VRAM and context.
// modelSizeMB is the model file size in MB.
// modelFitsGPU is whether the model fits entirely in GPU VRAM.
// vramMiB is the GPU VRAM in MiB (0 = CPU-only).
// contextSize is the configured context window (0 = unset).
func TuneBatchSize(modelSizeMB int64, modelFitsGPU bool, vramMiB int64, contextSize int) int {
	// CPU-only: moderate batch size (CPU can't handle huge batches efficiently).
	if !modelFitsGPU || vramMiB == 0 {
		batch := 512
		fmt.Printf("[iTaK Torch] Batch size: %d (CPU-only mode)\n", batch)
		return batch
	}

	// Estimate remaining VRAM after model + KV cache.
	// Model VRAM ~ 1.1x file size (weights + compute buffers).
	// KV cache VRAM ~ ctx_size * 2 * n_layers * d_head * 2 bytes (q8_0).
	// Simplified: remaining = total - model - kv_estimate.
	modelVRAM := int64(float64(modelSizeMB) * 1.1)
	kvEstimate := int64(contextSize) * 256 / 1024 // rough estimate in MB
	remainingMB := vramMiB - modelVRAM - kvEstimate

	var batch int
	switch {
	case remainingMB > 4096:
		// Server GPU with lots of headroom (A100, H100, 4090).
		batch = 4096
	case remainingMB > 2048:
		// High-end consumer GPU (4080, 4070 Ti Super).
		batch = 2048
	case remainingMB > 1024:
		// Mid-range GPU (4060 Ti, 3070).
		batch = 1024
	case remainingMB > 512:
		// Lower-end GPU or large model.
		batch = 512
	default:
		// Tight VRAM - keep batch small.
		batch = 256
	}

	fmt.Printf("[iTaK Torch] Batch size: %d (auto-tuned, ~%dMB VRAM headroom)\n", batch, remainingMB)
	return batch
}

// TuneUBatchSize returns the physical ubatch size.
// The ubatch is the actual computation chunk sent to the GPU per iteration.
// For most models, ubatch = batch is optimal since the GPU processes it in one go.
// For MoE (Mixture of Experts) models, smaller ubatch helps with expert routing.
func TuneUBatchSize(batch int, isMoE bool) int {
	if isMoE {
		// MoE models benefit from smaller ubatch due to expert routing overhead.
		ubatch := batch / 4
		if ubatch < 128 {
			ubatch = 128
		}
		return ubatch
	}
	// Dense models: ubatch = batch (process everything at once).
	return batch
}
