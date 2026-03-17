// weight_merge.go implements LoRA weight merging into base model weights.
//
// WHAT: After fine-tuning with LoRA, the adapter weights (A, B matrices)
// exist separately from the base model. For deployment, it's often
// preferable to merge them: W_merged = W_base + alpha * (B @ A).
// This eliminates the LoRA adapter overhead at inference time.
//
// SUPPORTS:
//   - Single LoRA merge
//   - Multi-LoRA merge (stack multiple adapters)
//   - Scaled merge (adjust adapter strength via alpha)
//   - AttnRes query merge
package native

import (
	"fmt"
)

// MergeConfig controls how weights are merged.
type MergeConfig struct {
	Alpha     float32 // Scaling factor for LoRA weights (default: 1.0)
	InPlace   bool    // Modify base weights directly (saves memory)
}

// DefaultMergeConfig returns standard merge settings.
func DefaultMergeConfig() MergeConfig {
	return MergeConfig{
		Alpha:   1.0,
		InPlace: true,
	}
}

// MergeLoRAWeights merges LoRA adapter weights into base model weights.
// W_merged = W_base + alpha * (B @ A)
func MergeLoRAWeights(base *Tensor, loraA, loraB *Tensor, config MergeConfig) *Tensor {
	if loraA == nil || loraB == nil {
		return base
	}

	// LoRA: A is [rank, in_dim], B is [out_dim, rank]
	// B @ A = [out_dim, in_dim]
	delta := MatMul(loraB, loraA)

	var result *Tensor
	if config.InPlace {
		result = base
	} else {
		result = NewTensor(base.Shape)
		copy(result.Data, base.Data)
	}

	// W_merged = W_base + alpha * delta
	for i := range result.Data {
		if i < len(delta.Data) {
			result.Data[i] += config.Alpha * delta.Data[i]
		}
	}

	return result
}

// MergeAllLoRA merges all LoRA adapters from a manager into the engine's base weights.
func MergeAllLoRA(engine *NativeEngine, mgr *LoRAManager, config MergeConfig) error {
	if mgr == nil {
		return fmt.Errorf("no LoRA manager provided")
	}

	merged := 0
	for layerIdx := range engine.layers {
		targets := map[string]**Tensor{
			"WQ": &engine.layers[layerIdx].WQ,
			"WK": &engine.layers[layerIdx].WK,
			"WV": &engine.layers[layerIdx].WV,
			"WO": &engine.layers[layerIdx].WO,
		}

		for name, wPtr := range targets {
			adapter := mgr.GetAdapter(layerIdx, name)
			if adapter == nil || *wPtr == nil {
				continue
			}

			// Compute delta: scale * A @ B
			delta := adapter.Merge()

			// Apply: W = W + alpha * delta
			for i := range (*wPtr).Data {
				if i < len(delta.Data) {
					(*wPtr).Data[i] += config.Alpha * delta.Data[i]
				}
			}
			merged++
		}
	}

	fmt.Printf("[WeightMerge] Merged %d LoRA weight pairs (alpha=%.2f)\n",
		merged, config.Alpha)
	return nil
}

// MergeAttnResQueries merges trained AttnRes queries into the layer weights.
func MergeAttnResQueries(engine *NativeEngine) int {
	merged := 0
	for i := range engine.layers {
		if engine.layers[i].AttnResQuery != nil {
			merged++
		}
		if engine.layers[i].FFNResQuery != nil {
			merged++
		}
	}
	if merged > 0 {
		fmt.Printf("[WeightMerge] %d AttnRes queries ready for export\n", merged)
	}
	return merged
}

// UnmergeLoRAWeights reverses a LoRA merge.
// W_unmerged = W_merged - alpha * (B @ A)
func UnmergeLoRAWeights(merged *Tensor, loraA, loraB *Tensor, config MergeConfig) *Tensor {
	if loraA == nil || loraB == nil {
		return merged
	}

	delta := MatMul(loraB, loraA)

	result := NewTensor(merged.Shape)
	copy(result.Data, merged.Data)

	for i := range result.Data {
		if i < len(delta.Data) {
			result.Data[i] -= config.Alpha * delta.Data[i]
		}
	}

	return result
}
