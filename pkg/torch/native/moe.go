// moe.go implements Mixture of Experts routing for models like Mixtral and DeepSeek.
//
// WHAT: In MoE models, the FFN layer is replaced by multiple "expert" FFN
// networks. A gating network selects the top-K experts for each token.
// Only the selected experts run, so compute cost stays fixed even with
// many experts (e.g., 8 active out of 64 total).
//
// MODELS: Mixtral 8x7B (8 experts, top-2), DeepSeek-V2 (160 experts, top-6),
// Qwen-MoE, Grok, DBRX.
package native

import (
	"fmt"
	"math"
	"sort"
)

// MoEConfig holds Mixture of Experts parameters.
type MoEConfig struct {
	NumExperts    int // Total number of expert FFN networks
	NumActive     int // Number of experts activated per token (top-K)
	Enabled       bool
}

// MoELayer holds the weights for one MoE transformer layer.
type MoELayer struct {
	Gate    *Tensor   // Gating network: [hidden_dim, num_experts]
	Experts []FFNExpert // Each expert is a full FFN
}

// FFNExpert holds weights for one expert FFN network.
type FFNExpert struct {
	WGate *Tensor // [hidden_dim, ffn_dim]
	WUp   *Tensor // [hidden_dim, ffn_dim]
	WDown *Tensor // [ffn_dim, hidden_dim]
}

// MoEForward runs MoE routing and expert computation.
//
// x: [hidden_dim] - input hidden state for one token
// layer: MoE layer with gate + experts
// config: routing parameters
//
// Returns: [hidden_dim] - weighted sum of top-K expert outputs
func MoEForward(x *Tensor, layer *MoELayer, config MoEConfig) *Tensor {
	dim := len(x.Data)

	// Step 1: Compute gating logits: gate_logits = x @ Gate
	gateLogits := make([]float32, config.NumExperts)
	for e := 0; e < config.NumExperts; e++ {
		var dot float32
		for d := 0; d < dim; d++ {
			dot += x.Data[d] * layer.Gate.Data[d*config.NumExperts+e]
		}
		gateLogits[e] = dot
	}

	// Step 2: Softmax over gate logits.
	gateProbs := softmaxMoE(gateLogits)

	// Step 3: Select top-K experts.
	topK := selectTopK(gateProbs, config.NumActive)

	// Step 4: Run selected experts and combine outputs.
	output := NewTensor([]int{dim})
	var totalWeight float32

	for _, sel := range topK {
		expert := layer.Experts[sel.index]

		// FFN: gate-up-down pattern.
		gate := MatVecMul(expert.WGate, x)
		up := MatVecMul(expert.WUp, x)

		// SiLU(gate) * up
		gated := NewTensor([]int{len(gate.Data)})
		for i := range gate.Data {
			sig := float32(1.0 / (1.0 + math.Exp(-float64(gate.Data[i]))))
			gated.Data[i] = gate.Data[i] * sig * up.Data[i]
		}

		// Down projection.
		expertOut := MatVecMul(expert.WDown, gated)

		// Weighted accumulation.
		for d := 0; d < dim; d++ {
			output.Data[d] += sel.weight * expertOut.Data[d]
		}
		totalWeight += sel.weight
	}

	// Normalize by total weight.
	if totalWeight > 0 {
		invWeight := float32(1.0) / totalWeight
		for d := range output.Data {
			output.Data[d] *= invWeight
		}
	}

	return output
}

// expertSelection holds one selected expert with its routing weight.
type expertSelection struct {
	index  int
	weight float32
}

// selectTopK returns the top-K experts by weight.
func selectTopK(probs []float32, k int) []expertSelection {
	type indexedProb struct {
		idx  int
		prob float32
	}
	sorted := make([]indexedProb, len(probs))
	for i, p := range probs {
		sorted[i] = indexedProb{i, p}
	}
	sort.Slice(sorted, func(i, j int) bool {
		return sorted[i].prob > sorted[j].prob
	})

	if k > len(sorted) {
		k = len(sorted)
	}

	result := make([]expertSelection, k)
	for i := 0; i < k; i++ {
		result[i] = expertSelection{
			index:  sorted[i].idx,
			weight: sorted[i].prob,
		}
	}
	return result
}

func softmaxMoE(logits []float32) []float32 {
	probs := make([]float32, len(logits))
	maxVal := logits[0]
	for _, v := range logits[1:] {
		if v > maxVal {
			maxVal = v
		}
	}
	var sum float32
	for i, v := range logits {
		probs[i] = float32(math.Exp(float64(v - maxVal)))
		sum += probs[i]
	}
	if sum > 0 {
		for i := range probs {
			probs[i] /= sum
		}
	}
	return probs
}

// LoadMoEFromGGUF loads MoE layers from GGUF metadata.
func LoadMoEFromGGUF(gf *GGUFFile) ([]MoELayer, MoEConfig) {
	numExperts := int(gf.GetMetadataUint32("llama.expert_count"))
	numActive := int(gf.GetMetadataUint32("llama.expert_used_count"))

	if numExperts == 0 {
		return nil, MoEConfig{}
	}

	config := MoEConfig{
		NumExperts: numExperts,
		NumActive:  numActive,
		Enabled:    true,
	}

	fmt.Printf("[MoE] Detected %d experts, top-%d routing\n", numExperts, numActive)
	return nil, config // Actual weight loading handled by GGUF tensor mapper
}
