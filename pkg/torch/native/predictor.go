// predictor.go implements the neuron activation predictor for PowerInfer-style
// sparse inference. The predictor determines which neurons will be active
// BEFORE computing the full FFN, enabling massive speedups by skipping
// inactive neurons entirely.
//
// Strategy tiers:
//  1. Static magnitude predictor (no extra weights, uses gate weight norms)
//  2. Learned predictor (small [hidden, ffn] weight matrix, loaded from GGUF)
//  3. Online profiling (tracks activation patterns during inference)
package native

import "math"

// NeuronPredictor predicts which FFN neurons will activate for a given input.
type NeuronPredictor struct {
	// LayerIdx identifies which transformer layer this predictor belongs to.
	LayerIdx int

	// Sparsity target: fraction of neurons to skip (0.0 = none, 0.9 = skip 90%).
	Sparsity float32

	// StaticHotNeurons: indices of neurons that are always active (from profiling).
	// These are computed once and cached.
	StaticHotNeurons []int

	// StaticColdNeurons: indices of neurons that are rarely active.
	StaticColdNeurons []int

	// GateMagnitudes: pre-computed L2 norm of each gate weight row.
	// Higher magnitude = more likely to activate. Used as a static proxy
	// when no learned predictor is available.
	GateMagnitudes []float32

	// ActivationCounts: running count of how often each neuron fires.
	// Used for online profiling to build hot/cold sets.
	ActivationCounts []int64
	TotalSamples     int64
}

// NewMagnitudePredictor creates a predictor from gate weight magnitudes.
// This is the zero-cost fallback: no extra model weights needed,
// just compute L2 norms of the gate matrix rows offline.
func NewMagnitudePredictor(gateWeight *Tensor, layerIdx int, sparsity float32) *NeuronPredictor {
	if len(gateWeight.Shape) != 2 {
		return &NeuronPredictor{LayerIdx: layerIdx, Sparsity: sparsity}
	}

	ffnDim := gateWeight.Shape[0]
	hiddenDim := gateWeight.Shape[1]

	// Compute L2 norm of each row (each neuron's gate weights).
	magnitudes := make([]float32, ffnDim)
	for i := 0; i < ffnDim; i++ {
		var sumSq float64
		for j := 0; j < hiddenDim; j++ {
			v := float64(gateWeight.Data[i*hiddenDim+j])
			sumSq += v * v
		}
		magnitudes[i] = float32(math.Sqrt(sumSq))
	}

	pred := &NeuronPredictor{
		LayerIdx:         layerIdx,
		Sparsity:         sparsity,
		GateMagnitudes:   magnitudes,
		ActivationCounts: make([]int64, ffnDim),
	}

	// Pre-classify neurons by magnitude.
	pred.classifyByMagnitude()
	return pred
}

// Predict returns the indices of neurons predicted to activate.
// Uses gate magnitudes as a proxy: neurons with higher gate weight norms
// are more likely to fire for any input.
func (p *NeuronPredictor) Predict(hiddenState *Tensor) []int {
	if len(p.StaticHotNeurons) > 0 {
		return p.StaticHotNeurons
	}

	// Fallback: return all neurons (no sparsity).
	total := len(p.GateMagnitudes)
	if total == 0 {
		return nil
	}
	keep := int(float32(total) * (1.0 - p.Sparsity))
	if keep <= 0 {
		keep = 1
	}
	return TopKIndices(p.GateMagnitudes, keep)
}

// PredictDynamic uses the actual gate activation values to decide
// which neurons to compute. This is more accurate than static prediction
// but requires computing the gate projection first.
func (p *NeuronPredictor) PredictDynamic(gateActivations []float32, upActivations []float32) []int {
	total := len(gateActivations)
	keep := int(float32(total) * (1.0 - p.Sparsity))
	if keep <= 0 {
		keep = 1
	}

	// Use the actual gate*sigmoid values to find top-K active neurons.
	activated := make([]float32, total)
	for i, g := range gateActivations {
		// SiLU(g) magnitude.
		sigmoid := float32(1.0 / (1.0 + exp64(-float64(g))))
		val := g * sigmoid
		if val < 0 {
			val = -val
		}
		activated[i] = val
	}

	return TopKIndices(activated, keep)
}

// RecordActivation tracks which neurons fired (for online profiling).
func (p *NeuronPredictor) RecordActivation(activeIndices []int) {
	p.TotalSamples++
	for _, idx := range activeIndices {
		if idx < len(p.ActivationCounts) {
			p.ActivationCounts[idx]++
		}
	}
}

// classifyByMagnitude splits neurons into hot/cold sets based on gate magnitude.
func (p *NeuronPredictor) classifyByMagnitude() {
	total := len(p.GateMagnitudes)
	if total == 0 {
		return
	}

	keep := int(float32(total) * (1.0 - p.Sparsity))
	if keep <= 0 {
		keep = 1
	}

	p.StaticHotNeurons = TopKIndices(p.GateMagnitudes, keep)

	// Cold = everything not hot.
	hotSet := make(map[int]bool, len(p.StaticHotNeurons))
	for _, idx := range p.StaticHotNeurons {
		hotSet[idx] = true
	}
	p.StaticColdNeurons = make([]int, 0, total-keep)
	for i := 0; i < total; i++ {
		if !hotSet[i] {
			p.StaticColdNeurons = append(p.StaticColdNeurons, i)
		}
	}
}

// ReclassifyFromProfile rebuilds hot/cold sets from observed activation counts.
// Call after running calibration data through the model.
func (p *NeuronPredictor) ReclassifyFromProfile() {
	if p.TotalSamples == 0 {
		return
	}

	total := len(p.ActivationCounts)
	keep := int(float32(total) * (1.0 - p.Sparsity))

	// Convert counts to activation rates.
	rates := make([]float32, total)
	for i, count := range p.ActivationCounts {
		rates[i] = float32(count) / float32(p.TotalSamples)
	}

	p.StaticHotNeurons = TopKIndices(rates, keep)

	hotSet := make(map[int]bool, len(p.StaticHotNeurons))
	for _, idx := range p.StaticHotNeurons {
		hotSet[idx] = true
	}
	p.StaticColdNeurons = make([]int, 0, total-keep)
	for i := 0; i < total; i++ {
		if !hotSet[i] {
			p.StaticColdNeurons = append(p.StaticColdNeurons, i)
		}
	}
}
