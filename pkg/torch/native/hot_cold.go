// hot_cold.go implements neuron profiling and hot/cold classification
// for PowerInfer-style sparse inference.
//
// The key insight from PowerInfer: in ReLU/SiLU-activated LLMs, neuron
// activations follow a power-law distribution. A small percentage of
// neurons ("hot") fire on nearly every input, while the vast majority
// ("cold") fire rarely and can be computed on-demand.
//
// This module provides:
//  1. Profiling: run calibration text through the model, track activations
//  2. Classification: split neurons into hot/cold sets per layer
//  3. Placement: decide where to compute each neuron (GPU vs CPU vs iGPU)
package native

import (
	"fmt"
	"sort"
)

// NeuronProfile holds activation statistics for all layers.
type NeuronProfile struct {
	// Per-layer activation counts: [layer][neuron] = count of activations.
	LayerActivations [][]int64

	// Total number of samples (forward passes) used for profiling.
	TotalSamples int64

	// FFN dimensions per layer.
	FFNDims []int
}

// NeuronPlacement describes where a neuron should be computed.
type NeuronPlacement int

const (
	PlacementGPU  NeuronPlacement = iota // dGPU VRAM: hot neurons, always preloaded
	PlacementIGPU                        // iGPU shared memory: warm neurons, fast access
	PlacementCPU                         // System RAM: cold neurons, computed on-demand
)

// LayerClassification holds the hot/cold split for one layer.
type LayerClassification struct {
	LayerIdx     int
	HotNeurons   []int           // Always compute (GPU)
	WarmNeurons  []int           // Pre-loaded but not always computed (iGPU)
	ColdNeurons  []int           // Compute on-demand (CPU)
	Placements   []NeuronPlacement // Per-neuron placement decisions
	HotFraction  float32         // What fraction of total neurons are hot
	WarmFraction float32
}

// NewNeuronProfile creates an empty profile for a model.
func NewNeuronProfile(numLayers int, ffnDims []int) *NeuronProfile {
	activations := make([][]int64, numLayers)
	for i := 0; i < numLayers; i++ {
		dim := ffnDims[i]
		if i >= len(ffnDims) {
			dim = ffnDims[len(ffnDims)-1]
		}
		activations[i] = make([]int64, dim)
	}
	return &NeuronProfile{
		LayerActivations: activations,
		FFNDims:          ffnDims,
	}
}

// RecordActivation records which neurons fired in a given layer.
func (p *NeuronProfile) RecordActivation(layerIdx int, activeNeurons []int) {
	if layerIdx >= len(p.LayerActivations) {
		return
	}
	for _, n := range activeNeurons {
		if n < len(p.LayerActivations[layerIdx]) {
			p.LayerActivations[layerIdx][n]++
		}
	}
}

// IncrementSamples marks one forward pass completed.
func (p *NeuronProfile) IncrementSamples() {
	p.TotalSamples++
}

// ClassifyLayer splits neurons into hot/warm/cold for a single layer.
//
// Thresholds:
//   - Hot: fires on > hotThreshold fraction of inputs (e.g., 0.8 = 80%)
//   - Warm: fires on > warmThreshold but <= hotThreshold
//   - Cold: fires on <= warmThreshold
func (p *NeuronProfile) ClassifyLayer(layerIdx int, hotThreshold, warmThreshold float32) LayerClassification {
	if layerIdx >= len(p.LayerActivations) || p.TotalSamples == 0 {
		return LayerClassification{LayerIdx: layerIdx}
	}

	counts := p.LayerActivations[layerIdx]
	total := len(counts)
	placements := make([]NeuronPlacement, total)

	var hot, warm, cold []int

	for i, count := range counts {
		rate := float32(count) / float32(p.TotalSamples)
		if rate > hotThreshold {
			hot = append(hot, i)
			placements[i] = PlacementGPU
		} else if rate > warmThreshold {
			warm = append(warm, i)
			placements[i] = PlacementIGPU
		} else {
			cold = append(cold, i)
			placements[i] = PlacementCPU
		}
	}

	return LayerClassification{
		LayerIdx:     layerIdx,
		HotNeurons:   hot,
		WarmNeurons:  warm,
		ColdNeurons:  cold,
		Placements:   placements,
		HotFraction:  float32(len(hot)) / float32(total),
		WarmFraction: float32(len(warm)) / float32(total),
	}
}

// ClassifyAll classifies all layers and returns a summary.
func (p *NeuronProfile) ClassifyAll(hotThreshold, warmThreshold float32) []LayerClassification {
	results := make([]LayerClassification, len(p.LayerActivations))
	for i := range p.LayerActivations {
		results[i] = p.ClassifyLayer(i, hotThreshold, warmThreshold)
	}
	return results
}

// PrintSummary prints a human-readable summary of the profiling results.
func (p *NeuronProfile) PrintSummary(hotThreshold, warmThreshold float32) {
	if p.TotalSamples == 0 {
		fmt.Println("[GOTensor] No profiling data collected yet.")
		return
	}

	fmt.Printf("[GOTensor] Neuron Profile: %d samples across %d layers\n",
		p.TotalSamples, len(p.LayerActivations))

	classifications := p.ClassifyAll(hotThreshold, warmThreshold)
	totalHot, totalWarm, totalCold, totalNeurons := 0, 0, 0, 0
	for _, c := range classifications {
		totalHot += len(c.HotNeurons)
		totalWarm += len(c.WarmNeurons)
		totalCold += len(c.ColdNeurons)
		totalNeurons += len(c.HotNeurons) + len(c.WarmNeurons) + len(c.ColdNeurons)
	}

	fmt.Printf("  Hot (GPU):   %5d neurons (%.1f%%)\n", totalHot, 100*float32(totalHot)/float32(totalNeurons))
	fmt.Printf("  Warm (iGPU): %5d neurons (%.1f%%)\n", totalWarm, 100*float32(totalWarm)/float32(totalNeurons))
	fmt.Printf("  Cold (CPU):  %5d neurons (%.1f%%)\n", totalCold, 100*float32(totalCold)/float32(totalNeurons))
	fmt.Printf("  Potential speedup: %.1fx (1 / %.1f%% active)\n",
		1.0/((float32(totalHot)+float32(totalWarm))/float32(totalNeurons)),
		100*(float32(totalHot)+float32(totalWarm))/float32(totalNeurons))
}

// TopActivatedNeurons returns the N most frequently activated neurons across all layers.
func (p *NeuronProfile) TopActivatedNeurons(n int) []struct {
	Layer  int
	Neuron int
	Rate   float32
} {
	type entry struct {
		layer  int
		neuron int
		rate   float32
	}

	var all []entry
	for l, counts := range p.LayerActivations {
		for n, c := range counts {
			if c > 0 {
				all = append(all, entry{l, n, float32(c) / float32(p.TotalSamples)})
			}
		}
	}

	sort.Slice(all, func(i, j int) bool {
		return all[i].rate > all[j].rate
	})

	if n > len(all) {
		n = len(all)
	}

	result := make([]struct {
		Layer  int
		Neuron int
		Rate   float32
	}, n)
	for i := 0; i < n; i++ {
		result[i].Layer = all[i].layer
		result[i].Neuron = all[i].neuron
		result[i].Rate = all[i].rate
	}
	return result
}
