// logit_processor.go implements a composable pipeline for transforming logits
// before sampling. Each processor modifies logits in-place.
//
// WHAT: Before picking the next token, we often want to apply multiple
// transformations: temperature scaling, top-k filtering, repetition
// penalty, presence/frequency penalty, etc. This file provides a
// chain-of-responsibility pattern where processors are applied in order.
//
// USAGE:
//   chain := NewLogitChain(
//       TemperatureProcessor(0.7),
//       RepetitionPenaltyProcessor(1.1, history),
//       TopKProcessor(40),
//       TopPProcessor(0.9),
//       MinPProcessor(0.05),
//   )
//   chain.Process(logits)
package native

import (
	"math"
	"sort"
)

// LogitProcessor transforms a logit vector in-place.
type LogitProcessor interface {
	Process(logits []float32)
}

// LogitChain applies a sequence of processors in order.
type LogitChain struct {
	processors []LogitProcessor
}

// NewLogitChain creates a processor chain.
func NewLogitChain(processors ...LogitProcessor) *LogitChain {
	return &LogitChain{processors: processors}
}

// Process applies all processors in sequence.
func (c *LogitChain) Process(logits []float32) {
	for _, p := range c.processors {
		p.Process(logits)
	}
}

// Add appends a processor to the chain.
func (c *LogitChain) Add(p LogitProcessor) {
	c.processors = append(c.processors, p)
}

// ---------- Temperature ----------

type temperatureProc struct {
	temp float32
}

// TemperatureProcessor scales logits by 1/temperature.
// temp > 1.0 = more random, temp < 1.0 = more deterministic.
func TemperatureProcessor(temp float32) LogitProcessor {
	if temp <= 0 {
		temp = 1.0
	}
	return &temperatureProc{temp: temp}
}

func (p *temperatureProc) Process(logits []float32) {
	if p.temp == 1.0 {
		return
	}
	invTemp := 1.0 / p.temp
	for i := range logits {
		logits[i] *= invTemp
	}
}

// ---------- Top-K ----------

type topKProc struct {
	k int
}

// TopKProcessor keeps only the top-K logits, sets rest to -inf.
func TopKProcessor(k int) LogitProcessor {
	return &topKProc{k: k}
}

func (p *topKProc) Process(logits []float32) {
	if p.k <= 0 || p.k >= len(logits) {
		return
	}

	// Find the k-th largest value.
	sorted := make([]float32, len(logits))
	copy(sorted, logits)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i] > sorted[j] })
	threshold := sorted[p.k-1]

	negInf := float32(math.Inf(-1))
	for i := range logits {
		if logits[i] < threshold {
			logits[i] = negInf
		}
	}
}

// ---------- Top-P (Nucleus Sampling) ----------

type topPProc struct {
	p float32
}

// TopPProcessor keeps the smallest set of tokens whose cumulative probability >= p.
func TopPProcessor(p float32) LogitProcessor {
	return &topPProc{p: p}
}

func (p *topPProc) Process(logits []float32) {
	if p.p >= 1.0 {
		return
	}

	// Compute softmax probabilities.
	probs := softmaxFloat32(logits)

	// Sort indices by probability (descending).
	type indexProb struct {
		idx  int
		prob float32
	}
	sorted := make([]indexProb, len(probs))
	for i, prob := range probs {
		sorted[i] = indexProb{i, prob}
	}
	sort.Slice(sorted, func(i, j int) bool { return sorted[i].prob > sorted[j].prob })

	// Find cutoff.
	cumSum := float32(0)
	cutoffIdx := len(sorted)
	for i, sp := range sorted {
		cumSum += sp.prob
		if cumSum >= p.p {
			cutoffIdx = i + 1
			break
		}
	}

	// Zero out tokens below cutoff.
	keep := make(map[int]bool, cutoffIdx)
	for i := 0; i < cutoffIdx; i++ {
		keep[sorted[i].idx] = true
	}
	negInf := float32(math.Inf(-1))
	for i := range logits {
		if !keep[i] {
			logits[i] = negInf
		}
	}
}

// ---------- Min-P ----------

type minPProc struct {
	minP float32
}

// MinPProcessor filters tokens with probability < minP * max_prob.
// This adapts the cutoff based on the highest probability token.
func MinPProcessor(minP float32) LogitProcessor {
	return &minPProc{minP: minP}
}

func (p *minPProc) Process(logits []float32) {
	if p.minP <= 0 {
		return
	}

	probs := softmaxFloat32(logits)

	// Find max probability.
	maxProb := probs[0]
	for _, prob := range probs[1:] {
		if prob > maxProb {
			maxProb = prob
		}
	}

	threshold := p.minP * maxProb
	negInf := float32(math.Inf(-1))
	for i := range logits {
		if probs[i] < threshold {
			logits[i] = negInf
		}
	}
}

// ---------- Repetition Penalty ----------

type repPenaltyProc struct {
	penalty float32
	history []int
}

// RepetitionPenaltyProcessor penalizes tokens that appeared in the history.
// penalty > 1.0 reduces likelihood, < 1.0 increases it.
func RepetitionPenaltyProcessor(penalty float32, history []int) LogitProcessor {
	return &repPenaltyProc{penalty: penalty, history: history}
}

func (p *repPenaltyProc) Process(logits []float32) {
	if p.penalty == 1.0 {
		return
	}

	seen := make(map[int]bool, len(p.history))
	for _, tok := range p.history {
		seen[tok] = true
	}

	for tok := range seen {
		if tok < 0 || tok >= len(logits) {
			continue
		}
		if logits[tok] > 0 {
			logits[tok] /= p.penalty
		} else {
			logits[tok] *= p.penalty
		}
	}
}

// ---------- Presence / Frequency Penalty ----------

type presFreqProc struct {
	presencePenalty  float32
	frequencyPenalty float32
	history          []int
}

// PresenceFrequencyProcessor applies OpenAI-style presence and frequency penalties.
// Presence penalty: flat penalty for tokens that appeared at all.
// Frequency penalty: scales with how often the token appeared.
func PresenceFrequencyProcessor(presence, frequency float32, history []int) LogitProcessor {
	return &presFreqProc{
		presencePenalty:  presence,
		frequencyPenalty: frequency,
		history:          history,
	}
}

func (p *presFreqProc) Process(logits []float32) {
	if p.presencePenalty == 0 && p.frequencyPenalty == 0 {
		return
	}

	counts := make(map[int]int, len(p.history))
	for _, tok := range p.history {
		counts[tok]++
	}

	for tok, count := range counts {
		if tok < 0 || tok >= len(logits) {
			continue
		}
		logits[tok] -= p.presencePenalty + p.frequencyPenalty*float32(count)
	}
}

// ---------- Mirostat V2 ----------

type mirostatV2Proc struct {
	tau    float32 // Target entropy (typical: 5.0)
	eta    float32 // Learning rate (typical: 0.1)
	mu     float32 // Current estimate of surprise (starts at 2*tau)
}

// MirostatV2Processor implements Mirostat v2 adaptive sampling.
// Dynamically adjusts the top-k cutoff to maintain a target entropy.
func MirostatV2Processor(tau, eta float32) LogitProcessor {
	return &mirostatV2Proc{
		tau: tau,
		eta: eta,
		mu:  2.0 * tau,
	}
}

func (p *mirostatV2Proc) Process(logits []float32) {
	probs := softmaxFloat32(logits)

	// Sort by probability (descending).
	type indexProb struct {
		idx  int
		prob float32
	}
	sorted := make([]indexProb, len(probs))
	for i, prob := range probs {
		sorted[i] = indexProb{i, prob}
	}
	sort.Slice(sorted, func(i, j int) bool { return sorted[i].prob > sorted[j].prob })

	// Find cutoff based on surprise threshold mu.
	negInf := float32(math.Inf(-1))
	for i, sp := range sorted {
		surprise := -float32(math.Log2(float64(sp.prob)))
		if surprise > p.mu && i > 0 {
			// Mask everything from here down.
			for j := i; j < len(sorted); j++ {
				logits[sorted[j].idx] = negInf
			}
			break
		}
	}

	// Update mu based on the selected token's surprise.
	// (In actual use, this happens after sampling, but we approximate here.)
	topSurprise := -float32(math.Log2(float64(sorted[0].prob)))
	p.mu -= p.eta * (topSurprise - p.tau)
}

// ---------- Utility ----------

func softmaxFloat32(logits []float32) []float32 {
	maxVal := logits[0]
	for _, v := range logits[1:] {
		if v > maxVal {
			maxVal = v
		}
	}

	probs := make([]float32, len(logits))
	var sum float64
	for i, v := range logits {
		exp := math.Exp(float64(v - maxVal))
		probs[i] = float32(exp)
		sum += exp
	}

	invSum := float32(1.0 / sum)
	for i := range probs {
		probs[i] *= invSum
	}
	return probs
}
