// token_pipeline.go implements pipelined token generation.
//
// WHY: Standard autoregressive generation is strictly sequential:
//   forward(token_n) -> sample -> forward(token_n+1) -> sample -> ...
//
// With pipelining, we overlap the sampling of token N with the forward
// pass preparation for token N+1. The stages run concurrently:
//
//   Time:  [T0]         [T1]          [T2]          [T3]
//   Stage1: forward(0) | forward(1)  | forward(2)  | forward(3)
//   Stage2:            | sample(0)   | sample(1)   | sample(2)
//   Stage3:            |             | emit(0)     | emit(1)
//
// For CPU inference, the overlap benefit comes from sampling (softmax +
// random selection) running concurrently with the start of the next
// forward pass. For GPU inference, the overlap is even larger since
// the CPU can sample while the GPU computes the next forward.
package native

import (
	"math"
	"math/rand"
)

// TokenPipeline manages pipelined token generation with overlapping stages.
type TokenPipeline struct {
	// Configuration.
	Temperature float32
	TopP        float32
	VocabSize   int

	// Pipeline channels.
	logitsCh chan []float32 // forward pass sends logits here
	tokenCh  chan int       // sampler sends tokens here

	// State.
	running bool
}

// NewTokenPipeline creates a pipelined token generator.
func NewTokenPipeline(vocabSize int, temperature, topP float32) *TokenPipeline {
	return &TokenPipeline{
		Temperature: temperature,
		TopP:        topP,
		VocabSize:   vocabSize,
		logitsCh:    make(chan []float32, 1), // buffer of 1 for overlap
		tokenCh:     make(chan int, 1),
	}
}

// StartSampler launches the sampling goroutine that consumes logits
// and produces tokens concurrently with the forward pass.
func (p *TokenPipeline) StartSampler() {
	p.running = true
	go func() {
		for logits := range p.logitsCh {
			token := p.sample(logits)
			p.tokenCh <- token
		}
		close(p.tokenCh)
	}()
}

// SubmitLogits sends logits from the forward pass to the sampler.
// This call may block briefly if the sampler hasn't consumed the previous logits.
func (p *TokenPipeline) SubmitLogits(logits []float32) {
	p.logitsCh <- logits
}

// NextToken receives the next sampled token from the pipeline.
func (p *TokenPipeline) NextToken() int {
	return <-p.tokenCh
}

// Stop shuts down the sampling pipeline gracefully.
func (p *TokenPipeline) Stop() {
	if p.running {
		close(p.logitsCh)
		p.running = false
	}
}

// sample performs temperature-scaled top-p (nucleus) sampling on logits.
func (p *TokenPipeline) sample(logits []float32) int {
	if p.Temperature <= 0 {
		// Greedy: return argmax.
		return pipelineArgmax(logits)
	}

	// Temperature scaling.
	invTemp := 1.0 / p.Temperature
	for i := range logits {
		logits[i] *= invTemp
	}

	// Softmax.
	maxLogit := logits[0]
	for _, l := range logits[1:] {
		if l > maxLogit {
			maxLogit = l
		}
	}

	var sumExp float32
	for i := range logits {
		logits[i] = float32(math.Exp(float64(logits[i] - maxLogit)))
		sumExp += logits[i]
	}
	for i := range logits {
		logits[i] /= sumExp
	}

	// Top-P filtering.
	if p.TopP > 0 && p.TopP < 1.0 {
		return topPSample(logits, p.TopP)
	}

	// Standard multinomial sampling.
	r := rand.Float32()
	var cumulative float32
	for i, prob := range logits {
		cumulative += prob
		if cumulative >= r {
			return i
		}
	}
	return len(logits) - 1
}

// pipelineArgmax returns the index of the largest element.
func pipelineArgmax(v []float32) int {
	best := 0
	bestVal := v[0]
	for i := 1; i < len(v); i++ {
		if v[i] > bestVal {
			bestVal = v[i]
			best = i
		}
	}
	return best
}

// topPSample performs nucleus sampling: only considers tokens whose
// cumulative probability exceeds topP.
func topPSample(probs []float32, topP float32) int {
	// Build index-probability pairs and sort by probability descending.
	type indexProb struct {
		index int
		prob  float32
	}

	// Partial sort: find top tokens until cumulative >= topP.
	// For small vocab this is fast enough; for large vocab we'd use
	// a heap but typical LLM vocab is 32k-128k and this is not the bottleneck.
	n := len(probs)
	indices := make([]indexProb, n)
	for i := range probs {
		indices[i] = indexProb{i, probs[i]}
	}

	// Simple selection: iterate through and pick greedily.
	// This is O(n*k) where k is typically small (top-p cuts off early).
	var cumulative float32
	r := rand.Float32() * topP // scale random to the nucleus

	for {
		// Find the current maximum.
		bestIdx := -1
		var bestProb float32
		for i, ip := range indices {
			if ip.prob > bestProb {
				bestProb = ip.prob
				bestIdx = i
			}
		}
		if bestIdx < 0 {
			break
		}

		cumulative += bestProb
		if cumulative >= r {
			return indices[bestIdx].index
		}

		// Remove this token from consideration.
		indices[bestIdx].prob = -1
	}

	return 0
}
