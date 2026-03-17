// beam_search.go implements beam search decoding for higher quality output.
//
// WHAT: Instead of greedily picking the best token at each step, beam search
// maintains K "beams" (partial sequences) and expands each one. At each
// step, it keeps the top-K scoring sequences across all expansions.
//
// WHEN TO USE: Beam search produces higher-quality output than greedy
// decoding at the cost of K forward passes per step. Use it for:
//   - Translation tasks
//   - Summarization
//   - Any task where output quality > latency
//
// BEAM WIDTH: K=4 is standard. K=1 is equivalent to greedy decoding.
package native

import (
	"math"
	"sort"
)

// BeamSearchConfig controls beam search parameters.
type BeamSearchConfig struct {
	BeamWidth    int     // Number of beams (default: 4)
	MaxTokens    int     // Maximum tokens to generate
	LengthPenalty float64 // Length normalization (>1 favors longer, <1 favors shorter)
	EarlyStop    bool    // Stop when all beams have finished
}

// DefaultBeamConfig returns recommended settings.
func DefaultBeamConfig() BeamSearchConfig {
	return BeamSearchConfig{
		BeamWidth:     4,
		MaxTokens:     128,
		LengthPenalty: 1.0,
		EarlyStop:     true,
	}
}

// Beam represents one active sequence in beam search.
type Beam struct {
	Tokens   []int   // Generated token sequence
	Score    float64 // Log probability sum
	Finished bool    // Whether this beam has produced EOS
}

// NormalizedScore returns the length-normalized score.
func (b *Beam) NormalizedScore(penalty float64) float64 {
	length := float64(len(b.Tokens))
	if length == 0 {
		return b.Score
	}
	// Length normalization: score / (length^penalty)
	return b.Score / math.Pow(length, penalty)
}

// BeamSearch runs beam search decoding on the engine.
func BeamSearch(engine *NativeEngine, promptTokens []int, config BeamSearchConfig, eosToken int) []int {
	// Initialize beams.
	beams := make([]*Beam, config.BeamWidth)
	for i := range beams {
		beams[i] = &Beam{
			Tokens: make([]int, 0, config.MaxTokens),
			Score:  0.0,
		}
	}

	for step := 0; step < config.MaxTokens; step++ {
		var candidates []*Beam

		for _, beam := range beams {
			if beam.Finished {
				candidates = append(candidates, beam)
				continue
			}

			// Build context: prompt + beam tokens.
			ctx := make([]int, 0, len(promptTokens)+len(beam.Tokens))
			ctx = append(ctx, promptTokens...)
			ctx = append(ctx, beam.Tokens...)

			// Get logits from engine.
			logits := engine.forward(ctx)

			// Convert to log probabilities.
			logProbs := logSoftmax(logits)

			// Expand beam: consider top-K next tokens.
			topTokens := topKIndices(logProbs, config.BeamWidth*2)

			for _, tok := range topTokens {
				newBeam := &Beam{
					Tokens:   make([]int, len(beam.Tokens), len(beam.Tokens)+1),
					Score:    beam.Score + float64(logProbs[tok]),
					Finished: tok == eosToken,
				}
				copy(newBeam.Tokens, beam.Tokens)
				newBeam.Tokens = append(newBeam.Tokens, tok)
				candidates = append(candidates, newBeam)
			}
		}

		// Select top-K beams by normalized score.
		sort.Slice(candidates, func(i, j int) bool {
			return candidates[i].NormalizedScore(config.LengthPenalty) >
				candidates[j].NormalizedScore(config.LengthPenalty)
		})

		if len(candidates) > config.BeamWidth {
			candidates = candidates[:config.BeamWidth]
		}
		beams = candidates

		// Early stopping: all beams finished.
		if config.EarlyStop {
			allDone := true
			for _, b := range beams {
				if !b.Finished {
					allDone = false
					break
				}
			}
			if allDone {
				break
			}
		}
	}

	// Return the best beam.
	sort.Slice(beams, func(i, j int) bool {
		return beams[i].NormalizedScore(config.LengthPenalty) >
			beams[j].NormalizedScore(config.LengthPenalty)
	})

	if len(beams) > 0 {
		return beams[0].Tokens
	}
	return nil
}

// logSoftmax converts logits to log probabilities.
func logSoftmax(logits []float32) []float32 {
	maxVal := logits[0]
	for _, v := range logits[1:] {
		if v > maxVal {
			maxVal = v
		}
	}

	var sumExp float64
	for _, v := range logits {
		sumExp += math.Exp(float64(v - maxVal))
	}
	logSumExp := float32(math.Log(sumExp)) + maxVal

	result := make([]float32, len(logits))
	for i, v := range logits {
		result[i] = v - logSumExp
	}
	return result
}

// topKIndices returns the indices of the top-K values in a slice.
func topKIndices(values []float32, k int) []int {
	type indexedVal struct {
		idx int
		val float32
	}

	sorted := make([]indexedVal, len(values))
	for i, v := range values {
		sorted[i] = indexedVal{i, v}
	}

	sort.Slice(sorted, func(i, j int) bool {
		return sorted[i].val > sorted[j].val
	})

	if k > len(sorted) {
		k = len(sorted)
	}

	result := make([]int, k)
	for i := 0; i < k; i++ {
		result[i] = sorted[i].idx
	}
	return result
}
