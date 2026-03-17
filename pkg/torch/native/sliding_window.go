// sliding_window.go implements sliding window attention for long contexts.
//
// WHAT: Instead of attending to all previous tokens (O(N^2) memory), each
// token only attends to the most recent W tokens (O(N*W) memory). Used by
// Mistral and other long-context models.
//
// WHY: For a 128K context model with W=4096, memory drops from 128K^2 to
// 128K*4K - a 32x reduction. Information still propagates through the full
// context via the cascade of overlapping windows across layers.
package native

import (
	"math"
)

// SlidingWindowConfig controls the window attention parameters.
type SlidingWindowConfig struct {
	WindowSize int  // Number of past tokens each position can attend to
	Enabled    bool // When false, falls back to full attention
}

// SlidingWindowAttention computes attention restricted to a sliding window.
//
// Q: [seqLen, headDim]
// K: [kvLen, headDim]
// V: [kvLen, headDim]
// windowSize: maximum lookback distance
//
// Returns: [seqLen, headDim]
func SlidingWindowAttention(Q, K, V *Tensor, windowSize int) *Tensor {
	seqLen := Q.Shape[0]
	headDim := Q.Shape[1]
	kvLen := K.Shape[0]
	scale := float32(1.0 / math.Sqrt(float64(headDim)))

	out := NewTensor([]int{seqLen, headDim})

	for qi := 0; qi < seqLen; qi++ {
		// Window bounds: attend to [max(0, qi-windowSize+1), qi].
		kStart := qi - windowSize + 1
		if kStart < 0 {
			kStart = 0
		}
		kEnd := qi + 1
		if kEnd > kvLen {
			kEnd = kvLen
		}

		// Compute scores within the window.
		windowLen := kEnd - kStart
		scores := make([]float32, windowLen)
		maxScore := float32(-1e30)

		for i := 0; i < windowLen; i++ {
			ki := kStart + i
			score := float32(0)
			for d := 0; d < headDim; d++ {
				score += Q.Data[qi*headDim+d] * K.Data[ki*headDim+d]
			}
			scores[i] = score * scale
			if scores[i] > maxScore {
				maxScore = scores[i]
			}
		}

		// Softmax over window.
		var sumExp float32
		for i := range scores {
			scores[i] = float32(math.Exp(float64(scores[i] - maxScore)))
			sumExp += scores[i]
		}
		if sumExp > 0 {
			invSum := float32(1.0) / sumExp
			for i := range scores {
				scores[i] *= invSum
			}
		}

		// Weighted sum of values within window.
		for i := 0; i < windowLen; i++ {
			ki := kStart + i
			w := scores[i]
			for d := 0; d < headDim; d++ {
				out.Data[qi*headDim+d] += w * V.Data[ki*headDim+d]
			}
		}
	}

	return out
}

// SlidingWindowMask generates a causal mask with sliding window constraint.
// Returns a bool mask where mask[i*seqLen+j] = true means position i can attend to j.
func SlidingWindowMask(seqLen, windowSize int) []bool {
	mask := make([]bool, seqLen*seqLen)
	for i := 0; i < seqLen; i++ {
		for j := 0; j < seqLen; j++ {
			// Causal: j <= i, and within window: i-j < windowSize
			if j <= i && (i-j) < windowSize {
				mask[i*seqLen+j] = true
			}
		}
	}
	return mask
}
