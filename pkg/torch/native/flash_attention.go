// flash_attention.go implements memory-efficient tiled attention (Flash Attention).
//
// WHAT: Standard attention materializes the full NxN attention matrix, using
// O(N^2) memory. Flash Attention tiles the computation so only small blocks
// are held in memory at once, reducing memory to O(N) while computing the
// exact same result. This unlocks 32K+ context without OOM.
//
// HOW: The key insight is that softmax can be computed incrementally. We
// process Q against blocks of K/V, keeping a running max and sum for the
// online softmax correction. This is mathematically identical to full
// attention but never allocates the NxN matrix.
//
// REFERENCE: Dao et al., "FlashAttention: Fast and Memory-Efficient Exact
// Attention with IO-Awareness" (2022), FlashAttention-2 (2023).
package native

import (
	"math"
)

// FlashAttentionConfig controls tiling parameters.
type FlashAttentionConfig struct {
	BlockSizeQ int // Tile size for query dimension (default: 64)
	BlockSizeK int // Tile size for key dimension (default: 64)
	Causal     bool // Apply causal mask (default: true for autoregressive)
}

// DefaultFlashConfig returns standard Flash Attention parameters.
func DefaultFlashConfig() FlashAttentionConfig {
	return FlashAttentionConfig{
		BlockSizeQ: 64,
		BlockSizeK: 64,
		Causal:     true,
	}
}

// FlashAttention computes scaled dot-product attention using tiled computation.
//
// Q: [seqLen, headDim] - queries for one head
// K: [kvLen, headDim]  - keys for one head
// V: [kvLen, headDim]  - values for one head
//
// Returns: [seqLen, headDim] - attention output
//
// Memory: O(seqLen * headDim) instead of O(seqLen * kvLen)
func FlashAttention(Q, K, V *Tensor, config FlashAttentionConfig) *Tensor {
	seqLen := Q.Shape[0]
	headDim := Q.Shape[1]
	kvLen := K.Shape[0]

	bQ := config.BlockSizeQ
	bK := config.BlockSizeK
	scale := float32(1.0 / math.Sqrt(float64(headDim)))

	// Output accumulator and softmax stats.
	out := NewTensor([]int{seqLen, headDim})
	rowMax := make([]float32, seqLen)  // Running max for numerical stability
	rowSum := make([]float32, seqLen)  // Running sum of exp(score - max)

	// Initialize max to -inf.
	for i := range rowMax {
		rowMax[i] = -1e30
	}

	// Process K/V in blocks.
	for kStart := 0; kStart < kvLen; kStart += bK {
		kEnd := kStart + bK
		if kEnd > kvLen {
			kEnd = kvLen
		}
		kBlockSize := kEnd - kStart

		// Process Q in blocks.
		for qStart := 0; qStart < seqLen; qStart += bQ {
			qEnd := qStart + bQ
			if qEnd > seqLen {
				qEnd = seqLen
			}

			// For each query position in this Q block...
			for qi := qStart; qi < qEnd; qi++ {
				// Compute attention scores for this query against the K block.
				for ki := 0; ki < kBlockSize; ki++ {
					kIdx := kStart + ki

					// Causal mask: query can only attend to keys at or before its position.
					if config.Causal && kIdx > qi {
						continue
					}

					// Dot product: Q[qi] . K[kIdx] * scale
					score := float32(0)
					qOff := qi * headDim
					kOff := kIdx * headDim
					for d := 0; d < headDim; d++ {
						score += Q.Data[qOff+d] * K.Data[kOff+d]
					}
					score *= scale

					// Online softmax update.
					prevMax := rowMax[qi]
					if score > prevMax {
						// New max: rescale previous accumulator.
						correction := float32(math.Exp(float64(prevMax - score)))
						rowSum[qi] *= correction
						for d := 0; d < headDim; d++ {
							out.Data[qi*headDim+d] *= correction
						}
						rowMax[qi] = score
					}

					// Accumulate: exp(score - max) * V[kIdx]
					w := float32(math.Exp(float64(score - rowMax[qi])))
					rowSum[qi] += w
					vOff := kIdx * headDim
					for d := 0; d < headDim; d++ {
						out.Data[qi*headDim+d] += w * V.Data[vOff+d]
					}
				}
			}
		}
	}

	// Final normalization: divide by sum.
	for qi := 0; qi < seqLen; qi++ {
		if rowSum[qi] > 0 {
			invSum := float32(1.0) / rowSum[qi]
			for d := 0; d < headDim; d++ {
				out.Data[qi*headDim+d] *= invSum
			}
		}
	}

	return out
}

// FlashGQAttention applies Flash Attention with Grouped Query Attention.
// Handles the head count mismatch between Q and K/V.
//
// Q: [seqLen, numHeads * headDim]
// K: [kvLen, numKVHeads * headDim]
// V: [kvLen, numKVHeads * headDim]
func FlashGQAttention(Q, K, V *Tensor, numHeads, numKVHeads int, config FlashAttentionConfig) *Tensor {
	seqLen := Q.Shape[0]
	totalQDim := Q.Shape[1]
	headDim := totalQDim / numHeads
	kvLen := K.Shape[0]
	headsPerGroup := numHeads / numKVHeads

	out := NewTensor([]int{seqLen, totalQDim})

	// Process each head.
	for h := 0; h < numHeads; h++ {
		kvHead := h / headsPerGroup

		// Extract single-head Q, K, V slices.
		qHead := NewTensor([]int{seqLen, headDim})
		kHead := NewTensor([]int{kvLen, headDim})
		vHead := NewTensor([]int{kvLen, headDim})

		for t := 0; t < seqLen; t++ {
			copy(qHead.Data[t*headDim:(t+1)*headDim],
				Q.Data[t*totalQDim+h*headDim:t*totalQDim+(h+1)*headDim])
		}
		kvDim := numKVHeads * headDim
		for t := 0; t < kvLen; t++ {
			copy(kHead.Data[t*headDim:(t+1)*headDim],
				K.Data[t*kvDim+kvHead*headDim:t*kvDim+(kvHead+1)*headDim])
			copy(vHead.Data[t*headDim:(t+1)*headDim],
				V.Data[t*kvDim+kvHead*headDim:t*kvDim+(kvHead+1)*headDim])
		}

		// Run Flash Attention for this head.
		headOut := FlashAttention(qHead, kHead, vHead, config)

		// Copy result back into output.
		for t := 0; t < seqLen; t++ {
			copy(out.Data[t*totalQDim+h*headDim:t*totalQDim+(h+1)*headDim],
				headOut.Data[t*headDim:(t+1)*headDim])
		}
	}

	return out
}
