// attention.go implements scaled dot-product attention for the GOTensor engine.
//
// Self-attention is the core mechanism of transformer models:
//   Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V
//
// This implementation is a straightforward, unoptimized version
// suitable for educational purposes and tiny models only.
package native

import "math"

// Attention computes scaled dot-product self-attention.
//
// Parameters:
//   - q: query tensor  [seq_len, head_dim]
//   - k: key tensor    [seq_len, head_dim]
//   - v: value tensor  [seq_len, head_dim]
//   - mask: causal mask [seq_len, seq_len] (true = masked, i.e. -inf)
//
// Returns: attention output [seq_len, head_dim]
//
// Algorithm:
//  1. scores = Q * K^T                    (dot product of queries and keys)
//  2. scores = scores / sqrt(head_dim)    (scale to prevent softmax saturation)
//  3. scores = scores + mask              (apply causal mask: future tokens = -inf)
//  4. weights = softmax(scores)           (normalize to probabilities)
//  5. output = weights * V                (weighted sum of values)
func Attention(q, k, v *Tensor, mask []bool) *Tensor {
	seqLen := q.Shape[0]
	headDim := q.Shape[1]
	scale := float32(1.0 / math.Sqrt(float64(headDim)))

	// Step 1-2: Compute scaled Q * K^T -> [seq_len, seq_len].
	scores := NewTensor([]int{seqLen, seqLen})
	for i := 0; i < seqLen; i++ {
		for j := 0; j < seqLen; j++ {
			var dot float32
			for d := 0; d < headDim; d++ {
				dot += q.Data[i*headDim+d] * k.Data[j*headDim+d]
			}
			scores.Data[i*seqLen+j] = dot * scale
		}
	}

	// Step 3: Apply causal mask.
	// In a causal (autoregressive) model, token i can only attend to tokens <= i.
	// We set scores[i][j] = -inf for j > i, so softmax assigns them zero weight.
	if mask != nil {
		for i := 0; i < seqLen; i++ {
			for j := 0; j < seqLen; j++ {
				if mask[i*seqLen+j] {
					scores.Data[i*seqLen+j] = -1e9 // effectively -infinity
				}
			}
		}
	}

	// Step 4: Softmax over the last dimension (key positions).
	weights := Softmax(scores)

	// Step 5: Weighted sum of values -> [seq_len, head_dim].
	output := NewTensor([]int{seqLen, headDim})
	for i := 0; i < seqLen; i++ {
		for d := 0; d < headDim; d++ {
			var sum float32
			for j := 0; j < seqLen; j++ {
				sum += weights.Data[i*seqLen+j] * v.Data[j*headDim+d]
			}
			output.Data[i*headDim+d] = sum
		}
	}

	return output
}

// CausalMask generates a lower-triangular causal mask.
// mask[i][j] = true (masked) if j > i, false otherwise.
// This prevents tokens from attending to future positions.
//
// Example for seq_len=4:
//
//	F F F F    (token 0 can't see anything - wait, it can see itself)
//	Actually:
//	F T T T    (token 0 sees only itself)
//	F F T T    (token 1 sees 0,1)
//	F F F T    (token 2 sees 0,1,2)
//	F F F F    (token 3 sees 0,1,2,3)
func CausalMask(seqLen int) []bool {
	mask := make([]bool, seqLen*seqLen)
	for i := 0; i < seqLen; i++ {
		for j := i + 1; j < seqLen; j++ {
			mask[i*seqLen+j] = true // mask future positions
		}
	}
	return mask
}

// MultiHeadAttention is the backward-compatible wrapper. Calls GQAAttention
// with numKVHeads == numQHeads (standard multi-head attention).
func MultiHeadAttention(q, k, v *Tensor, numHeads int, mask []bool) *Tensor {
	return GQAAttention(q, k, v, numHeads, numHeads, mask)
}

// GQAAttention implements Grouped-Query Attention (GQA).
//
// In standard MHA, every query head has its own K and V head.
// In GQA, multiple query heads share a single K/V head:
//
//	numQHeads=16, numKVHeads=2 -> 8 query heads per KV group (Qwen3)
//	numQHeads=32, numKVHeads=8 -> 4 query heads per KV group (Llama3)
//	numQHeads=N,  numKVHeads=N -> standard MHA (no grouping)
//	numQHeads=N,  numKVHeads=1 -> Multi-Query Attention (MQA)
//
// Parameters:
//   - q: [seq_len, numQHeads * headDim]
//   - k: [seq_len, numKVHeads * headDim]
//   - v: [seq_len, numKVHeads * headDim]
func GQAAttention(q, k, v *Tensor, numQHeads, numKVHeads int, mask []bool) *Tensor {
	seqLen := q.Shape[0]
	qDim := q.Shape[1]
	headDim := qDim / numQHeads

	// How many Q heads share each KV head.
	groupSize := 1
	if numKVHeads > 0 {
		groupSize = numQHeads / numKVHeads
	}

	// Output has Q's full width.
	result := NewTensor([]int{seqLen, qDim})

	kvDim := k.Shape[1]
	kvHeadDim := kvDim / numKVHeads

	// Use the smaller of the two head dims for attention dot products.
	attnDim := headDim
	if kvHeadDim < attnDim {
		attnDim = kvHeadDim
	}

	for h := 0; h < numQHeads; h++ {
		// Which KV head does this Q head belong to?
		kvIdx := h / groupSize

		// Extract Q head from the Q tensor.
		qHead := extractHeadSafe(q, h, headDim, qDim)
		// Extract the shared KV head.
		kHead := extractHeadSafe(k, kvIdx, kvHeadDim, kvDim)
		vHead := extractHeadSafe(v, kvIdx, kvHeadDim, kvDim)

		// Run attention (uses attnDim for dot products).
		headOut := AttentionGQA(qHead, kHead, vHead, attnDim, mask)

		// Copy head output back to result.
		outDim := headDim
		if len(headOut.Data)/seqLen < outDim {
			outDim = len(headOut.Data) / seqLen
		}
		for i := 0; i < seqLen; i++ {
			for d := 0; d < outDim; d++ {
				result.Data[i*qDim+h*headDim+d] = headOut.Data[i*kvHeadDim+d]
			}
		}
	}

	return result
}

// AttentionGQA computes attention where Q and K/V may have different head dims.
// Uses the minimum of Q's dim and K's dim for the dot product.
func AttentionGQA(q, k, v *Tensor, dotDim int, mask []bool) *Tensor {
	seqLen := q.Shape[0]
	kvHeadDim := k.Shape[1]
	scale := float32(1.0 / math.Sqrt(float64(dotDim)))

	// Q * K^T -> [seq_len, seq_len].
	scores := NewTensor([]int{seqLen, seqLen})
	for i := 0; i < seqLen; i++ {
		for j := 0; j < seqLen; j++ {
			var dot float32
			for d := 0; d < dotDim; d++ {
				qVal := float32(0)
				kVal := float32(0)
				if d < q.Shape[1] {
					qVal = q.Data[i*q.Shape[1]+d]
				}
				if d < kvHeadDim {
					kVal = k.Data[j*kvHeadDim+d]
				}
				dot += qVal * kVal
			}
			scores.Data[i*seqLen+j] = dot * scale
		}
	}

	// Causal mask.
	if mask != nil {
		for i := 0; i < seqLen; i++ {
			for j := 0; j < seqLen; j++ {
				if mask[i*seqLen+j] {
					scores.Data[i*seqLen+j] = -1e9
				}
			}
		}
	}

	// Softmax over key dimension.
	weights := Softmax(scores)

	// Weighted sum of V -> [seq_len, kvHeadDim].
	output := NewTensor([]int{seqLen, kvHeadDim})
	for i := 0; i < seqLen; i++ {
		for d := 0; d < kvHeadDim; d++ {
			var sum float32
			for j := 0; j < seqLen; j++ {
				sum += weights.Data[i*seqLen+j] * v.Data[j*kvHeadDim+d]
			}
			output.Data[i*kvHeadDim+d] = sum
		}
	}

	return output
}

// extractHeadSafe slices a [seq_len, totalDim] tensor to get one head.
// Bounds-safe: returns zeros for out-of-range indices.
func extractHeadSafe(t *Tensor, headIdx, headDim, totalDim int) *Tensor {
	seqLen := t.Shape[0]
	head := NewTensor([]int{seqLen, headDim})

	offset := headIdx * headDim
	for i := 0; i < seqLen; i++ {
		for d := 0; d < headDim; d++ {
			srcIdx := i*totalDim + offset + d
			if srcIdx < len(t.Data) {
				head.Data[i*headDim+d] = t.Data[srcIdx]
			}
		}
	}
	return head
}

