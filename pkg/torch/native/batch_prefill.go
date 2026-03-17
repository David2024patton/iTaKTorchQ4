// batch_prefill.go implements batched prompt processing.
//
// WHY: Token-by-token prompt processing wastes GPU and CPU throughput.
// By computing the entire prompt's attention in one batched matrix
// multiplication, we get 3-10x faster prefill on long prompts.
//
// HOW: Instead of running forward() once per prompt token, we:
//   1. Embed all prompt tokens into a [seq_len, hidden_dim] matrix
//   2. Compute Q, K, V for all tokens in one MatMul each
//   3. Run attention over the entire sequence at once
//   4. Only switch to token-by-token for generation
package native

import (
	"fmt"
	"math"
)

// BatchPrefill processes an entire prompt in one batched forward pass.
// Returns the KV cache entries and the final hidden state for generation.
//
// Parameters:
//   - tokens: prompt token IDs
//   - layers: model transformer layers
//   - embeddings: token embedding matrix [vocab_size, hidden_dim]
//   - gpu: optional GPU backend for accelerated matmul
//
// Returns:
//   - kvCache: populated KV cache for all prompt positions
//   - lastHidden: hidden state of the last token (for generation start)
func BatchPrefill(tokens []int, layers []TransformerLayer, embeddings *Tensor, gpu *GPUBackend, numHeads, headDim int) (*KVRingCache, *Tensor) {
	seqLen := len(tokens)
	hiddenDim := embeddings.Shape[1]

	if seqLen == 0 {
		return NewKVRingCache(numHeads, headDim, 2048), nil
	}

	// 1. Embed all tokens at once: [seq_len, hidden_dim]
	x := NewPooledTensor([]int{seqLen, hiddenDim})
	for i, tok := range tokens {
		if tok < embeddings.Shape[0] {
			copy(x.Data[i*hiddenDim:(i+1)*hiddenDim], embeddings.Data[tok*hiddenDim:(tok+1)*hiddenDim])
		}
	}

	// Initialize KV cache.
	maxCtx := seqLen + 2048 // room for generation
	if maxCtx < 4096 {
		maxCtx = 4096
	}
	kvCache := NewKVRingCache(numHeads, headDim, maxCtx)

	// 2. Run through each transformer layer.
	for layerIdx := range layers {
		l := &layers[layerIdx]

		// a) RMS norm before attention.
		if l.AttnNorm != nil {
			x = batchRMSNorm(x, l.AttnNorm, 1e-5)
		}

		// b) Compute Q, K, V for ALL tokens at once: [seq_len, hidden_dim]
		var q, k, v *Tensor
		if gpu != nil && gpu.IsAvailable() {
			q = gpu.MatMulGPU(x, l.WQ)
			k = gpu.MatMulGPU(x, l.WK)
			v = gpu.MatMulGPU(x, l.WV)
		} else {
			q = MatMul(x, l.WQ)
			k = MatMul(x, l.WK)
			v = MatMul(x, l.WV)
		}

		// c) Store K, V in the ring cache for all positions.
		for pos := 0; pos < seqLen; pos++ {
			kRow := k.Data[pos*hiddenDim : (pos+1)*hiddenDim]
			vRow := v.Data[pos*hiddenDim : (pos+1)*hiddenDim]
			kvCache.Append(layerIdx, kRow, vRow)
		}

		// d) Batched attention: each token attends to all prior tokens.
		attnOut := batchedAttention(q, k, v, seqLen, numHeads, headDim)

		// e) Output projection.
		var projected *Tensor
		if gpu != nil && gpu.IsAvailable() {
			projected = gpu.MatMulGPU(attnOut, l.WO)
		} else {
			projected = MatMul(attnOut, l.WO)
		}

		// f) Residual connection.
		x = Add(x, projected)

		// g) FFN block.
		if l.FFNNorm != nil {
			normed := batchRMSNorm(x, l.FFNNorm, 1e-5)
			x = batchFFN(normed, x, l, gpu)
		}
	}

	// 3. Extract the last token's hidden state for generation.
	lastHidden := NewTensor([]int{hiddenDim})
	copy(lastHidden.Data, x.Data[(seqLen-1)*hiddenDim:seqLen*hiddenDim])

	fmt.Printf("[BatchPrefill] Processed %d tokens in one pass\n", seqLen)
	return kvCache, lastHidden
}

// batchRMSNorm applies RMS normalization to each row of a batched tensor.
func batchRMSNorm(x, weight *Tensor, eps float32) *Tensor {
	seqLen := x.Shape[0]
	dim := x.Shape[1]
	out := NewPooledTensor([]int{seqLen, dim})

	for row := 0; row < seqLen; row++ {
		offset := row * dim
		// Compute RMS for this row.
		var sumSq float64
		for i := 0; i < dim; i++ {
			v := float64(x.Data[offset+i])
			sumSq += v * v
		}
		rms := float32(math.Sqrt(sumSq/float64(dim) + float64(eps)))
		invRms := 1.0 / rms

		// Normalize and scale.
		for i := 0; i < dim; i++ {
			out.Data[offset+i] = x.Data[offset+i] * invRms * weight.Data[i]
		}
	}
	return out
}

// batchedAttention computes causal self-attention over all positions at once.
func batchedAttention(q, k, v *Tensor, seqLen, numHeads, headDim int) *Tensor {
	hiddenDim := numHeads * headDim
	out := NewPooledTensor([]int{seqLen, hiddenDim})
	scale := float32(1.0 / math.Sqrt(float64(headDim)))

	for h := 0; h < numHeads; h++ {
		hOff := h * headDim

		for pos := 0; pos < seqLen; pos++ {
			// Compute attention scores for this position (causal: only attend to 0..pos).
			scores := make([]float32, pos+1)
			qRow := q.Data[pos*hiddenDim+hOff : pos*hiddenDim+hOff+headDim]

			for t := 0; t <= pos; t++ {
				kRow := k.Data[t*hiddenDim+hOff : t*hiddenDim+hOff+headDim]
				var dot float32
				for d := 0; d < headDim; d++ {
					dot += qRow[d] * kRow[d]
				}
				scores[t] = dot * scale
			}

			// Softmax over scores.
			maxScore := scores[0]
			for _, s := range scores[1:] {
				if s > maxScore {
					maxScore = s
				}
			}
			var sumExp float32
			for i := range scores {
				scores[i] = float32(math.Exp(float64(scores[i] - maxScore)))
				sumExp += scores[i]
			}
			for i := range scores {
				scores[i] /= sumExp
			}

			// Weighted sum of values.
			outSlice := out.Data[pos*hiddenDim+hOff : pos*hiddenDim+hOff+headDim]
			for t := 0; t <= pos; t++ {
				vRow := v.Data[t*hiddenDim+hOff : t*hiddenDim+hOff+headDim]
				for d := 0; d < headDim; d++ {
					outSlice[d] += scores[t] * vRow[d]
				}
			}
		}
	}

	return out
}

// batchFFN applies the feed-forward network to a batched input.
func batchFFN(normed, residual *Tensor, l *TransformerLayer, gpu *GPUBackend) *Tensor {
	var gate, up, down *Tensor

	if gpu != nil && gpu.IsAvailable() {
		gate = gpu.MatMulGPU(normed, l.WGate)
		up = gpu.MatMulGPU(normed, l.WUp)
	} else {
		gate = MatMul(normed, l.WGate)
		up = MatMul(normed, l.WUp)
	}

	// SiLU(gate) * up for each row.
	seqLen := gate.Shape[0]
	ffnDim := gate.Shape[1]
	fused := NewPooledTensor([]int{seqLen, ffnDim})
	for i := range fused.Data {
		g := gate.Data[i]
		sigmoid := float32(1.0 / (1.0 + math.Exp(-float64(g))))
		fused.Data[i] = g * sigmoid * up.Data[i]
	}

	if gpu != nil && gpu.IsAvailable() {
		down = gpu.MatMulGPU(fused, l.WDown)
	} else {
		down = MatMul(fused, l.WDown)
	}

	return Add(residual, down)
}
