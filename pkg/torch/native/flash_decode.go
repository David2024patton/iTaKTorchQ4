// flash_decode.go implements optimized attention for the decode phase.
//
// WHAT: Flash Attention is optimized for prefill (processing many tokens at
// once). But during decode, we generate one token at a time, attending to the
// entire KV cache. Flash Decoding parallelizes this differently:
//
//   Standard decode: for each new query token, sequentially scan the KV cache.
//   Flash Decode:    split the KV cache into chunks, compute partial attention
//                    in parallel, then reduce the partial results.
//
// WHY: During decode, the KV cache can be very long (thousands of tokens).
// Sequential attention over this is bottlenecked by memory bandwidth.
// Flash Decoding splits the work across parallel workers and uses the
// online softmax trick to merge partial results without materializing
// the full attention matrix.
//
// GAIN: 2-8x faster decode on long sequences (especially > 2K context).
//
// REFERENCE: Flash-Decoding (Dao et al., 2023)
package native

import (
	"math"
	"sync"
)

// FlashDecoder optimizes single-token attention during the decode phase.
type FlashDecoder struct {
	numWorkers int     // Number of parallel reduction workers
	chunkSize  int     // Number of KV positions per chunk
	scale      float32 // 1/sqrt(headDim)
}

// FlashDecodeConfig configures flash decoding.
type FlashDecodeConfig struct {
	NumWorkers int // Parallel workers for KV split (default: 4)
	ChunkSize  int // KV positions per chunk (default: 256)
}

// DefaultFlashDecodeConfig returns recommended settings.
func DefaultFlashDecodeConfig() FlashDecodeConfig {
	return FlashDecodeConfig{
		NumWorkers: 4,
		ChunkSize:  256,
	}
}

// NewFlashDecoder creates a flash decoder.
func NewFlashDecoder(headDim int, config FlashDecodeConfig) *FlashDecoder {
	if config.NumWorkers < 1 {
		config.NumWorkers = 4
	}
	if config.ChunkSize < 1 {
		config.ChunkSize = 256
	}
	return &FlashDecoder{
		numWorkers: config.NumWorkers,
		chunkSize:  config.ChunkSize,
		scale:      float32(1.0 / math.Sqrt(float64(headDim))),
	}
}

// partialResult holds the partial attention output from one chunk.
type partialResult struct {
	output []float32 // Partial weighted sum of V
	maxQK  float32   // Maximum Q*K score in this chunk (for log-sum-exp)
	expSum float64   // Sum of exp(Q*K - maxQK) in this chunk
}

// Decode computes attention for a single query token against the full KV cache.
// This is the hot path during token generation.
//
// query:  [headDim] - the query vector for the new token
// keys:   [seqLen * headDim] - all cached key vectors
// values: [seqLen * headDim] - all cached value vectors
// seqLen: number of tokens in the KV cache
// headDim: dimension per head
//
// Returns: [headDim] - the attention output
func (fd *FlashDecoder) Decode(query, keys, values []float32, seqLen, headDim int) []float32 {
	if seqLen <= fd.chunkSize {
		// Short sequence: single-pass is faster than parallel split.
		return fd.singlePassDecode(query, keys, values, seqLen, headDim)
	}

	// Split KV cache into chunks for parallel processing.
	numChunks := (seqLen + fd.chunkSize - 1) / fd.chunkSize
	results := make([]partialResult, numChunks)

	var wg sync.WaitGroup
	for c := 0; c < numChunks; c++ {
		wg.Add(1)
		go func(chunkIdx int) {
			defer wg.Done()

			startPos := chunkIdx * fd.chunkSize
			endPos := startPos + fd.chunkSize
			if endPos > seqLen {
				endPos = seqLen
			}

			results[chunkIdx] = fd.computeChunk(query, keys, values, startPos, endPos, headDim)
		}(c)
	}
	wg.Wait()

	// Reduce: merge partial results using online softmax.
	return fd.reducePartials(results, headDim)
}

// computeChunk computes partial attention for one KV chunk.
func (fd *FlashDecoder) computeChunk(query, keys, values []float32, startPos, endPos, headDim int) partialResult {
	chunkLen := endPos - startPos
	result := partialResult{
		output: make([]float32, headDim),
		maxQK:  float32(-math.MaxFloat32),
	}

	// Compute Q*K scores for this chunk.
	scores := make([]float32, chunkLen)
	for i := 0; i < chunkLen; i++ {
		kOff := (startPos + i) * headDim
		var dot float64
		for d := 0; d < headDim; d++ {
			dot += float64(query[d]) * float64(keys[kOff+d])
		}
		scores[i] = float32(dot) * fd.scale
		if scores[i] > result.maxQK {
			result.maxQK = scores[i]
		}
	}

	// Compute exp(score - max) and weighted sum of V.
	for i := 0; i < chunkLen; i++ {
		exp := math.Exp(float64(scores[i] - result.maxQK))
		result.expSum += exp

		vOff := (startPos + i) * headDim
		weight := float32(exp)
		for d := 0; d < headDim; d++ {
			result.output[d] += weight * values[vOff+d]
		}
	}

	return result
}

// reducePartials merges partial attention results using the log-sum-exp trick.
// This is numerically stable and produces the exact same result as computing
// attention over the full sequence in one pass.
func (fd *FlashDecoder) reducePartials(partials []partialResult, headDim int) []float32 {
	// Find global maximum across all chunks.
	globalMax := float32(-math.MaxFloat32)
	for _, p := range partials {
		if p.maxQK > globalMax {
			globalMax = p.maxQK
		}
	}

	// Merge partial sums with corrected exponentials.
	output := make([]float32, headDim)
	var totalExpSum float64

	for _, p := range partials {
		// Correction factor: exp(chunk_max - global_max).
		correction := math.Exp(float64(p.maxQK - globalMax))
		correctedExpSum := p.expSum * correction
		totalExpSum += correctedExpSum

		weight := float32(correction)
		for d := 0; d < headDim; d++ {
			output[d] += weight * p.output[d]
		}
	}

	// Normalize by total exp sum.
	if totalExpSum > 0 {
		invSum := float32(1.0 / totalExpSum)
		for d := range output {
			output[d] *= invSum
		}
	}

	return output
}

// singlePassDecode handles short sequences without parallelism.
func (fd *FlashDecoder) singlePassDecode(query, keys, values []float32, seqLen, headDim int) []float32 {
	// Find max score.
	maxScore := float32(-math.MaxFloat32)
	scores := make([]float32, seqLen)
	for i := 0; i < seqLen; i++ {
		kOff := i * headDim
		var dot float64
		for d := 0; d < headDim; d++ {
			dot += float64(query[d]) * float64(keys[kOff+d])
		}
		scores[i] = float32(dot) * fd.scale
		if scores[i] > maxScore {
			maxScore = scores[i]
		}
	}

	// Softmax + weighted sum.
	output := make([]float32, headDim)
	var expSum float64
	for i := 0; i < seqLen; i++ {
		exp := math.Exp(float64(scores[i] - maxScore))
		expSum += exp

		vOff := i * headDim
		w := float32(exp)
		for d := 0; d < headDim; d++ {
			output[d] += w * values[vOff+d]
		}
	}

	invSum := float32(1.0 / expSum)
	for d := range output {
		output[d] *= invSum
	}
	return output
}
