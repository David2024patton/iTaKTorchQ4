// embedding_pool.go implements embedding pooling strategies for text embeddings.
//
// WHAT: LLMs produce per-token hidden states, but many applications need a
// single vector representing the entire input (for similarity search,
// classification, clustering). Pooling compresses the sequence of
// hidden states into one fixed-length vector.
//
// STRATEGIES:
//   - Mean pooling: Average all token vectors (best for most tasks)
//   - CLS pooling: Use only the first token's vector (BERT-style)
//   - Last token pooling: Use the last token (decoder-model friendly)
//   - Max pooling: Element-wise max across all tokens
package native

import (
	"math"
)

// PoolingStrategy specifies how to reduce a sequence to a single vector.
type PoolingStrategy int

const (
	PoolMean     PoolingStrategy = iota // Average all token vectors
	PoolCLS                             // First token only
	PoolLastToken                       // Last token only
	PoolMax                             // Element-wise maximum
)

func (p PoolingStrategy) String() string {
	switch p {
	case PoolCLS:
		return "cls"
	case PoolLastToken:
		return "last_token"
	case PoolMax:
		return "max"
	default:
		return "mean"
	}
}

// PoolEmbeddings reduces a sequence of hidden states to a single embedding.
//
// hiddenStates: [seqLen][hiddenDim] - per-token hidden states
// strategy: pooling method to use
//
// Returns: [hiddenDim] - single embedding vector
func PoolEmbeddings(hiddenStates [][]float32, strategy PoolingStrategy) []float32 {
	if len(hiddenStates) == 0 {
		return nil
	}

	dim := len(hiddenStates[0])

	switch strategy {
	case PoolCLS:
		result := make([]float32, dim)
		copy(result, hiddenStates[0])
		return result

	case PoolLastToken:
		result := make([]float32, dim)
		copy(result, hiddenStates[len(hiddenStates)-1])
		return result

	case PoolMax:
		result := make([]float32, dim)
		// Initialize with first token.
		copy(result, hiddenStates[0])
		// Element-wise max.
		for i := 1; i < len(hiddenStates); i++ {
			for j := 0; j < dim; j++ {
				if hiddenStates[i][j] > result[j] {
					result[j] = hiddenStates[i][j]
				}
			}
		}
		return result

	default: // PoolMean
		result := make([]float32, dim)
		invLen := float32(1.0) / float32(len(hiddenStates))
		for _, state := range hiddenStates {
			for j := 0; j < dim; j++ {
				result[j] += state[j]
			}
		}
		for j := range result {
			result[j] *= invLen
		}
		return result
	}
}

// NormalizeEmbedding L2-normalizes an embedding vector (unit length).
// Essential for cosine similarity comparisons.
func NormalizeEmbedding(embedding []float32) []float32 {
	var sumSq float64
	for _, v := range embedding {
		sumSq += float64(v) * float64(v)
	}
	norm := float32(math.Sqrt(sumSq))
	if norm == 0 {
		return embedding
	}

	result := make([]float32, len(embedding))
	invNorm := float32(1.0) / norm
	for i, v := range embedding {
		result[i] = v * invNorm
	}
	return result
}

// CosineSimilarity computes cosine similarity between two embedding vectors.
func CosineSimilarity(a, b []float32) float32 {
	if len(a) != len(b) {
		return 0
	}

	var dot, normA, normB float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		normA += float64(a[i]) * float64(a[i])
		normB += float64(b[i]) * float64(b[i])
	}

	denom := math.Sqrt(normA) * math.Sqrt(normB)
	if denom == 0 {
		return 0
	}
	return float32(dot / denom)
}

// BatchPoolEmbeddings pools multiple sequences at once.
func BatchPoolEmbeddings(batch [][][]float32, strategy PoolingStrategy) [][]float32 {
	results := make([][]float32, len(batch))
	for i, seq := range batch {
		results[i] = PoolEmbeddings(seq, strategy)
	}
	return results
}
