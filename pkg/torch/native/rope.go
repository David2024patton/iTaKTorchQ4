// rope.go implements Rotary Position Embeddings (RoPE) for transformer models.
//
// WHAT: RoPE encodes token position by rotating pairs of dimensions in the
// query and key vectors. This gives the model awareness of relative token
// distance without explicit position embeddings. Used by Llama, Qwen, Mistral,
// Phi, and most modern architectures.
//
// HOW: For each pair of dimensions (2i, 2i+1), rotate by angle = pos * theta_i
// where theta_i = 1 / (base ^ (2i / dim)). Default base = 10000.
//
// EXTENDED CONTEXT: Supports NTK-aware scaling and YaRN for context extension
// beyond the model's training length.
package native

import (
	"math"
)

// RoPEConfig holds rotary embedding parameters.
type RoPEConfig struct {
	Dim       int     // Head dimension (must be even)
	MaxSeqLen int     // Maximum sequence length
	Base      float64 // Frequency base (default: 10000)
	// NTK scaling for extended context.
	NTKAlpha  float64 // NTK-aware scaling factor (0 = disabled)
	ScaleFactor float64 // Linear scaling factor (1.0 = no scaling)
}

// DefaultRoPEConfig returns standard RoPE parameters.
func DefaultRoPEConfig(headDim, maxSeqLen int) RoPEConfig {
	return RoPEConfig{
		Dim:         headDim,
		MaxSeqLen:   maxSeqLen,
		Base:        10000.0,
		ScaleFactor: 1.0,
	}
}

// RoPECache precomputes sin/cos tables for all positions.
// Computing these once avoids repeated trig calls during inference.
type RoPECache struct {
	config  RoPEConfig
	cosTable []float32 // [maxSeqLen * dim/2] - cos(pos * theta_i) for each pair
	sinTable []float32 // [maxSeqLen * dim/2] - sin(pos * theta_i) for each pair
}

// NewRoPECache precomputes the rotary embedding tables.
func NewRoPECache(config RoPEConfig) *RoPECache {
	halfDim := config.Dim / 2
	base := config.Base

	// NTK-aware scaling: adjust the base frequency.
	if config.NTKAlpha > 0 {
		base = base * math.Pow(config.NTKAlpha*float64(config.Dim)/(float64(config.Dim)-2), float64(config.Dim)/(float64(config.Dim)-2))
	}

	cache := &RoPECache{
		config:   config,
		cosTable: make([]float32, config.MaxSeqLen*halfDim),
		sinTable: make([]float32, config.MaxSeqLen*halfDim),
	}

	// Precompute theta_i = 1 / (base ^ (2i / dim)) for each pair index i.
	thetas := make([]float64, halfDim)
	for i := 0; i < halfDim; i++ {
		thetas[i] = 1.0 / math.Pow(base, float64(2*i)/float64(config.Dim))
	}

	// For each position, compute sin/cos of pos * theta_i.
	for pos := 0; pos < config.MaxSeqLen; pos++ {
		scaledPos := float64(pos) / config.ScaleFactor
		for i := 0; i < halfDim; i++ {
			angle := scaledPos * thetas[i]
			idx := pos*halfDim + i
			cache.cosTable[idx] = float32(math.Cos(angle))
			cache.sinTable[idx] = float32(math.Sin(angle))
		}
	}

	return cache
}

// Apply rotates the query or key vector at the given position using RoPE.
// vec: [dim] float32 - modified in-place.
// pos: token position in the sequence.
//
// For each pair (vec[2i], vec[2i+1]):
//   rotated[2i]   = vec[2i]*cos - vec[2i+1]*sin
//   rotated[2i+1] = vec[2i]*sin + vec[2i+1]*cos
func (rc *RoPECache) Apply(vec []float32, pos int) {
	if pos >= rc.config.MaxSeqLen {
		pos = rc.config.MaxSeqLen - 1
	}
	halfDim := rc.config.Dim / 2
	base := pos * halfDim

	for i := 0; i < halfDim; i++ {
		cos := rc.cosTable[base+i]
		sin := rc.sinTable[base+i]

		x0 := vec[2*i]
		x1 := vec[2*i+1]

		vec[2*i] = x0*cos - x1*sin
		vec[2*i+1] = x0*sin + x1*cos
	}
}

// ApplyBatched applies RoPE to an entire sequence of vectors.
// vecs: [seqLen * dim] flat array where each vec is dim floats.
// startPos: position offset (for KV cache continuation).
func (rc *RoPECache) ApplyBatched(vecs []float32, seqLen, dim, startPos int) {
	halfDim := dim / 2

	for t := 0; t < seqLen; t++ {
		pos := startPos + t
		if pos >= rc.config.MaxSeqLen {
			pos = rc.config.MaxSeqLen - 1
		}

		vecOffset := t * dim
		cacheOffset := pos * halfDim

		for i := 0; i < halfDim; i++ {
			cos := rc.cosTable[cacheOffset+i]
			sin := rc.sinTable[cacheOffset+i]

			x0 := vecs[vecOffset+2*i]
			x1 := vecs[vecOffset+2*i+1]

			vecs[vecOffset+2*i] = x0*cos - x1*sin
			vecs[vecOffset+2*i+1] = x0*sin + x1*cos
		}
	}
}

// ApplyGQA applies RoPE to queries and keys with different head counts.
// Q: [seqLen, numHeads, headDim], K: [seqLen, numKVHeads, headDim]
func (rc *RoPECache) ApplyGQA(q, k []float32, seqLen, numHeads, numKVHeads, headDim, startPos int) {
	// Apply to all Q heads.
	for t := 0; t < seqLen; t++ {
		for h := 0; h < numHeads; h++ {
			offset := t*numHeads*headDim + h*headDim
			rc.Apply(q[offset:offset+headDim], startPos+t)
		}
	}

	// Apply to all KV heads.
	for t := 0; t < seqLen; t++ {
		for h := 0; h < numKVHeads; h++ {
			offset := t*numKVHeads*headDim + h*headDim
			rc.Apply(k[offset:offset+headDim], startPos+t)
		}
	}
}
