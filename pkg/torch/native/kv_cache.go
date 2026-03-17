// kv_cache.go implements a ring buffer KV cache for transformer inference.
//
// WHY: Standard KV cache implementations use growing slices that cause
// repeated allocations and copies. A ring buffer keeps a fixed-size
// allocation and wraps the write pointer, eliminating all allocations
// during generation. Memory usage is constant regardless of sequence length.
//
// HOW: Pre-allocates [maxSeqLen * headDim] for each head's K and V.
// New entries overwrite the oldest when the buffer is full (sliding window).
// The current write position wraps around using modular arithmetic.
package native

// KVRingCache stores key and value vectors for all layers and heads
// in a fixed-size ring buffer. No allocations after initialization.
type KVRingCache struct {
	// Dimensions.
	numHeads int
	headDim  int
	maxLen   int // maximum sequence length (ring buffer size)

	// Per-layer storage: kvKeys[layer][head * maxLen * headDim]
	kvKeys   [][]float32 // keys for each layer
	kvValues [][]float32 // values for each layer

	// Write position (wraps at maxLen). Shared across layers since
	// all layers process the same token positions.
	writePos int
	length   int // current number of valid entries (up to maxLen)
}

// NewKVRingCache creates a pre-allocated KV cache.
//
// Parameters:
//   - numHeads: number of attention heads
//   - headDim: dimension per head
//   - maxLen: maximum sequence length (ring buffer wraps after this)
func NewKVRingCache(numHeads, headDim, maxLen int) *KVRingCache {
	return &KVRingCache{
		numHeads: numHeads,
		headDim:  headDim,
		maxLen:   maxLen,
		kvKeys:   make([][]float32, 0),
		kvValues: make([][]float32, 0),
		writePos: 0,
		length:   0,
	}
}

// ensureLayer allocates storage for a layer if not yet created.
func (c *KVRingCache) ensureLayer(layer int) {
	for len(c.kvKeys) <= layer {
		size := c.numHeads * c.maxLen * c.headDim
		c.kvKeys = append(c.kvKeys, make([]float32, size))
		c.kvValues = append(c.kvValues, make([]float32, size))
	}
}

// Append adds a key-value pair for all heads at the current position.
// The kv vectors should be [numHeads * headDim] sized (full hidden dim).
func (c *KVRingCache) Append(layer int, key, value []float32) {
	c.ensureLayer(layer)

	pos := c.writePos % c.maxLen

	// Copy key and value data for each head into the ring buffer.
	for h := 0; h < c.numHeads; h++ {
		srcOff := h * c.headDim
		dstOff := (h*c.maxLen + pos) * c.headDim

		copy(c.kvKeys[layer][dstOff:dstOff+c.headDim], key[srcOff:srcOff+c.headDim])
		copy(c.kvValues[layer][dstOff:dstOff+c.headDim], value[srcOff:srcOff+c.headDim])
	}

	// Only increment writePos once per token, not per layer.
	// The caller is responsible for calling Append for each layer at the same position.
}

// Advance moves the write pointer forward by one position.
// Call this after writing all layers for a token.
func (c *KVRingCache) Advance() {
	c.writePos++
	if c.length < c.maxLen {
		c.length++
	}
}

// Len returns the number of valid entries in the cache.
func (c *KVRingCache) Len() int {
	return c.length
}

// GetKey returns the key vector for a given layer, head, and sequence position.
// Position is relative to the start of the sequence (0 = first token).
func (c *KVRingCache) GetKey(layer, head, seqPos int) []float32 {
	// Map sequence position to ring buffer position.
	ringPos := seqPos % c.maxLen
	off := (head*c.maxLen + ringPos) * c.headDim
	return c.kvKeys[layer][off : off+c.headDim]
}

// GetValue returns the value vector for a given layer, head, and sequence position.
func (c *KVRingCache) GetValue(layer, head, seqPos int) []float32 {
	ringPos := seqPos % c.maxLen
	off := (head*c.maxLen + ringPos) * c.headDim
	return c.kvValues[layer][off : off+c.headDim]
}

// Reset clears the cache without deallocating memory.
func (c *KVRingCache) Reset() {
	c.writePos = 0
	c.length = 0
	// Data is not zeroed - it will be overwritten by new entries.
}
