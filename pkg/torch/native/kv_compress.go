// kv_compress.go implements KV cache compression for reduced memory at long contexts.
//
// WHAT: During generation, each layer stores K and V vectors for every past
// token. At 128K context with 32 layers, this can exceed 16GB. KV compression
// quantizes these cached vectors from FP32 to FP8 or INT8, reducing memory
// by 4x with minimal quality loss.
//
// HOW: After computing K and V for each new token, we quantize them before
// storing in the cache. On retrieval, we dequantize back to FP32 for the
// attention computation. Block-wise quantization preserves relative magnitudes.
package native

import (
	"fmt"
	"math"
)

// KVCompressionMode specifies how KV cache values are compressed.
type KVCompressionMode int

const (
	KVCompressNone KVCompressionMode = iota
	KVCompressFP8                    // 8-bit float (E4M3)
	KVCompressINT8                   // Symmetric INT8 with per-block scale
)

func (m KVCompressionMode) String() string {
	switch m {
	case KVCompressFP8:
		return "FP8"
	case KVCompressINT8:
		return "INT8"
	default:
		return "none"
	}
}

// KVCompressConfig controls cache compression behavior.
type KVCompressConfig struct {
	Mode      KVCompressionMode
	BlockSize int  // Quantization block size (default: 32)
	Enabled   bool
}

// DefaultKVCompressConfig returns recommended compression settings.
func DefaultKVCompressConfig() KVCompressConfig {
	return KVCompressConfig{
		Mode:      KVCompressINT8,
		BlockSize: 32,
		Enabled:   true,
	}
}

// CompressedKVEntry stores a quantized KV cache entry.
type CompressedKVEntry struct {
	QuantizedData []int8     // Quantized values
	Scales        []float32  // Per-block scale factors
	BlockSize     int
	OriginalLen   int        // Length of original float32 data
}

// CompressKV quantizes float32 KV data to INT8 with per-block scales.
func CompressKV(data []float32, blockSize int) *CompressedKVEntry {
	entry := &CompressedKVEntry{
		QuantizedData: make([]int8, len(data)),
		BlockSize:     blockSize,
		OriginalLen:   len(data),
	}

	numBlocks := (len(data) + blockSize - 1) / blockSize
	entry.Scales = make([]float32, numBlocks)

	for b := 0; b < numBlocks; b++ {
		start := b * blockSize
		end := start + blockSize
		if end > len(data) {
			end = len(data)
		}
		block := data[start:end]

		// Find absolute max for the block.
		absMax := float32(0)
		for _, v := range block {
			av := float32(math.Abs(float64(v)))
			if av > absMax {
				absMax = av
			}
		}

		if absMax == 0 {
			entry.Scales[b] = 0
			continue
		}

		// Scale to [-127, 127] range.
		scale := absMax / 127.0
		entry.Scales[b] = scale
		invScale := float32(127.0) / absMax

		for i, v := range block {
			q := int8(math.Round(float64(v * invScale)))
			entry.QuantizedData[start+i] = q
		}
	}

	return entry
}

// DecompressKV restores float32 data from compressed INT8.
func (e *CompressedKVEntry) DecompressKV() []float32 {
	data := make([]float32, e.OriginalLen)
	numBlocks := len(e.Scales)

	for b := 0; b < numBlocks; b++ {
		start := b * e.BlockSize
		end := start + e.BlockSize
		if end > e.OriginalLen {
			end = e.OriginalLen
		}

		scale := e.Scales[b]
		for i := start; i < end; i++ {
			data[i] = float32(e.QuantizedData[i]) * scale
		}
	}

	return data
}

// CompressionRatio returns the memory savings ratio.
func (e *CompressedKVEntry) CompressionRatio() float64 {
	originalBytes := e.OriginalLen * 4 // FP32
	compressedBytes := len(e.QuantizedData) + len(e.Scales)*4
	return float64(originalBytes) / float64(compressedBytes)
}

// KVCompressedCache manages a full compressed KV cache across layers.
type KVCompressedCache struct {
	config   KVCompressConfig
	layers   int
	// [layer][position] -> compressed K/V
	keyCache   []map[int]*CompressedKVEntry
	valueCache []map[int]*CompressedKVEntry
}

// NewKVCompressedCache creates a compressed cache for the given number of layers.
func NewKVCompressedCache(numLayers int, config KVCompressConfig) *KVCompressedCache {
	c := &KVCompressedCache{
		config:     config,
		layers:     numLayers,
		keyCache:   make([]map[int]*CompressedKVEntry, numLayers),
		valueCache: make([]map[int]*CompressedKVEntry, numLayers),
	}
	for i := 0; i < numLayers; i++ {
		c.keyCache[i] = make(map[int]*CompressedKVEntry)
		c.valueCache[i] = make(map[int]*CompressedKVEntry)
	}
	fmt.Printf("[KVCompress] %s cache for %d layers (block=%d)\n",
		config.Mode, numLayers, config.BlockSize)
	return c
}

// StoreKV compresses and stores K/V vectors for a layer at a position.
func (c *KVCompressedCache) StoreKV(layer, pos int, keys, values []float32) {
	c.keyCache[layer][pos] = CompressKV(keys, c.config.BlockSize)
	c.valueCache[layer][pos] = CompressKV(values, c.config.BlockSize)
}

// GetKV retrieves and decompresses K/V vectors for a layer at a position.
func (c *KVCompressedCache) GetKV(layer, pos int) ([]float32, []float32, bool) {
	kEntry, kOk := c.keyCache[layer][pos]
	vEntry, vOk := c.valueCache[layer][pos]
	if !kOk || !vOk {
		return nil, nil, false
	}
	return kEntry.DecompressKV(), vEntry.DecompressKV(), true
}

// MemoryUsage returns total memory used by the compressed cache in bytes.
func (c *KVCompressedCache) MemoryUsage() int64 {
	var total int64
	for l := 0; l < c.layers; l++ {
		for _, entry := range c.keyCache[l] {
			total += int64(len(entry.QuantizedData) + len(entry.Scales)*4)
		}
		for _, entry := range c.valueCache[l] {
			total += int64(len(entry.QuantizedData) + len(entry.Scales)*4)
		}
	}
	return total
}
