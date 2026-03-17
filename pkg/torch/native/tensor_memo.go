// tensor_memo.go implements memoization for intermediate tensor computations.
//
// WHAT: During inference, many computations are repeated across requests:
//   - RoPE sin/cos tables are the same for the same positions
//   - LayerNorm statistics for identical inputs
//   - Attention masks for the same sequence length
//   - Embedding lookups for common tokens
//
// This module caches these intermediate results using a content-addressable
// store keyed by a fast hash of the input data.
package native

import (
	"encoding/binary"
	"fmt"
	"hash"
	"hash/fnv"
	"sync"
	"time"
)

// MemoKey is a hash of the input to a computation.
type MemoKey uint64

// MemoEntry stores a cached tensor computation result.
type MemoEntry struct {
	Result   *Tensor
	HitCount int
	Created  time.Time
	LastUsed time.Time
	SizeBytes int
}

// TensorMemoConfig controls memoization behavior.
type TensorMemoConfig struct {
	Enabled       bool
	MaxEntries    int   // Max cached computations
	MaxMemoryMB   int   // Max total memory for cache
}

// DefaultTensorMemoConfig returns recommended settings.
func DefaultTensorMemoConfig() TensorMemoConfig {
	return TensorMemoConfig{
		Enabled:     true,
		MaxEntries:  4096,
		MaxMemoryMB: 512,
	}
}

// TensorMemo is a content-addressable cache for tensor results.
type TensorMemo struct {
	mu        sync.RWMutex
	config    TensorMemoConfig
	cache     map[MemoKey]*MemoEntry
	totalBytes int64
	hits      int64
	misses    int64
}

// NewTensorMemo creates a tensor memoization cache.
func NewTensorMemo(config TensorMemoConfig) *TensorMemo {
	fmt.Printf("[TensorMemo] Enabled: max %d entries, %d MB\n",
		config.MaxEntries, config.MaxMemoryMB)
	return &TensorMemo{
		config:  config,
		cache:   make(map[MemoKey]*MemoEntry, config.MaxEntries),
	}
}

// HashTensor computes a fast content hash of a tensor.
func HashTensor(t *Tensor) MemoKey {
	h := fnv.New64a()
	// Hash shape.
	for _, dim := range t.Shape {
		var buf [8]byte
		binary.LittleEndian.PutUint64(buf[:], uint64(dim))
		h.Write(buf[:])
	}
	// Hash first, middle, and last data points (fast approximation).
	if len(t.Data) > 0 {
		hashFloat(h, t.Data[0])
		if len(t.Data) > 1 {
			hashFloat(h, t.Data[len(t.Data)/2])
			hashFloat(h, t.Data[len(t.Data)-1])
		}
		// Hash length for uniqueness.
		var buf [8]byte
		binary.LittleEndian.PutUint64(buf[:], uint64(len(t.Data)))
		h.Write(buf[:])
	}
	return MemoKey(h.Sum64())
}

// HashFloats computes a hash for a float32 slice (for non-tensor inputs).
func HashFloats(data []float32) MemoKey {
	h := fnv.New64a()
	var buf [8]byte
	binary.LittleEndian.PutUint64(buf[:], uint64(len(data)))
	h.Write(buf[:])
	if len(data) > 0 {
		hashFloat(h, data[0])
		if len(data) > 1 {
			hashFloat(h, data[len(data)-1])
		}
	}
	return MemoKey(h.Sum64())
}

// HashInts computes a hash for an int slice (e.g., token IDs).
func HashInts(data []int) MemoKey {
	h := fnv.New64a()
	for _, v := range data {
		var buf [8]byte
		binary.LittleEndian.PutUint64(buf[:], uint64(v))
		h.Write(buf[:])
	}
	return MemoKey(h.Sum64())
}

func hashFloat(h hash.Hash64, f float32) {
	var buf [4]byte
	binary.LittleEndian.PutUint32(buf[:], uint32(int32(f*1000)))
	h.Write(buf[:])
}

// Get retrieves a cached result if available.
func (m *TensorMemo) Get(key MemoKey) (*Tensor, bool) {
	if !m.config.Enabled {
		return nil, false
	}

	m.mu.RLock()
	entry, ok := m.cache[key]
	m.mu.RUnlock()

	if !ok {
		m.misses++
		return nil, false
	}

	m.mu.Lock()
	entry.HitCount++
	entry.LastUsed = time.Now()
	m.mu.Unlock()

	m.hits++
	return entry.Result, true
}

// Store saves a computation result in the cache.
func (m *TensorMemo) Store(key MemoKey, result *Tensor) {
	if !m.config.Enabled || result == nil {
		return
	}

	sizeBytes := len(result.Data) * 4

	m.mu.Lock()
	defer m.mu.Unlock()

	// Check memory limit.
	maxBytes := int64(m.config.MaxMemoryMB) * 1024 * 1024
	if m.totalBytes+int64(sizeBytes) > maxBytes {
		m.evictLRU()
	}

	// Check entry count limit.
	if len(m.cache) >= m.config.MaxEntries {
		m.evictLRU()
	}

	// Deep copy the result.
	cached := NewTensor(result.Shape)
	copy(cached.Data, result.Data)

	m.cache[key] = &MemoEntry{
		Result:    cached,
		Created:   time.Now(),
		LastUsed:  time.Now(),
		SizeBytes: sizeBytes,
	}
	m.totalBytes += int64(sizeBytes)
}

// evictLRU removes the least recently used entry (must hold write lock).
func (m *TensorMemo) evictLRU() {
	var oldestKey MemoKey
	var oldestTime time.Time
	first := true

	for key, entry := range m.cache {
		if first || entry.LastUsed.Before(oldestTime) {
			oldestKey = key
			oldestTime = entry.LastUsed
			first = false
		}
	}

	if !first {
		if entry, ok := m.cache[oldestKey]; ok {
			m.totalBytes -= int64(entry.SizeBytes)
		}
		delete(m.cache, oldestKey)
	}
}

// Stats returns cache performance metrics.
func (m *TensorMemo) Stats() map[string]interface{} {
	m.mu.RLock()
	defer m.mu.RUnlock()

	hitRate := float64(0)
	total := m.hits + m.misses
	if total > 0 {
		hitRate = float64(m.hits) / float64(total) * 100
	}

	return map[string]interface{}{
		"entries":     len(m.cache),
		"max_entries": m.config.MaxEntries,
		"total_mb":    fmt.Sprintf("%.1f", float64(m.totalBytes)/(1024*1024)),
		"max_mb":      m.config.MaxMemoryMB,
		"hits":        m.hits,
		"misses":      m.misses,
		"hit_rate":    fmt.Sprintf("%.1f%%", hitRate),
	}
}

// Clear empties the memo cache.
func (m *TensorMemo) Clear() {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.cache = make(map[MemoKey]*MemoEntry, m.config.MaxEntries)
	m.totalBytes = 0
	m.hits = 0
	m.misses = 0
}
