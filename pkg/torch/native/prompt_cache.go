// prompt_cache.go implements KV cache prefix sharing for common prompt prefixes.
//
// WHAT: When multiple requests share the same system prompt or prefix, the
// KV cache computations for that prefix can be reused. This avoids
// recomputing attention for shared content.
//
// HOW: A trie-based cache stores computed KV states keyed by token prefix.
// On each request, we find the longest cached prefix, restore its KV state,
// and only compute the forward pass for the novel suffix tokens.
//
// SAVINGS: For a 2048-token system prompt, a cache hit means skipping 2048
// token forward passes. At 100ms prefill for 2048 tokens, this saves ~100ms
// per request.
package native

import (
	"fmt"
	"sync"
	"time"
)

// PromptCache stores precomputed prefix states for reuse.
type PromptCache struct {
	mu       sync.RWMutex
	entries  map[string]*PromptCacheEntry // Key: hash of token prefix
	maxSize  int                          // Maximum entries
	hits     int64
	misses   int64
}

// PromptCacheEntry stores the KV state for a token prefix.
type PromptCacheEntry struct {
	Prefix      []int     // The token prefix this entry represents
	PrefixLen   int       // Length of the prefix
	KVStates    []KVState // Per-layer KV cache states
	LastUsed    time.Time
	UseCount    int64
}

// KVState holds cached key/value vectors for one layer at a given prefix.
type KVState struct {
	Keys   []float32 // [prefixLen * headDim] flattened K cache
	Values []float32 // [prefixLen * headDim] flattened V cache
}

// NewPromptCache creates a prompt cache with the given max entries.
func NewPromptCache(maxSize int) *PromptCache {
	return &PromptCache{
		entries: make(map[string]*PromptCacheEntry),
		maxSize: maxSize,
	}
}

// Lookup finds the longest cached prefix for the given tokens.
// Returns the cached entry and the number of tokens that matched.
func (pc *PromptCache) Lookup(tokens []int) (*PromptCacheEntry, int) {
	pc.mu.RLock()
	defer pc.mu.RUnlock()

	// Try progressively shorter prefixes.
	bestEntry := (*PromptCacheEntry)(nil)
	bestLen := 0

	for _, entry := range pc.entries {
		matchLen := prefixMatch(tokens, entry.Prefix)
		if matchLen > bestLen {
			bestLen = matchLen
			bestEntry = entry
		}
	}

	if bestEntry != nil {
		bestEntry.LastUsed = time.Now()
		bestEntry.UseCount++
		pc.hits++
		return bestEntry, bestLen
	}

	pc.misses++
	return nil, 0
}

// Store saves a computed KV state for a token prefix.
func (pc *PromptCache) Store(tokens []int, kvStates []KVState) {
	pc.mu.Lock()
	defer pc.mu.Unlock()

	// Evict if at capacity (LRU).
	if len(pc.entries) >= pc.maxSize {
		pc.evictOldest()
	}

	key := prefixKey(tokens)
	prefix := make([]int, len(tokens))
	copy(prefix, tokens)

	pc.entries[key] = &PromptCacheEntry{
		Prefix:    prefix,
		PrefixLen: len(tokens),
		KVStates:  kvStates,
		LastUsed:  time.Now(),
	}
}

// Stats returns cache performance metrics.
func (pc *PromptCache) Stats() PromptCacheStats {
	pc.mu.RLock()
	defer pc.mu.RUnlock()

	return PromptCacheStats{
		Entries: len(pc.entries),
		MaxSize: pc.maxSize,
		Hits:    pc.hits,
		Misses:  pc.misses,
	}
}

// PromptCacheStats holds cache metrics.
type PromptCacheStats struct {
	Entries int
	MaxSize int
	Hits    int64
	Misses  int64
}

func (s PromptCacheStats) HitRate() float64 {
	total := s.Hits + s.Misses
	if total == 0 {
		return 0
	}
	return float64(s.Hits) / float64(total)
}

func (s PromptCacheStats) Print() {
	fmt.Printf("[PromptCache] %d/%d entries, %d hits, %d misses (%.1f%% hit rate)\n",
		s.Entries, s.MaxSize, s.Hits, s.Misses, s.HitRate()*100)
}

// evictOldest removes the least recently used entry.
func (pc *PromptCache) evictOldest() {
	var oldestKey string
	var oldestTime time.Time

	for key, entry := range pc.entries {
		if oldestKey == "" || entry.LastUsed.Before(oldestTime) {
			oldestKey = key
			oldestTime = entry.LastUsed
		}
	}

	if oldestKey != "" {
		delete(pc.entries, oldestKey)
	}
}

// prefixMatch returns the length of the common prefix between two token sequences.
func prefixMatch(a, b []int) int {
	maxLen := len(a)
	if len(b) < maxLen {
		maxLen = len(b)
	}
	for i := 0; i < maxLen; i++ {
		if a[i] != b[i] {
			return i
		}
	}
	return maxLen
}

// prefixKey generates a cache key from a token prefix.
func prefixKey(tokens []int) string {
	// Use first 64 tokens + length as key (fast, collision-acceptable).
	maxKeyLen := 64
	if len(tokens) < maxKeyLen {
		maxKeyLen = len(tokens)
	}
	key := fmt.Sprintf("%d:", len(tokens))
	for i := 0; i < maxKeyLen; i++ {
		key += fmt.Sprintf("%d,", tokens[i])
	}
	return key
}
