// prefix_cache.go implements KV cache state reuse for shared system prompts.
// When multiple requests share the same system prompt (common in agent swarms),
// the first request processes the prompt normally and caches the resulting KV state.
// Subsequent requests with the same prompt restore the cached state, skipping
// prompt processing entirely. This can save 100-500ms per request.
package torch

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"sync"

	"github.com/David2024patton/iTaKTorch/pkg/torch/llama"
)

// PrefixCacheEntry holds a cached KV state for a system prompt.
type PrefixCacheEntry struct {
	Hash      string        // SHA-256 of the prompt text
	StateData []byte        // Serialized KV cache state (from StateSeqGetData)
	Tokens    []llama.Token // The tokens that produced this state
	HitCount  uint64        // Number of cache hits
}

// PrefixCache stores KV cache states keyed by prompt hash.
type PrefixCache struct {
	mu      sync.RWMutex
	entries map[string]*PrefixCacheEntry
	maxSize int // Maximum number of cached entries
	hits    uint64
	misses  uint64
}

// PrefixCacheStats exposes cache performance metrics.
type PrefixCacheStats struct {
	Entries int
	Hits    uint64
	Misses  uint64
	HitRate float64
}

// NewPrefixCache creates a cache with the given max entries.
func NewPrefixCache(maxEntries int) *PrefixCache {
	if maxEntries <= 0 {
		maxEntries = 16
	}
	return &PrefixCache{
		entries: make(map[string]*PrefixCacheEntry),
		maxSize: maxEntries,
	}
}

// hashPrompt generates a SHA-256 hash of the prompt text.
func hashPrompt(prompt string) string {
	h := sha256.Sum256([]byte(prompt))
	return hex.EncodeToString(h[:16]) // Use first 16 bytes (128 bits) for key
}

// Lookup checks if a cached KV state exists for the given prompt.
// Returns the entry and true if found, nil and false otherwise.
func (pc *PrefixCache) Lookup(prompt string) (*PrefixCacheEntry, bool) {
	hash := hashPrompt(prompt)
	pc.mu.Lock()
	defer pc.mu.Unlock()

	entry, ok := pc.entries[hash]
	if ok {
		pc.hits++
		entry.HitCount++
		return entry, true
	}
	return nil, false
}

// Save captures the current KV state for a prompt and stores it in the cache.
func (pc *PrefixCache) Save(ctx llama.Context, prompt string, tokens []llama.Token) error {
	hash := hashPrompt(prompt)

	// Get the size needed to store this sequence's state.
	seqSize := llama.StateSeqGetSize(ctx, 0)
	if seqSize == 0 {
		return fmt.Errorf("state seq size is zero, nothing to cache")
	}

	// Allocate buffer and copy state.
	stateData := make([]byte, seqSize)
	written := llama.StateSeqGetData(ctx, stateData, 0)
	if written == 0 {
		return fmt.Errorf("failed to get sequence state data")
	}

	entry := &PrefixCacheEntry{
		Hash:      hash,
		StateData: stateData[:written],
		Tokens:    make([]llama.Token, len(tokens)),
	}
	copy(entry.Tokens, tokens)

	pc.mu.Lock()
	defer pc.mu.Unlock()

	// Evict LRU if at capacity.
	if len(pc.entries) >= pc.maxSize {
		pc.evictLRU()
	}

	pc.entries[hash] = entry
	pc.misses++ // The save implies a miss on this prompt
	return nil
}

// Restore loads a cached KV state into the context.
// Returns the cached tokens so the engine knows the prompt is already processed.
func (pc *PrefixCache) Restore(ctx llama.Context, entry *PrefixCacheEntry) ([]llama.Token, error) {
	written := llama.StateSeqSetData(ctx, entry.StateData, 0)
	if written == 0 {
		return nil, fmt.Errorf("failed to restore sequence state data")
	}
	return entry.Tokens, nil
}

// Stats returns cache performance metrics.
func (pc *PrefixCache) Stats() PrefixCacheStats {
	pc.mu.RLock()
	defer pc.mu.RUnlock()

	total := pc.hits + pc.misses
	hitRate := float64(0)
	if total > 0 {
		hitRate = float64(pc.hits) / float64(total) * 100
	}

	return PrefixCacheStats{
		Entries: len(pc.entries),
		Hits:    pc.hits,
		Misses:  pc.misses,
		HitRate: hitRate,
	}
}

// evictLRU removes the least-used entry from the cache.
// Must be called with pc.mu held.
func (pc *PrefixCache) evictLRU() {
	var minKey string
	var minHits uint64 = ^uint64(0) // max uint64

	for key, entry := range pc.entries {
		if entry.HitCount < minHits {
			minHits = entry.HitCount
			minKey = key
		}
	}

	if minKey != "" {
		delete(pc.entries, minKey)
	}
}
