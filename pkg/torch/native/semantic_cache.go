// semantic_cache.go implements embedding-similarity-based response caching.
//
// WHAT: Traditional caching uses exact string matching. Semantic caching
// embeds the prompt and finds cached responses where the prompt embedding
// has high cosine similarity (> threshold). This means "What is Python?"
// and "Tell me about the Python language" can share a cached response.
//
// SAVINGS: Avoids full inference for semantically identical queries.
// Especially valuable for common questions in customer support,
// FAQ bots, and repeated system-prompt-heavy workloads.
package native

import (
	"fmt"
	"sync"
	"time"
)

// SemanticCacheEntry stores a cached response with its prompt embedding.
type SemanticCacheEntry struct {
	PromptHash    string    // Quick hash for exact match fast path
	PromptEmbed   []float32 // Embedding of the prompt
	Response      string    // Cached response text
	CreatedAt     time.Time
	HitCount      int
	Model         string    // Model that produced the response
}

// SemanticCacheConfig controls cache behavior.
type SemanticCacheConfig struct {
	Enabled          bool
	SimilarityThresh float32 // Min cosine similarity to count as a hit (default: 0.95)
	MaxEntries       int     // Max cache entries
	TTL              time.Duration // Entry expiration
	EmbeddingDim     int     // Dimension of prompt embeddings
}

// DefaultSemanticCacheConfig returns recommended settings.
func DefaultSemanticCacheConfig() SemanticCacheConfig {
	return SemanticCacheConfig{
		Enabled:          true,
		SimilarityThresh: 0.95,
		MaxEntries:       1000,
		TTL:              1 * time.Hour,
		EmbeddingDim:     384,
	}
}

// SemanticCache stores and retrieves responses by prompt similarity.
type SemanticCache struct {
	mu      sync.RWMutex
	config  SemanticCacheConfig
	entries []*SemanticCacheEntry
	hits    int64
	misses  int64
}

// NewSemanticCache creates a semantic response cache.
func NewSemanticCache(config SemanticCacheConfig) *SemanticCache {
	fmt.Printf("[SemanticCache] Enabled: thresh=%.2f, max=%d, ttl=%s\n",
		config.SimilarityThresh, config.MaxEntries, config.TTL)
	return &SemanticCache{
		config:  config,
		entries: make([]*SemanticCacheEntry, 0, config.MaxEntries),
	}
}

// Lookup finds a cached response matching the prompt embedding.
// Returns the cached response and true if found, empty string and false otherwise.
func (sc *SemanticCache) Lookup(promptEmbed []float32) (string, bool) {
	sc.mu.RLock()
	defer sc.mu.RUnlock()

	now := time.Now()
	bestSim := float32(0)
	bestIdx := -1

	for i, entry := range sc.entries {
		// Check TTL.
		if now.Sub(entry.CreatedAt) > sc.config.TTL {
			continue
		}

		sim := CosineSimilarity(promptEmbed, entry.PromptEmbed)
		if sim > bestSim && sim >= sc.config.SimilarityThresh {
			bestSim = sim
			bestIdx = i
		}
	}

	if bestIdx >= 0 {
		sc.entries[bestIdx].HitCount++
		return sc.entries[bestIdx].Response, true
	}

	return "", false
}

// Store adds a response to the cache.
func (sc *SemanticCache) Store(promptEmbed []float32, response, model string) {
	sc.mu.Lock()
	defer sc.mu.Unlock()

	// Evict expired entries.
	sc.evictExpired()

	// Evict LRU if at capacity.
	if len(sc.entries) >= sc.config.MaxEntries {
		sc.evictLRU()
	}

	entry := &SemanticCacheEntry{
		PromptEmbed: make([]float32, len(promptEmbed)),
		Response:    response,
		CreatedAt:   time.Now(),
		Model:       model,
	}
	copy(entry.PromptEmbed, promptEmbed)

	sc.entries = append(sc.entries, entry)
}

// RecordHit increments the hit counter.
func (sc *SemanticCache) RecordHit() { sc.hits++ }

// RecordMiss increments the miss counter.
func (sc *SemanticCache) RecordMiss() { sc.misses++ }

// evictExpired removes entries past their TTL.
func (sc *SemanticCache) evictExpired() {
	now := time.Now()
	filtered := sc.entries[:0]
	for _, entry := range sc.entries {
		if now.Sub(entry.CreatedAt) <= sc.config.TTL {
			filtered = append(filtered, entry)
		}
	}
	sc.entries = filtered
}

// evictLRU removes the least recently hit entry.
func (sc *SemanticCache) evictLRU() {
	if len(sc.entries) == 0 {
		return
	}

	minHits := sc.entries[0].HitCount
	minIdx := 0
	for i, entry := range sc.entries {
		if entry.HitCount < minHits {
			minHits = entry.HitCount
			minIdx = i
		}
	}

	// Remove by swapping with last.
	sc.entries[minIdx] = sc.entries[len(sc.entries)-1]
	sc.entries = sc.entries[:len(sc.entries)-1]
}

// Stats returns cache performance metrics.
func (sc *SemanticCache) Stats() map[string]interface{} {
	sc.mu.RLock()
	defer sc.mu.RUnlock()

	hitRate := float64(0)
	total := sc.hits + sc.misses
	if total > 0 {
		hitRate = float64(sc.hits) / float64(total) * 100
	}

	return map[string]interface{}{
		"entries":   len(sc.entries),
		"max_size":  sc.config.MaxEntries,
		"hits":      sc.hits,
		"misses":    sc.misses,
		"hit_rate":  fmt.Sprintf("%.1f%%", hitRate),
		"threshold": sc.config.SimilarityThresh,
	}
}

// Clear empties the cache.
func (sc *SemanticCache) Clear() {
	sc.mu.Lock()
	defer sc.mu.Unlock()
	sc.entries = sc.entries[:0]
	sc.hits = 0
	sc.misses = 0
}
