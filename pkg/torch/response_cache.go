// response_cache.go implements an LRU cache for identical inference requests.
//
// WHY: If the same prompt comes in twice (common with system prompts, health
// checks, or retried requests), we return the cached response instantly instead
// of running the model again. This saves GPU cycles and returns in <1ms.
//
// The cache key is a SHA256 hash of model + messages + generation params.
// Entries expire after 5 minutes by default.
package torch

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"sync"
	"time"
)

// CacheStats holds current cache performance metrics.
type CacheStats struct {
	Entries    int     `json:"entries"`
	MaxEntries int     `json:"max_entries"`
	Hits       int64   `json:"hits"`
	Misses     int64   `json:"misses"`
	HitRate    float64 `json:"hit_rate"`
}

// ResponseCache is a thread-safe LRU cache for inference responses.
type ResponseCache struct {
	mu         sync.RWMutex
	entries    map[string]*cacheEntry
	maxEntries int
	ttl        time.Duration
	hits       int64
	misses     int64
	seq        uint64 // monotonic insertion counter for deterministic LRU
}

type cacheEntry struct {
	text    string
	metrics *InferenceMetrics
	created time.Time
	seq     uint64 // insertion order (lower = older)
}

// NewResponseCache creates a new response cache.
// maxEntries is the max number of cached responses. TTL defaults to 5 minutes.
func NewResponseCache(maxEntries int, ttl ...time.Duration) *ResponseCache {
	if maxEntries <= 0 {
		maxEntries = 256
	}
	cacheTTL := 5 * time.Minute
	if len(ttl) > 0 && ttl[0] > 0 {
		cacheTTL = ttl[0]
	}
	return &ResponseCache{
		entries:    make(map[string]*cacheEntry, maxEntries),
		maxEntries: maxEntries,
		ttl:        cacheTTL,
	}
}

// Get retrieves a cached response by key.
// Returns (text, metrics, true) on cache hit, or ("", nil, false) on miss.
func (rc *ResponseCache) Get(key string) (string, *InferenceMetrics, bool) {
	rc.mu.RLock()
	entry, ok := rc.entries[key]
	rc.mu.RUnlock()

	if !ok {
		rc.mu.Lock()
		rc.misses++
		rc.mu.Unlock()
		return "", nil, false
	}

	// Check TTL.
	if time.Since(entry.created) > rc.ttl {
		rc.mu.Lock()
		delete(rc.entries, key)
		rc.misses++
		rc.mu.Unlock()
		return "", nil, false
	}

	rc.mu.Lock()
	rc.hits++
	rc.mu.Unlock()
	return entry.text, entry.metrics, true
}

// Put stores a response in the cache.
func (rc *ResponseCache) Put(key, text string, metrics *InferenceMetrics) {
	rc.mu.Lock()
	defer rc.mu.Unlock()

	// Evict oldest entry if at capacity.
	if len(rc.entries) >= rc.maxEntries {
		rc.evictOldest()
	}

	rc.seq++
	rc.entries[key] = &cacheEntry{
		text:    text,
		metrics: metrics,
		created: time.Now(),
		seq:     rc.seq,
	}
}

// Clear removes all entries and resets counters.
func (rc *ResponseCache) Clear() {
	rc.mu.Lock()
	defer rc.mu.Unlock()
	rc.entries = make(map[string]*cacheEntry, rc.maxEntries)
	rc.hits = 0
	rc.misses = 0
}

// Stats returns current cache performance metrics.
func (rc *ResponseCache) Stats() CacheStats {
	rc.mu.RLock()
	defer rc.mu.RUnlock()

	total := rc.hits + rc.misses
	hitRate := 0.0
	if total > 0 {
		hitRate = float64(rc.hits) / float64(total) * 100
	}

	return CacheStats{
		Entries:    len(rc.entries),
		MaxEntries: rc.maxEntries,
		Hits:       rc.hits,
		Misses:     rc.misses,
		HitRate:    hitRate,
	}
}

// Size returns the current number of cached entries.
func (rc *ResponseCache) Size() int {
	rc.mu.RLock()
	defer rc.mu.RUnlock()
	return len(rc.entries)
}

// CacheKey builds a deterministic cache key from model, messages, and params.
// Uses SHA256 to keep keys fixed-size regardless of prompt length.
func CacheKey(model string, messages []ChatMessage, params CompletionParams) string {
	h := sha256.New()
	fmt.Fprintf(h, "model=%s|", model)
	for _, m := range messages {
		fmt.Fprintf(h, "%s:%s|", m.Role, m.Content)
	}
	fmt.Fprintf(h, "max=%d|temp=%.4f|top_p=%.4f",
		params.MaxTokens, params.Temperature, params.TopP)
	if len(params.Stop) > 0 {
		stopJSON, _ := json.Marshal(params.Stop)
		h.Write(stopJSON)
	}
	return fmt.Sprintf("%x", h.Sum(nil))
}

// evictOldest removes the oldest entry by insertion order. Called with lock held.
func (rc *ResponseCache) evictOldest() {
	var oldestKey string
	var oldestSeq uint64

	for k, v := range rc.entries {
		if oldestKey == "" || v.seq < oldestSeq {
			oldestKey = k
			oldestSeq = v.seq
		}
	}
	if oldestKey != "" {
		delete(rc.entries, oldestKey)
	}
}
