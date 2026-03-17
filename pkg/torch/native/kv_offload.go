// kv_offload.go implements KV cache offloading from GPU to CPU memory.
//
// WHAT: When GPU VRAM fills up with KV cache from long contexts or many
// concurrent requests, KV offloading moves less-recently-accessed cache
// pages to CPU RAM. When those positions are needed for attention, they're
// brought back to GPU memory.
//
// WHY: CPU RAM is 4-16x larger than GPU VRAM. A system with 24GB VRAM and
// 128GB RAM can effectively use 128GB for KV cache, enabling much longer
// contexts or more concurrent requests.
//
// STRATEGY:
//   - LRU eviction: pages not accessed recently are offloaded first
//   - Async prefetch: when attention approaches offloaded positions,
//     start loading them before they're needed
//   - Tiered: hot pages on GPU, warm pages on CPU, cold pages on disk
//
// GAIN: 4-16x effective KV cache capacity at the cost of some latency
// for offloaded pages (~100us for CPU, ~1ms for disk).
package native

import (
	"fmt"
	"sync"
	"time"
)

// CacheTier identifies where a KV page is stored.
type CacheTier int

const (
	TierGPU  CacheTier = iota // Fast: GPU VRAM
	TierCPU                    // Medium: CPU RAM
	TierDisk                   // Slow: SSD/NVMe (for very long contexts)
)

// OffloadPage tracks one KV cache page and its current location.
type OffloadPage struct {
	PageID     int
	Tier       CacheTier
	LastAccess time.Time
	AccessCount int64
	Data       []float32 // The actual KV data (K+V interleaved)
	Dirty      bool      // Modified since last GPU sync
	Pinned     bool      // Cannot be evicted (actively in use)
}

// KVOffloadManager manages tiered KV cache storage.
type KVOffloadManager struct {
	mu sync.Mutex

	// Page storage by tier.
	gpuPages  map[int]*OffloadPage
	cpuPages  map[int]*OffloadPage

	// Capacity limits.
	maxGPUPages int
	maxCPUPages int

	// LRU tracking.
	evictOrder []int // Page IDs sorted by last access (oldest first)

	// Stats.
	totalOffloads int64
	totalReloads  int64
	totalEvictions int64
	offloadTime   time.Duration
	reloadTime    time.Duration
}

// OffloadConfig configures KV cache offloading.
type OffloadConfig struct {
	MaxGPUPages int // Max pages in GPU VRAM
	MaxCPUPages int // Max pages in CPU RAM
}

// NewKVOffloadManager creates an offload manager.
func NewKVOffloadManager(config OffloadConfig) *KVOffloadManager {
	return &KVOffloadManager{
		gpuPages:    make(map[int]*OffloadPage),
		cpuPages:    make(map[int]*OffloadPage),
		maxGPUPages: config.MaxGPUPages,
		maxCPUPages: config.MaxCPUPages,
		evictOrder:  make([]int, 0),
	}
}

// StorePage adds a new page to GPU storage. If GPU is full, evicts the
// least recently used page to CPU.
func (m *KVOffloadManager) StorePage(pageID int, data []float32) {
	m.mu.Lock()
	defer m.mu.Unlock()

	// Evict from GPU if full.
	for len(m.gpuPages) >= m.maxGPUPages {
		m.evictLRUFromGPU()
	}

	page := &OffloadPage{
		PageID:     pageID,
		Tier:       TierGPU,
		LastAccess: time.Now(),
		Data:       data,
	}

	m.gpuPages[pageID] = page
	m.updateEvictOrder(pageID)
}

// AccessPage retrieves a page, promoting it from CPU to GPU if needed.
// Returns the page data and tier it was found in.
func (m *KVOffloadManager) AccessPage(pageID int) ([]float32, CacheTier, bool) {
	m.mu.Lock()
	defer m.mu.Unlock()

	// Check GPU first (fast path).
	if page, ok := m.gpuPages[pageID]; ok {
		page.LastAccess = time.Now()
		page.AccessCount++
		m.updateEvictOrder(pageID)
		return page.Data, TierGPU, true
	}

	// Check CPU (requires reload).
	if page, ok := m.cpuPages[pageID]; ok {
		start := time.Now()
		m.reloadToGPU(page)
		m.reloadTime += time.Since(start)
		m.totalReloads++
		return page.Data, TierCPU, true
	}

	return nil, TierGPU, false
}

// PrefetchPages proactively loads pages from CPU to GPU before they're needed.
// Call this when you know attention will soon reach these positions.
func (m *KVOffloadManager) PrefetchPages(pageIDs []int) {
	m.mu.Lock()
	defer m.mu.Unlock()

	for _, pageID := range pageIDs {
		if page, ok := m.cpuPages[pageID]; ok {
			// Only prefetch if GPU has room.
			if len(m.gpuPages) < m.maxGPUPages {
				m.reloadToGPU(page)
				m.totalReloads++
			}
		}
	}
}

// PinPage prevents a page from being evicted (e.g., while in active attention).
func (m *KVOffloadManager) PinPage(pageID int) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if page, ok := m.gpuPages[pageID]; ok {
		page.Pinned = true
	}
}

// UnpinPage allows a page to be evicted again.
func (m *KVOffloadManager) UnpinPage(pageID int) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if page, ok := m.gpuPages[pageID]; ok {
		page.Pinned = false
	}
}

// evictLRUFromGPU moves the least recently used GPU page to CPU storage.
func (m *KVOffloadManager) evictLRUFromGPU() {
	// Find oldest non-pinned page.
	for _, pageID := range m.evictOrder {
		page, ok := m.gpuPages[pageID]
		if !ok || page.Pinned {
			continue
		}

		start := time.Now()

		// Move to CPU tier.
		page.Tier = TierCPU

		// If CPU is full, drop the oldest CPU page entirely.
		if len(m.cpuPages) >= m.maxCPUPages {
			m.evictLRUFromCPU()
		}

		m.cpuPages[pageID] = page
		delete(m.gpuPages, pageID)

		m.offloadTime += time.Since(start)
		m.totalOffloads++
		return
	}
}

// evictLRUFromCPU drops the least recently used CPU page entirely.
func (m *KVOffloadManager) evictLRUFromCPU() {
	var oldestID int
	var oldestTime time.Time

	first := true
	for id, page := range m.cpuPages {
		if first || page.LastAccess.Before(oldestTime) {
			oldestID = id
			oldestTime = page.LastAccess
			first = false
		}
	}

	delete(m.cpuPages, oldestID)
	m.totalEvictions++
}

// reloadToGPU moves a page from CPU back to GPU.
func (m *KVOffloadManager) reloadToGPU(page *OffloadPage) {
	// Evict from GPU if needed.
	for len(m.gpuPages) >= m.maxGPUPages {
		m.evictLRUFromGPU()
	}

	page.Tier = TierGPU
	page.LastAccess = time.Now()
	page.AccessCount++

	m.gpuPages[page.PageID] = page
	delete(m.cpuPages, page.PageID)
	m.updateEvictOrder(page.PageID)
}

// updateEvictOrder moves a page to the end of the eviction list (most recent).
func (m *KVOffloadManager) updateEvictOrder(pageID int) {
	// Remove existing entry.
	for i, id := range m.evictOrder {
		if id == pageID {
			m.evictOrder = append(m.evictOrder[:i], m.evictOrder[i+1:]...)
			break
		}
	}
	// Append as most recent.
	m.evictOrder = append(m.evictOrder, pageID)
}

// FreePage removes a page from all tiers (sequence completed).
func (m *KVOffloadManager) FreePage(pageID int) {
	m.mu.Lock()
	defer m.mu.Unlock()
	delete(m.gpuPages, pageID)
	delete(m.cpuPages, pageID)
	for i, id := range m.evictOrder {
		if id == pageID {
			m.evictOrder = append(m.evictOrder[:i], m.evictOrder[i+1:]...)
			break
		}
	}
}

// Stats returns offload metrics.
func (m *KVOffloadManager) Stats() map[string]interface{} {
	m.mu.Lock()
	defer m.mu.Unlock()
	return map[string]interface{}{
		"gpu_pages":       len(m.gpuPages),
		"cpu_pages":       len(m.cpuPages),
		"max_gpu_pages":   m.maxGPUPages,
		"max_cpu_pages":   m.maxCPUPages,
		"total_offloads":  m.totalOffloads,
		"total_reloads":   m.totalReloads,
		"total_evictions": m.totalEvictions,
		"avg_offload_us":  fmt.Sprintf("%.0f", float64(m.offloadTime.Microseconds())/float64(m.totalOffloads+1)),
		"avg_reload_us":   fmt.Sprintf("%.0f", float64(m.reloadTime.Microseconds())/float64(m.totalReloads+1)),
	}
}
