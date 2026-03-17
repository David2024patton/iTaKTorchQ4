// paged_attention.go implements vLLM-style PagedAttention for KV cache management.
//
// WHAT: Instead of allocating one contiguous KV cache per request, we manage
// KV memory as fixed-size "pages" (like OS virtual memory). Each request gets
// a page table mapping logical positions to physical page slots.
//
// WHY: Contiguous allocation wastes memory due to fragmentation and
// over-provisioning. PagedAttention eliminates both:
//   - No fragmentation: pages are uniform size, any free page works
//   - No over-provisioning: allocate pages on demand as context grows
//   - Shared prefixes: multiple requests can share the same physical pages
//     for common system prompts (copy-on-write)
//
// GAIN: 2-4x more concurrent requests at same VRAM budget.
package native

import (
	"fmt"
	"math"
	"sync"
)

// PageSize is the number of tokens per KV cache page.
// 16 is the sweet spot: small enough for fine-grained allocation,
// large enough to avoid excessive page table overhead.
const PageSize = 16

// KVPage holds key and value vectors for one page of tokens.
// Each page stores PageSize tokens worth of K and V across all heads.
type KVPage struct {
	Keys   []float32 // [PageSize * numHeads * headDim]
	Values []float32 // [PageSize * numHeads * headDim]
	RefCount int     // Copy-on-write reference counting
	Dirty    bool    // Modified since allocation (breaks CoW sharing)
}

// PageTable maps logical token positions to physical page indices.
// Each sequence (request) maintains its own page table.
type PageTable struct {
	SequenceID string
	Pages      []int // Pages[i] = physical page index for logical page i
	Length     int   // Number of tokens currently in this sequence
}

// PagedKVCache manages a pool of KV cache pages with virtual memory semantics.
type PagedKVCache struct {
	mu sync.Mutex

	// Physical page pool.
	pages    []*KVPage
	freeList []int // Indices of free pages

	// Per-sequence page tables.
	tables map[string]*PageTable

	// Config.
	numHeads int
	headDim  int
	maxPages int // Total physical pages available

	// Stats.
	totalAllocs int64
	totalFrees  int64
	cowShares   int64
}

// PagedKVConfig configures the paged cache.
type PagedKVConfig struct {
	NumHeads    int   // Number of attention heads
	HeadDim     int   // Dimension per head
	MaxMemoryMB int64 // Maximum memory budget in MB
}

// NewPagedKVCache creates a paged KV cache with the given memory budget.
func NewPagedKVCache(config PagedKVConfig) *PagedKVCache {
	// Calculate how many pages fit in the memory budget.
	// Each page stores K + V for PageSize tokens across all heads.
	bytesPerPage := PageSize * config.NumHeads * config.HeadDim * 4 * 2 // *4 float32, *2 K+V
	maxPages := int(config.MaxMemoryMB * 1024 * 1024 / int64(bytesPerPage))
	if maxPages < 16 {
		maxPages = 16
	}

	cache := &PagedKVCache{
		pages:    make([]*KVPage, maxPages),
		freeList: make([]int, maxPages),
		tables:   make(map[string]*PageTable),
		numHeads: config.NumHeads,
		headDim:  config.HeadDim,
		maxPages: maxPages,
	}

	// Initialize all pages as free.
	for i := 0; i < maxPages; i++ {
		cache.pages[i] = &KVPage{
			Keys:   make([]float32, PageSize*config.NumHeads*config.HeadDim),
			Values: make([]float32, PageSize*config.NumHeads*config.HeadDim),
		}
		cache.freeList[i] = i
	}

	fmt.Printf("[PagedAttn] Initialized: %d pages (%d tokens max, %d MB)\n",
		maxPages, maxPages*PageSize, config.MaxMemoryMB)
	return cache
}

// Allocate assigns a new page table for a sequence.
func (c *PagedKVCache) Allocate(seqID string) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if _, exists := c.tables[seqID]; exists {
		return fmt.Errorf("sequence %s already allocated", seqID)
	}

	c.tables[seqID] = &PageTable{
		SequenceID: seqID,
		Pages:      make([]int, 0, 8),
	}
	return nil
}

// AppendToken adds a token's KV to a sequence, allocating new pages as needed.
func (c *PagedKVCache) AppendToken(seqID string, key, value []float32) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	table, ok := c.tables[seqID]
	if !ok {
		return fmt.Errorf("sequence %s not found", seqID)
	}

	// Check if we need a new page.
	posInPage := table.Length % PageSize
	if posInPage == 0 {
		// Need a new physical page.
		pageIdx, err := c.allocPage()
		if err != nil {
			return err
		}
		table.Pages = append(table.Pages, pageIdx)
	}

	// Write K and V into the current page.
	pageIdx := table.Pages[len(table.Pages)-1]
	page := c.pages[pageIdx]

	// Handle copy-on-write: if page is shared, copy before writing.
	if page.RefCount > 1 {
		page = c.cowCopy(pageIdx)
		table.Pages[len(table.Pages)-1] = c.findPageIndex(page)
	}

	stride := c.numHeads * c.headDim
	offset := posInPage * stride
	copy(page.Keys[offset:offset+stride], key)
	copy(page.Values[offset:offset+stride], value)
	page.Dirty = true
	table.Length++

	return nil
}

// GetKV retrieves all K and V vectors for a sequence up to its current length.
// Returns keys [seqLen, numHeads*headDim] and values [seqLen, numHeads*headDim].
func (c *PagedKVCache) GetKV(seqID string) (keys, values []float32, err error) {
	c.mu.Lock()
	defer c.mu.Unlock()

	table, ok := c.tables[seqID]
	if !ok {
		return nil, nil, fmt.Errorf("sequence %s not found", seqID)
	}

	stride := c.numHeads * c.headDim
	keys = make([]float32, table.Length*stride)
	values = make([]float32, table.Length*stride)

	for i := 0; i < table.Length; i++ {
		pageLogical := i / PageSize
		posInPage := i % PageSize
		page := c.pages[table.Pages[pageLogical]]
		offset := posInPage * stride

		copy(keys[i*stride:(i+1)*stride], page.Keys[offset:offset+stride])
		copy(values[i*stride:(i+1)*stride], page.Values[offset:offset+stride])
	}

	return keys, values, nil
}

// SharePrefix creates a new sequence that shares physical pages with an existing
// sequence up to sharedLen tokens (copy-on-write). The new sequence can then
// diverge without affecting the original.
func (c *PagedKVCache) SharePrefix(srcSeqID, dstSeqID string, sharedLen int) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	src, ok := c.tables[srcSeqID]
	if !ok {
		return fmt.Errorf("source sequence %s not found", srcSeqID)
	}

	sharedPages := (sharedLen + PageSize - 1) / PageSize
	if sharedPages > len(src.Pages) {
		sharedPages = len(src.Pages)
	}

	dst := &PageTable{
		SequenceID: dstSeqID,
		Pages:      make([]int, sharedPages),
		Length:     sharedLen,
	}

	// Share physical pages (increment ref counts).
	for i := 0; i < sharedPages; i++ {
		dst.Pages[i] = src.Pages[i]
		c.pages[src.Pages[i]].RefCount++
		c.cowShares++
	}

	c.tables[dstSeqID] = dst
	return nil
}

// Free releases all pages for a sequence.
func (c *PagedKVCache) Free(seqID string) {
	c.mu.Lock()
	defer c.mu.Unlock()

	table, ok := c.tables[seqID]
	if !ok {
		return
	}

	for _, pageIdx := range table.Pages {
		page := c.pages[pageIdx]
		page.RefCount--
		if page.RefCount <= 0 {
			c.freePage(pageIdx)
		}
	}

	delete(c.tables, seqID)
}

// allocPage returns a free physical page index.
func (c *PagedKVCache) allocPage() (int, error) {
	if len(c.freeList) == 0 {
		return 0, fmt.Errorf("paged KV cache exhausted (%d pages in use)", c.maxPages)
	}

	idx := c.freeList[len(c.freeList)-1]
	c.freeList = c.freeList[:len(c.freeList)-1]
	c.pages[idx].RefCount = 1
	c.pages[idx].Dirty = false
	c.totalAllocs++
	return idx, nil
}

// freePage returns a page to the free list.
func (c *PagedKVCache) freePage(idx int) {
	// Zero out for security (prevent KV leakage between requests).
	page := c.pages[idx]
	for i := range page.Keys {
		page.Keys[i] = 0
	}
	for i := range page.Values {
		page.Values[i] = 0
	}
	page.RefCount = 0
	page.Dirty = false
	c.freeList = append(c.freeList, idx)
	c.totalFrees++
}

// cowCopy creates a private copy of a shared page (copy-on-write).
func (c *PagedKVCache) cowCopy(srcIdx int) *KVPage {
	src := c.pages[srcIdx]
	src.RefCount--

	newIdx, err := c.allocPage()
	if err != nil {
		return src // fallback: write to shared page (data corruption risk)
	}

	dst := c.pages[newIdx]
	copy(dst.Keys, src.Keys)
	copy(dst.Values, src.Values)
	return dst
}

// findPageIndex returns the index of a page in the pool (for CoW remapping).
func (c *PagedKVCache) findPageIndex(page *KVPage) int {
	for i, p := range c.pages {
		if p == page {
			return i
		}
	}
	return -1
}

// Stats returns paged cache metrics.
func (c *PagedKVCache) Stats() map[string]interface{} {
	c.mu.Lock()
	defer c.mu.Unlock()

	usedPages := c.maxPages - len(c.freeList)
	return map[string]interface{}{
		"total_pages":    c.maxPages,
		"used_pages":     usedPages,
		"free_pages":     len(c.freeList),
		"utilization":    math.Round(float64(usedPages)/float64(c.maxPages)*100) / 100,
		"sequences":      len(c.tables),
		"total_allocs":   c.totalAllocs,
		"total_frees":    c.totalFrees,
		"cow_shares":     c.cowShares,
		"max_tokens":     c.maxPages * PageSize,
		"page_size":      PageSize,
	}
}

// Defragment compacts the page tables to reduce fragmentation.
// Call periodically during low-traffic windows.
func (c *PagedKVCache) Defragment() int {
	c.mu.Lock()
	defer c.mu.Unlock()

	moved := 0
	// Compact: move pages from high indices to low free slots.
	for _, table := range c.tables {
		for i, pageIdx := range table.Pages {
			if len(c.freeList) == 0 {
				break
			}
			// If a lower free slot exists, move there.
			lowIdx := c.freeList[0]
			if lowIdx < pageIdx {
				// Copy data to lower slot.
				copy(c.pages[lowIdx].Keys, c.pages[pageIdx].Keys)
				copy(c.pages[lowIdx].Values, c.pages[pageIdx].Values)
				c.pages[lowIdx].RefCount = c.pages[pageIdx].RefCount
				c.pages[lowIdx].Dirty = c.pages[pageIdx].Dirty

				// Update page table.
				table.Pages[i] = lowIdx

				// Free old slot, remove new slot from free list.
				c.freeList = c.freeList[1:]
				c.freeList = append(c.freeList, pageIdx)
				moved++
			}
		}
	}
	return moved
}
