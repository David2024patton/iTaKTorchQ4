// kv_prefetch.go implements Predictive KV Cache Prefetching.
//
// WHAT: As models scale to massive context windows (128K-1M tokens),
// GPU VRAM cannot hold all KV cache. Offloading to CPU RAM or NVMe
// causes severe latency spikes when that context is referenced later.
//
// HOW: We implement a memory architecture inspired by 2026 neoclouds
// (like Lightbits' LightInferra). Instead of reacting to page faults,
// we track access patterns across the memory stack. When a new prompt
// arrives, the orchestrator immediately evaluates its prefix and signals
// the PrefetchManager to begin streaming required offloaded KV blocks
// into GPU memory asynchronously, *before* the computation reaches
// those tokens.
//
// WHY: Converts memory-bound stalls into compute-hidden async DMA
// transfers. Keeps the GPUs saturated running math instead of waiting
// on PCIe/NVMe interfaces.
package native

import (
	"crypto/sha256"
	"encoding/hex"
	"sync"
	"time"
)

// BlockLocation represents where a KV block is currently stored.
type BlockLocation int

const (
	LocGPU BlockLocation = iota
	LocRAM
	LocNVMe
)

// KVBlockMeta tracks the location and access history of a KV block.
type KVBlockMeta struct {
	BlockID     string
	Location    BlockLocation
	Size        int64
	LastUsed    time.Time
	AccessCount int
	
	// Simulated pointers for pure Go 
	RAMData  []byte
	GPUIndex int
}

// PrefetchManager coordinates moving data between NVMe -> RAM -> GPU ahead of time.
type PrefetchManager struct {
	mu           sync.RWMutex
	blockMap     map[string]*KVBlockMeta
	prefetchChan chan string
	wg           sync.WaitGroup
	
	maxGPUMemory int64
	usedGPUMemory int64
}

// NewPrefetchManager creates a new background predictive loader.
func NewPrefetchManager(gpuMemoryBudget int64) *PrefetchManager {
	pm := &PrefetchManager{
		blockMap:      make(map[string]*KVBlockMeta),
		prefetchChan:  make(chan string, 1024), // High capacity queue for async fetches
		maxGPUMemory:  gpuMemoryBudget,
	}
	
	// Start background prefetch worker
	pm.wg.Add(1)
	go pm.prefetchWorker()
	
	return pm
}

// Stop shuts down the background DMA simulator.
func (pm *PrefetchManager) Stop() {
	close(pm.prefetchChan)
	pm.wg.Wait()
}

// RegisterBlock tells the manager about a newly generated KV block.
func (pm *PrefetchManager) RegisterBlock(prefixTokens []int32, gpuIndex int, size int64) string {
	hasher := sha256.New()
	for _, t := range prefixTokens {
		hasher.Write([]byte{byte(t >> 24), byte(t >> 16), byte(t >> 8), byte(t)})
	}
	blockID := hex.EncodeToString(hasher.Sum(nil))
	
	pm.mu.Lock()
	defer pm.mu.Lock()
	
	pm.blockMap[blockID] = &KVBlockMeta{
		BlockID:     blockID,
		Location:    LocGPU,
		Size:        size,
		LastUsed:    time.Now(),
		AccessCount: 1,
		GPUIndex:    gpuIndex,
	}
	pm.usedGPUMemory += size
	
	return blockID
}

// PredictAndPrefetch is called by the Router/Orchestrator the millisecond
// a new prompt arrives. It hash-matches the prefix, determining all
// underlying KV blocks. If they are in RAM or NVMe, it queues them for
// immediate prefetch.
func (pm *PrefetchManager) PredictAndPrefetch(prefixTokens []int32) {
	// In a real system, we'd break the prefix into block-sized chunks
	// and hash them to find the BlockIDs.
	// For simulation, we'll hash the whole thing to check if it's a known block.
	hasher := sha256.New()
	for _, t := range prefixTokens {
		hasher.Write([]byte{byte(t >> 24), byte(t >> 16), byte(t >> 8), byte(t)})
	}
	blockID := hex.EncodeToString(hasher.Sum(nil))
	
	pm.mu.RLock()
	meta, exists := pm.blockMap[blockID]
	pm.mu.RUnlock()
	
	if exists && meta.Location != LocGPU {
		// Non-blocking send to prefetch queue
		select {
		case pm.prefetchChan <- blockID:
		default:
			// Queue full, will have to fault later
		}
	}
}

// EvictLRU forces the coldest GPU block into RAM to make space.
func (pm *PrefetchManager) EvictLRU(requiredSize int64) {
	pm.mu.Lock()
	defer pm.mu.Unlock()
	
	for pm.usedGPUMemory+requiredSize > pm.maxGPUMemory {
		var lruID string
		var oldest time.Time
		
		for id, meta := range pm.blockMap {
			if meta.Location == LocGPU {
				if lruID == "" || meta.LastUsed.Before(oldest) {
					lruID = id
					oldest = meta.LastUsed
				}
			}
		}
		
		if lruID != "" {
			meta := pm.blockMap[lruID]
			// Execute offload (GPU -> RAM)
			meta.Location = LocRAM
			// In CGo, cudaMemcpyAsync(DeviceToHost)
			pm.usedGPUMemory -= meta.Size
		} else {
			break // Nothing left to evict
		}
	}
}

// prefetchWorker acts as the DMA engine, moving RAM/NVMe blocks back to GPU.
func (pm *PrefetchManager) prefetchWorker() {
	defer pm.wg.Done()
	
	for blockID := range pm.prefetchChan {
		pm.mu.Lock()
		meta, exists := pm.blockMap[blockID]
		if !exists || meta.Location == LocGPU {
			pm.mu.Unlock()
			continue
		}
		
		sizeReq := meta.Size
		pm.mu.Unlock()
		
		// Ensure space
		pm.EvictLRU(sizeReq)
		
		pm.mu.Lock()
		// Execute load (RAM -> GPU)
		meta.Location = LocGPU
		meta.LastUsed = time.Now()
		pm.usedGPUMemory += sizeReq
		pm.mu.Unlock()
	}
}

// EnsureBlock is the hard synchronization point. The attention layer calls
// this right before computing. If prefetch succeeded, it returns instantly.
// If not, it blocks and forces a synchronous transfer (a page fault).
func (pm *PrefetchManager) EnsureBlock(blockID string) int {
	pm.mu.RLock()
	meta, exists := pm.blockMap[blockID]
	pm.mu.RUnlock()
	
	if !exists { return -1 }
	
	if meta.Location != LocGPU {
		// Cache miss / prefetch too slow. Force sync load.
		pm.EvictLRU(meta.Size)
		
		pm.mu.Lock()
		meta.Location = LocGPU
		meta.LastUsed = time.Now()
		pm.usedGPUMemory += meta.Size
		gpuIdx := len(pm.blockMap) // Simulated allocation
		meta.GPUIndex = gpuIdx
		pm.mu.Unlock()
		return gpuIdx
	}
	
	pm.mu.Lock()
	meta.LastUsed = time.Now()
	meta.AccessCount++
	pm.mu.Unlock()
	
	return meta.GPUIndex
}
