// data_prefetch.go implements background data loading and tokenization.
//
// WHAT: During training, the GPU sits idle while the CPU loads and tokenizes
// the next batch. Data prefetching uses a background goroutine to prepare
// the next N batches while the current batch is training.
//
// PIPELINE:
//   Main goroutine:     Train(batch_N) -> Train(batch_N+1) -> Train(batch_N+2)
//   Prefetch goroutine:  Load(batch_N+2) -> Tokenize -> Buffer
//
// The prefetch goroutine stays 1-2 batches ahead, so by the time the main
// goroutine needs the next batch, it's already tokenized and ready in memory.
//
// GAIN: Eliminates data loading stalls. On I/O-bound training (reading from
// disk, large tokenized sequences), this can improve training throughput
// by 20-50%.
package native

import (
	"context"
	"fmt"
	"sync"
	"time"
)

// PrefetchBatch holds a pre-loaded and tokenized training batch.
type PrefetchBatch struct {
	InputIDs  [][]int32   // [batchSize][seqLen]
	Labels    [][]int32   // [batchSize][seqLen]
	Attention [][]float32 // [batchSize][seqLen] attention mask
	Index     int         // Batch index in the epoch
}

// DataPrefetcher manages background batch loading with a ring buffer.
type DataPrefetcher struct {
	mu sync.Mutex

	// Ring buffer of pre-loaded batches.
	buffer     []*PrefetchBatch
	bufferSize int
	readIdx    int
	writeIdx   int
	count      int

	// Data source.
	loadFn func(batchIdx int) (*PrefetchBatch, error)

	// Control.
	ctx    context.Context
	cancel context.CancelFunc
	wg     sync.WaitGroup

	// State.
	nextBatchIdx int
	running      bool

	// Stats.
	totalLoaded   int64
	totalWaits    int64
	loadTimeTotal time.Duration
}

// PrefetchConfig configures the data prefetcher.
type PrefetchConfig struct {
	BufferSize int // Number of batches to keep ahead (default: 3)
	NumWorkers int // Number of parallel loading goroutines (default: 1)
}

// DefaultPrefetchConfig returns recommended settings.
func DefaultPrefetchConfig() PrefetchConfig {
	return PrefetchConfig{
		BufferSize: 3,
		NumWorkers: 1,
	}
}

// NewDataPrefetcher creates a prefetcher that calls loadFn to produce batches.
// loadFn should be a function that loads and tokenizes one batch by index.
func NewDataPrefetcher(config PrefetchConfig, loadFn func(batchIdx int) (*PrefetchBatch, error)) *DataPrefetcher {
	if config.BufferSize < 1 {
		config.BufferSize = 3
	}

	ctx, cancel := context.WithCancel(context.Background())

	return &DataPrefetcher{
		buffer:     make([]*PrefetchBatch, config.BufferSize),
		bufferSize: config.BufferSize,
		loadFn:     loadFn,
		ctx:        ctx,
		cancel:     cancel,
	}
}

// Start begins background batch loading.
func (dp *DataPrefetcher) Start() {
	dp.mu.Lock()
	if dp.running {
		dp.mu.Unlock()
		return
	}
	dp.running = true
	dp.mu.Unlock()

	dp.wg.Add(1)
	go dp.prefetchLoop()
	fmt.Printf("[DataPrefetch] Started with buffer size %d\n", dp.bufferSize)
}

// prefetchLoop runs in a background goroutine, loading batches ahead.
func (dp *DataPrefetcher) prefetchLoop() {
	defer dp.wg.Done()

	for {
		select {
		case <-dp.ctx.Done():
			return
		default:
		}

		// Check if buffer has room.
		dp.mu.Lock()
		if dp.count >= dp.bufferSize {
			dp.mu.Unlock()
			// Buffer full, wait a bit.
			time.Sleep(1 * time.Millisecond)
			continue
		}
		batchIdx := dp.nextBatchIdx
		dp.nextBatchIdx++
		dp.mu.Unlock()

		// Load batch (potentially slow: disk I/O, tokenization).
		start := time.Now()
		batch, err := dp.loadFn(batchIdx)
		loadTime := time.Since(start)

		if err != nil {
			// End of data or error.
			dp.mu.Lock()
			dp.running = false
			dp.mu.Unlock()
			return
		}

		batch.Index = batchIdx

		// Store in ring buffer.
		dp.mu.Lock()
		dp.buffer[dp.writeIdx] = batch
		dp.writeIdx = (dp.writeIdx + 1) % dp.bufferSize
		dp.count++
		dp.totalLoaded++
		dp.loadTimeTotal += loadTime
		dp.mu.Unlock()
	}
}

// Next returns the next pre-loaded batch, blocking if not yet available.
// Returns nil when no more data (end of epoch or stopped).
func (dp *DataPrefetcher) Next() *PrefetchBatch {
	for {
		dp.mu.Lock()
		if dp.count > 0 {
			batch := dp.buffer[dp.readIdx]
			dp.buffer[dp.readIdx] = nil // GC
			dp.readIdx = (dp.readIdx + 1) % dp.bufferSize
			dp.count--
			dp.mu.Unlock()
			return batch
		}

		// Check if prefetcher has stopped (end of data).
		if !dp.running {
			dp.mu.Unlock()
			return nil
		}

		dp.totalWaits++
		dp.mu.Unlock()

		// Buffer empty but prefetcher still running. Wait for it.
		time.Sleep(100 * time.Microsecond)
	}
}

// Reset restarts from batch index 0 (new epoch).
func (dp *DataPrefetcher) Reset() {
	dp.Stop()

	dp.mu.Lock()
	dp.nextBatchIdx = 0
	dp.readIdx = 0
	dp.writeIdx = 0
	dp.count = 0
	dp.mu.Unlock()

	// Restart with fresh context.
	dp.ctx, dp.cancel = context.WithCancel(context.Background())
	dp.Start()
}

// Stop halts background prefetching.
func (dp *DataPrefetcher) Stop() {
	dp.cancel()
	dp.wg.Wait()

	dp.mu.Lock()
	dp.running = false
	dp.mu.Unlock()
}

// BufferedCount returns how many batches are currently pre-loaded.
func (dp *DataPrefetcher) BufferedCount() int {
	dp.mu.Lock()
	defer dp.mu.Unlock()
	return dp.count
}

// Stats returns prefetcher metrics.
func (dp *DataPrefetcher) Stats() map[string]interface{} {
	dp.mu.Lock()
	defer dp.mu.Unlock()

	avgLoadMs := float64(0)
	if dp.totalLoaded > 0 {
		avgLoadMs = float64(dp.loadTimeTotal.Milliseconds()) / float64(dp.totalLoaded)
	}

	return map[string]interface{}{
		"buffer_size":   dp.bufferSize,
		"buffered":      dp.count,
		"total_loaded":  dp.totalLoaded,
		"total_waits":   dp.totalWaits,
		"avg_load_ms":   fmt.Sprintf("%.1f", avgLoadMs),
		"running":       dp.running,
	}
}
