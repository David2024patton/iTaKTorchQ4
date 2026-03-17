// dynamic_batch.go implements adaptive batch sizing based on system load.
//
// WHAT: Instead of fixed batch sizes, dynamically adjust the batch size
// based on queue depth, GPU memory pressure, and latency targets.
//
// HOW:
//   - Queue deep? Increase batch size for throughput.
//   - Latency spiking? Decrease batch size for responsiveness.
//   - GPU memory tight? Cap batch size to prevent OOM.
//   - Mix of short and long sequences? Pack efficiently.
//
// WHY: Fixed batch sizes waste resources. A batch size of 32 is great for
// high load but wasteful for 3 pending requests. Dynamic batching adapts
// in real-time to maximize both throughput and latency.
package native

import (
	"sync"
	"sync/atomic"
	"time"
)

// DynamicBatchConfig configures the adaptive batcher.
type DynamicBatchConfig struct {
	MinBatchSize     int           // Floor (default: 1)
	MaxBatchSize     int           // Ceiling (default: 64)
	TargetLatencyMs  float64       // Latency target in ms (default: 100)
	MemoryBudgetMB   int64         // GPU memory budget for batching
	WaitTimeout      time.Duration // Max wait for batch to fill (default: 5ms)
	ScaleUpFactor    float32       // How aggressively to increase batch (default: 1.5)
	ScaleDownFactor  float32       // How aggressively to decrease batch (default: 0.75)
}

// DefaultDynamicBatchConfig returns standard settings.
func DefaultDynamicBatchConfig() DynamicBatchConfig {
	return DynamicBatchConfig{
		MinBatchSize:    1,
		MaxBatchSize:    64,
		TargetLatencyMs: 100,
		MemoryBudgetMB:  4096,
		WaitTimeout:     5 * time.Millisecond,
		ScaleUpFactor:   1.5,
		ScaleDownFactor: 0.75,
	}
}

// DynamicBatcher manages adaptive batch sizing.
type DynamicBatcher struct {
	mu     sync.Mutex
	config DynamicBatchConfig

	// Current state.
	currentBatchSize int32
	queueDepth       int32
	recentLatencies  []float64
	latencyIdx       int

	// Memory tracking.
	usedMemoryMB int64

	// EMA of metrics.
	emaLatency   float64
	emaThroughput float64

	// Stats.
	totalBatches  int64
	totalRequests int64
	adjustments   int64
}

// NewDynamicBatcher creates an adaptive batcher.
func NewDynamicBatcher(config DynamicBatchConfig) *DynamicBatcher {
	return &DynamicBatcher{
		config:           config,
		currentBatchSize: int32(config.MinBatchSize),
		recentLatencies:  make([]float64, 100),
	}
}

// RecommendBatchSize returns the optimal batch size for the current conditions.
func (db *DynamicBatcher) RecommendBatchSize() int {
	db.mu.Lock()
	defer db.mu.Unlock()

	current := int(atomic.LoadInt32(&db.currentBatchSize))
	queue := int(atomic.LoadInt32(&db.queueDepth))
	newSize := current

	// Rule 1: If queue is deep, scale up for throughput.
	if queue > current*2 {
		newSize = int(float32(current) * db.config.ScaleUpFactor)
	}

	// Rule 2: If latency exceeds target, scale down for responsiveness.
	if db.emaLatency > db.config.TargetLatencyMs {
		newSize = int(float32(current) * db.config.ScaleDownFactor)
	}

	// Rule 3: If latency is well under target and queue exists, scale up.
	if db.emaLatency < db.config.TargetLatencyMs*0.5 && queue > 0 {
		newSize = int(float32(current) * db.config.ScaleUpFactor)
	}

	// Rule 4: Memory budget constraint.
	memPerRequest := db.usedMemoryMB / int64(max(current, 1))
	if memPerRequest > 0 {
		maxByMem := int(db.config.MemoryBudgetMB / memPerRequest)
		if newSize > maxByMem {
			newSize = maxByMem
		}
	}

	// Clamp to bounds.
	if newSize < db.config.MinBatchSize {
		newSize = db.config.MinBatchSize
	}
	if newSize > db.config.MaxBatchSize {
		newSize = db.config.MaxBatchSize
	}

	// Don't exceed queue depth (no point batching more than available).
	if queue > 0 && newSize > queue {
		newSize = queue
	}

	if newSize != current {
		db.adjustments++
	}

	atomic.StoreInt32(&db.currentBatchSize, int32(newSize))
	return newSize
}

// RecordLatency updates latency EMA after a batch completes.
func (db *DynamicBatcher) RecordLatency(latencyMs float64) {
	db.mu.Lock()
	defer db.mu.Unlock()

	db.recentLatencies[db.latencyIdx%len(db.recentLatencies)] = latencyMs
	db.latencyIdx++

	// EMA: alpha = 0.1.
	db.emaLatency = db.emaLatency*0.9 + latencyMs*0.1
}

// RecordThroughput updates throughput EMA.
func (db *DynamicBatcher) RecordThroughput(tokPerSec float64) {
	db.mu.Lock()
	defer db.mu.Unlock()
	db.emaThroughput = db.emaThroughput*0.9 + tokPerSec*0.1
}

// SetQueueDepth updates the current queue depth.
func (db *DynamicBatcher) SetQueueDepth(depth int) {
	atomic.StoreInt32(&db.queueDepth, int32(depth))
}

// SetMemoryUsage updates the current GPU memory usage.
func (db *DynamicBatcher) SetMemoryUsage(usedMB int64) {
	db.mu.Lock()
	defer db.mu.Unlock()
	db.usedMemoryMB = usedMB
}

// CurrentSize returns the current batch size.
func (db *DynamicBatcher) CurrentSize() int {
	return int(atomic.LoadInt32(&db.currentBatchSize))
}

// WaitForBatch waits up to WaitTimeout for the batch to fill, or returns early
// if enough requests arrive.
func (db *DynamicBatcher) WaitForBatch() {
	target := int(atomic.LoadInt32(&db.currentBatchSize))
	deadline := time.Now().Add(db.config.WaitTimeout)

	for time.Now().Before(deadline) {
		if int(atomic.LoadInt32(&db.queueDepth)) >= target {
			return // Batch is full.
		}
		time.Sleep(500 * time.Microsecond)
	}
}

// Stats returns dynamic batching metrics.
func (db *DynamicBatcher) Stats() map[string]interface{} {
	db.mu.Lock()
	defer db.mu.Unlock()
	return map[string]interface{}{
		"current_batch_size": atomic.LoadInt32(&db.currentBatchSize),
		"queue_depth":        atomic.LoadInt32(&db.queueDepth),
		"ema_latency_ms":     db.emaLatency,
		"ema_throughput":     db.emaThroughput,
		"total_adjustments":  db.adjustments,
		"memory_used_mb":     db.usedMemoryMB,
	}
}

// max returns the larger of two ints.
func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
