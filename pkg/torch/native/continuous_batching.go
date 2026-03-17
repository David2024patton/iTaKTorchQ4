// continuous_batching.go implements continuous batching for serving multiple
// concurrent inference requests.
//
// WHAT: Instead of processing one request at a time, continuous batching
// dynamically adds and removes sequences from the batch as they start and
// finish. This keeps GPU utilization high even when requests have different
// lengths.
//
// HOW: A RequestQueue accepts incoming inference requests. The BatchScheduler
// picks up to maxBatchSize active sequences each step, runs one forward pass
// for all of them, and emits generated tokens. Finished sequences are removed
// and pending ones are added.
package native

import (
	"context"
	"fmt"
	"sync"
	"time"
)

// InferenceRequest represents one pending or active generation request.
type InferenceRequest struct {
	ID        string
	Prompt    []int     // Input token IDs
	MaxTokens int       // Maximum tokens to generate
	Generated []int     // Tokens generated so far
	Done      bool      // True when complete
	Result    chan string // Channel to send final result
	StartTime time.Time
	Ctx       context.Context
}

// ContinuousBatcher manages concurrent inference requests.
type ContinuousBatcher struct {
	engine       *NativeEngine
	maxBatchSize int

	// Request management.
	mu       sync.Mutex
	pending  []*InferenceRequest // Waiting to be processed
	active   []*InferenceRequest // Currently being processed
	nextID   int

	// Metrics.
	totalRequests    int
	totalTokensGen   int64
	batchUtilization []float64
}

// NewContinuousBatcher creates a batcher for the given engine.
func NewContinuousBatcher(engine *NativeEngine, maxBatchSize int) *ContinuousBatcher {
	return &ContinuousBatcher{
		engine:       engine,
		maxBatchSize: maxBatchSize,
	}
}

// Submit adds a new request to the queue. Returns a channel that will receive the result.
func (cb *ContinuousBatcher) Submit(prompt []int, maxTokens int, ctx context.Context) (string, chan string) {
	cb.mu.Lock()
	defer cb.mu.Unlock()

	cb.nextID++
	id := fmt.Sprintf("req-%d", cb.nextID)

	req := &InferenceRequest{
		ID:        id,
		Prompt:    prompt,
		MaxTokens: maxTokens,
		Result:    make(chan string, 1),
		StartTime: time.Now(),
		Ctx:       ctx,
	}

	cb.pending = append(cb.pending, req)
	cb.totalRequests++

	return id, req.Result
}

// Run starts the continuous batching loop. Blocks until ctx is cancelled.
func (cb *ContinuousBatcher) Run(ctx context.Context) {
	fmt.Printf("[ContinuousBatch] Started with max batch size %d\n", cb.maxBatchSize)

	for {
		select {
		case <-ctx.Done():
			fmt.Println("[ContinuousBatch] Shutting down")
			return
		default:
		}

		// Fill batch from pending queue.
		cb.fillBatch()

		if len(cb.active) == 0 {
			// No work; sleep briefly to avoid a busy wait.
			time.Sleep(10 * time.Millisecond)
			continue
		}

		// Process one step for all active requests.
		cb.processBatchStep()

		// Remove completed requests.
		cb.cleanupCompleted()

		// Track utilization.
		util := float64(len(cb.active)) / float64(cb.maxBatchSize)
		cb.batchUtilization = append(cb.batchUtilization, util)
	}
}

// fillBatch moves pending requests into the active batch.
func (cb *ContinuousBatcher) fillBatch() {
	cb.mu.Lock()
	defer cb.mu.Unlock()

	for len(cb.active) < cb.maxBatchSize && len(cb.pending) > 0 {
		req := cb.pending[0]
		cb.pending = cb.pending[1:]

		// Check if request was cancelled.
		if req.Ctx.Err() != nil {
			close(req.Result)
			continue
		}

		cb.active = append(cb.active, req)
	}
}

// processBatchStep runs one generation step for all active requests.
func (cb *ContinuousBatcher) processBatchStep() {
	for _, req := range cb.active {
		if req.Done {
			continue
		}

		// Check context cancellation.
		if req.Ctx.Err() != nil {
			req.Done = true
			continue
		}

		// Build current context: prompt + generated so far.
		current := make([]int, 0, len(req.Prompt)+len(req.Generated))
		current = append(current, req.Prompt...)
		current = append(current, req.Generated...)

		// Forward pass.
		logits := cb.engine.forward(current)

		// Greedy decode.
		nextToken := 0
		maxLogit := logits[0]
		for i := 1; i < len(logits); i++ {
			if logits[i] > maxLogit {
				maxLogit = logits[i]
				nextToken = i
			}
		}

		req.Generated = append(req.Generated, nextToken)
		cb.totalTokensGen++

		// Check completion.
		if len(req.Generated) >= req.MaxTokens || nextToken == 2 { // 2 = EOS
			req.Done = true
		}
	}
}

// cleanupCompleted removes finished requests and sends results.
func (cb *ContinuousBatcher) cleanupCompleted() {
	cb.mu.Lock()
	defer cb.mu.Unlock()

	remaining := make([]*InferenceRequest, 0, len(cb.active))
	for _, req := range cb.active {
		if req.Done {
			// Send result.
			result := tokensToText(req.Generated)
			req.Result <- result
			close(req.Result)
		} else {
			remaining = append(remaining, req)
		}
	}
	cb.active = remaining
}

// Stats returns current batching metrics.
func (cb *ContinuousBatcher) Stats() BatchStats {
	cb.mu.Lock()
	defer cb.mu.Unlock()

	avgUtil := 0.0
	if len(cb.batchUtilization) > 0 {
		sum := 0.0
		for _, u := range cb.batchUtilization {
			sum += u
		}
		avgUtil = sum / float64(len(cb.batchUtilization))
	}

	return BatchStats{
		TotalRequests:    cb.totalRequests,
		PendingRequests:  len(cb.pending),
		ActiveRequests:   len(cb.active),
		TotalTokensGen:   cb.totalTokensGen,
		AvgUtilization:   avgUtil,
	}
}

// BatchStats holds continuous batching performance metrics.
type BatchStats struct {
	TotalRequests   int
	PendingRequests int
	ActiveRequests  int
	TotalTokensGen  int64
	AvgUtilization  float64
}

// PrintStats logs batcher metrics.
func (s BatchStats) PrintStats() {
	fmt.Printf("[ContinuousBatch] Requests: %d total, %d pending, %d active\n",
		s.TotalRequests, s.PendingRequests, s.ActiveRequests)
	fmt.Printf("[ContinuousBatch] Tokens: %d, Avg utilization: %.1f%%\n",
		s.TotalTokensGen, s.AvgUtilization*100)
}
