// scheduler.go implements a request queue with priority ordering and response channels
// for concurrent multi-request serving. Since llama.cpp uses a single inference context,
// requests are processed sequentially but HTTP connections can wait concurrently.
package torch

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"
	"time"
)

// RequestPriority controls processing order.
// Higher values = processed first. Critical requests (health checks, internal)
// always jump ahead of normal user requests.
type RequestPriority int

const (
	PriorityNormal   RequestPriority = 0 // regular user requests
	PriorityHigh     RequestPriority = 1 // elevated priority (paid users, system tasks)
	PriorityCritical RequestPriority = 2 // health checks, internal operations (always first)
)

type InferenceRequest struct {
	ID          uint64
	Messages    []ChatMessage
	InputTokens []int32
	Params      CompletionParams
	Priority    RequestPriority
	Created  time.Time
	Ctx      context.Context

	// Response channel - scheduler writes result here.
	ResultCh chan InferenceResult

	// StreamCh delivers token deltas for SSE streaming.
	// When non-nil, each generated token piece is sent to this channel.
	// The channel is closed when generation completes.
	StreamCh chan string
}

// InferenceResult is the outcome of a queued request.
type InferenceResult struct {
	Text    string
	Metrics *InferenceMetrics
	Err     error
}

// SchedulerStats exposes queue and throughput metrics.
type SchedulerStats struct {
	QueueDepth      int    `json:"queue_depth"`       // total pending requests
	HighQueueDepth  int    `json:"high_queue_depth"`  // pending high/critical requests
	TotalProcessed  uint64 `json:"total_processed"`
	TotalDropped    uint64 `json:"total_dropped"`
	AvgWaitMs       float64 `json:"avg_wait_ms"`
	AvgProcessingMs float64 `json:"avg_processing_ms"`
}

// Scheduler manages a queue of inference requests and processes them.
// When maxSlots > 1, uses continuous batching via BatchEngine.
// When maxSlots == 1, processes requests sequentially (original behavior).
//
// Architecture:
// The scheduler uses TWO channels: highQueue for critical/high priority requests,
// and normalQueue for regular requests. The processing loop ALWAYS drains the
// high queue before picking from the normal queue. This gives system-critical
// requests (health checks, internal operations) near-instant processing.
type Scheduler struct {
	engine      Engine
	batchEngine *BatchEngine // nil for sequential mode.
	highQueue   chan *InferenceRequest // critical + high priority
	normalQueue chan *InferenceRequest // normal priority
	maxQueue    int

	// Metrics.
	nextID         atomic.Uint64
	totalProcessed atomic.Uint64
	totalDropped   atomic.Uint64
	totalWaitNs    atomic.Int64
	totalProcNs    atomic.Int64

	// Lifecycle.
	stopCh chan struct{}
	wg     sync.WaitGroup
}

// NewScheduler creates a scheduler with the given queue capacity.
func NewScheduler(engine Engine, maxQueue int) *Scheduler {
	if maxQueue <= 0 {
		maxQueue = 64
	}
	return &Scheduler{
		engine:      engine,
		highQueue:   make(chan *InferenceRequest, maxQueue/4+1), // smaller high queue
		normalQueue: make(chan *InferenceRequest, maxQueue),
		maxQueue:    maxQueue,
		stopCh:      make(chan struct{}),
	}
}

// NewBatchScheduler creates a scheduler with continuous batching support.
func NewBatchScheduler(engine *TorchEngine, maxQueue, maxSlots int) *Scheduler {
	if maxQueue <= 0 {
		maxQueue = 64
	}
	s := &Scheduler{
		engine:      engine,
		highQueue:   make(chan *InferenceRequest, maxQueue/4+1),
		normalQueue: make(chan *InferenceRequest, maxQueue),
		maxQueue:    maxQueue,
		stopCh:      make(chan struct{}),
	}
	if maxSlots > 1 {
		s.batchEngine = NewBatchEngine(engine, maxSlots)
	}
	return s
}

// Start launches the scheduler processing loop.
func (s *Scheduler) Start() {
	s.wg.Add(1)
	if s.batchEngine != nil {
		go s.processBatchLoop()
		fmt.Printf("[iTaK Torch] Scheduler started (continuous batching, queue: %d)\n", s.maxQueue)
	} else {
		go s.processLoop()
		fmt.Printf("[iTaK Torch] Scheduler started (queue capacity: %d, priority lanes: high+normal)\n", s.maxQueue)
	}
}

// Stop gracefully shuts down the scheduler, finishing the current request.
func (s *Scheduler) Stop() {
	close(s.stopCh)
	s.wg.Wait()
	if s.batchEngine != nil {
		s.batchEngine.Close()
	}
	fmt.Printf("[iTaK Torch] Scheduler stopped (processed: %d, dropped: %d)\n",
		s.totalProcessed.Load(), s.totalDropped.Load())
}

// Submit queues a request for processing. Returns immediately.
// Critical/High priority requests go to the high queue (always drained first).
// Normal requests go to the normal queue.
// If the appropriate queue is full, the request is dropped with a 503-style error.
func (s *Scheduler) Submit(req *InferenceRequest) {
	req.ID = s.nextID.Add(1)
	req.Created = time.Now()
	req.ResultCh = make(chan InferenceResult, 1)

	// Route to appropriate queue based on priority.
	if req.Priority >= PriorityHigh {
		select {
		case s.highQueue <- req:
			// Queued successfully in high-priority lane.
		default:
			// High queue full - reject.
			s.totalDropped.Add(1)
			req.ResultCh <- InferenceResult{
				Err: fmt.Errorf("server overloaded: high-priority queue full (%d requests)", cap(s.highQueue)),
			}
		}
	} else {
		select {
		case s.normalQueue <- req:
			// Queued successfully in normal lane.
		default:
			// Normal queue full - reject.
			s.totalDropped.Add(1)
			req.ResultCh <- InferenceResult{
				Err: fmt.Errorf("server overloaded: %d requests queued", s.maxQueue),
			}
		}
	}
}

// Stats returns current scheduler metrics.
func (s *Scheduler) Stats() SchedulerStats {
	processed := s.totalProcessed.Load()
	avgWait := float64(0)
	avgProc := float64(0)
	if processed > 0 {
		avgWait = float64(s.totalWaitNs.Load()) / float64(processed) / 1e6 // ns -> ms
		avgProc = float64(s.totalProcNs.Load()) / float64(processed) / 1e6 // ns -> ms
	}
	return SchedulerStats{
		QueueDepth:      len(s.highQueue) + len(s.normalQueue),
		HighQueueDepth:  len(s.highQueue),
		TotalProcessed:  processed,
		TotalDropped:    s.totalDropped.Load(),
		AvgWaitMs:       avgWait,
		AvgProcessingMs: avgProc,
	}
}

// QueueDepth returns the current number of pending requests (both queues).
func (s *Scheduler) QueueDepth() int {
	return len(s.highQueue) + len(s.normalQueue)
}

// processLoop is the main scheduler goroutine. It processes one request at a time.
// PRIORITY ORDERING: always drain highQueue before normalQueue.
// This ensures critical requests (health checks, system ops) get processed ASAP.
func (s *Scheduler) processLoop() {
	defer s.wg.Done()

	for {
		// Phase 1: Check for shutdown.
		select {
		case <-s.stopCh:
			s.drainOnShutdown()
			return
		default:
		}

		// Phase 2: Always try high-priority first (non-blocking).
		select {
		case req := <-s.highQueue:
			s.processRequest(req)
			continue
		default:
		}

		// Phase 3: Wait for either queue or shutdown.
		// High queue gets checked first via the select ordering.
		select {
		case <-s.stopCh:
			s.drainOnShutdown()
			return
		case req := <-s.highQueue:
			s.processRequest(req)
		case req := <-s.normalQueue:
			s.processRequest(req)
		}
	}
}

// drainOnShutdown sends cancellation errors to all pending requests.
func (s *Scheduler) drainOnShutdown() {
	for {
		select {
		case req := <-s.highQueue:
			req.ResultCh <- InferenceResult{Err: fmt.Errorf("server shutting down")}
		case req := <-s.normalQueue:
			req.ResultCh <- InferenceResult{Err: fmt.Errorf("server shutting down")}
		default:
			return
		}
	}
}

// processRequest handles a single inference request.
func (s *Scheduler) processRequest(req *InferenceRequest) {
	waitDuration := time.Since(req.Created)
	s.totalWaitNs.Add(waitDuration.Nanoseconds())

	// Check if the client already disconnected while waiting in queue.
	select {
	case <-req.Ctx.Done():
		req.ResultCh <- InferenceResult{Err: req.Ctx.Err()}
		s.totalDropped.Add(1)
		return
	default:
	}

	// Run inference.
	procStart := time.Now()
	var result string
	var err error

	// Use streaming path if StreamCh is set and engine supports it.
	if req.StreamCh != nil {
		if te, ok := s.engine.(*TorchEngine); ok {
			result, err = te.CompleteStream(req.Ctx, req.Messages, req.Params, req.StreamCh)
		} else {
			// Fallback: non-streaming engine, close stream channel.
			if len(req.InputTokens) > 0 {
				result, err = s.engine.GenerateTokens(req.Ctx, req.InputTokens, req.Params)
			} else {
				result, err = s.engine.Complete(req.Ctx, req.Messages, req.Params)
			}
			close(req.StreamCh)
		}
	} else {
		if len(req.InputTokens) > 0 {
			result, err = s.engine.GenerateTokens(req.Ctx, req.InputTokens, req.Params)
		} else {
			result, err = s.engine.Complete(req.Ctx, req.Messages, req.Params)
		}
	}

	procDuration := time.Since(procStart)
	s.totalProcNs.Add(procDuration.Nanoseconds())

	// Get metrics from engine.
	stats := s.engine.GetStats()

	req.ResultCh <- InferenceResult{
		Text:    result,
		Metrics: stats.LastMetrics,
		Err:     err,
	}

	s.totalProcessed.Add(1)
}

// processBatchLoop delegates to the BatchEngine for continuous batching.
// Note: In batch mode, all requests go through normalQueue.
// Priority routing is handled by the sequential processLoop only.
func (s *Scheduler) processBatchLoop() {
	defer s.wg.Done()
	s.batchEngine.Run(s.normalQueue, s.stopCh)
}
