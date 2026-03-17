// priority_queue.go implements a priority-based request queue for inference.
//
// WHAT: In a multi-tenant server, some requests are more important than others.
// This queue uses priority levels to determine processing order:
//   Priority 0 (Critical): System prompts, health checks
//   Priority 1 (High):     Paid/premium API keys
//   Priority 2 (Normal):   Standard requests
//   Priority 3 (Low):      Batch/background jobs
//   Priority 4 (Bulk):     Data processing, non-interactive
//
// FAIRNESS: Within the same priority level, requests are served FIFO.
// A starvation timer promotes long-waiting low-priority requests.
package torch

import (
	"container/heap"
	"context"
	"sync"
	"time"
)

// RequestPriority type and PriorityNormal/PriorityHigh/PriorityCritical are
// defined in scheduler.go. We add two more levels here.
const (
	PriorityLow  RequestPriority = -1 // Batch/background jobs
	PriorityBulk RequestPriority = -2 // Data processing, non-interactive
)

// PriorityString returns a human-readable label for the priority.
func PriorityString(p RequestPriority) string {
	switch p {
	case PriorityCritical:
		return "critical"
	case PriorityHigh:
		return "high"
	case PriorityNormal:
		return "normal"
	case PriorityLow:
		return "low"
	case PriorityBulk:
		return "bulk"
	default:
		return "unknown"
	}
}

// QueuedRequest wraps an inference request with priority metadata.
type QueuedRequest struct {
	ID         string
	Priority   RequestPriority
	EnqueuedAt time.Time
	Ctx        context.Context
	Messages   []ChatMessage
	Params     CompletionParams
	ResultCh   chan QueueResult
	index      int // heap internal
}

// QueueResult is the response sent back to the caller.
type QueueResult struct {
	Text  string
	Error error
}

// PriorityQueue implements heap.Interface for priority-based scheduling.
type PriorityQueue []*QueuedRequest

func (pq PriorityQueue) Len() int { return len(pq) }

func (pq PriorityQueue) Less(i, j int) bool {
	// Lower priority number = higher priority.
	if pq[i].Priority != pq[j].Priority {
		return pq[i].Priority < pq[j].Priority
	}
	// Same priority: FIFO by enqueue time.
	return pq[i].EnqueuedAt.Before(pq[j].EnqueuedAt)
}

func (pq PriorityQueue) Swap(i, j int) {
	pq[i], pq[j] = pq[j], pq[i]
	pq[i].index = i
	pq[j].index = j
}

func (pq *PriorityQueue) Push(x interface{}) {
	n := len(*pq)
	item := x.(*QueuedRequest)
	item.index = n
	*pq = append(*pq, item)
}

func (pq *PriorityQueue) Pop() interface{} {
	old := *pq
	n := len(old)
	item := old[n-1]
	old[n-1] = nil // GC
	item.index = -1
	*pq = old[:n-1]
	return item
}

// RequestScheduler manages the priority queue and dispatches to the engine.
type RequestScheduler struct {
	mu             sync.Mutex
	queue          PriorityQueue
	maxQueueSize   int
	starvationTime time.Duration // Promote after this wait time
	notify         chan struct{} // Signal that a request was enqueued
}

// NewRequestScheduler creates a scheduler.
func NewRequestScheduler(maxQueueSize int) *RequestScheduler {
	rs := &RequestScheduler{
		queue:          make(PriorityQueue, 0),
		maxQueueSize:   maxQueueSize,
		starvationTime: 30 * time.Second,
		notify:         make(chan struct{}, 1),
	}
	heap.Init(&rs.queue)
	return rs
}

// Enqueue adds a request to the priority queue.
// Returns error if the queue is full.
func (rs *RequestScheduler) Enqueue(req *QueuedRequest) error {
	rs.mu.Lock()
	defer rs.mu.Unlock()

	if rs.queue.Len() >= rs.maxQueueSize {
		return &QueueFullError{Size: rs.queue.Len(), Max: rs.maxQueueSize}
	}

	req.EnqueuedAt = time.Now()
	req.ResultCh = make(chan QueueResult, 1)
	heap.Push(&rs.queue, req)

	// Non-blocking signal.
	select {
	case rs.notify <- struct{}{}:
	default:
	}

	return nil
}

// Dequeue removes and returns the highest-priority request.
// Returns nil if the queue is empty.
func (rs *RequestScheduler) Dequeue() *QueuedRequest {
	rs.mu.Lock()
	defer rs.mu.Unlock()

	// Anti-starvation: promote old low-priority requests.
	rs.promoteStarved()

	if rs.queue.Len() == 0 {
		return nil
	}

	return heap.Pop(&rs.queue).(*QueuedRequest)
}

// promoteStarved upgrades priority of requests waiting too long.
func (rs *RequestScheduler) promoteStarved() {
	now := time.Now()
	for _, req := range rs.queue {
		if req.Priority > PriorityHigh && now.Sub(req.EnqueuedAt) > rs.starvationTime {
			req.Priority-- // Promote by one level.
		}
	}
	heap.Init(&rs.queue) // Re-sort after promotions.
}

// QueueSize returns the current queue depth.
func (rs *RequestScheduler) QueueSize() int {
	rs.mu.Lock()
	defer rs.mu.Unlock()
	return rs.queue.Len()
}

// WaitCh returns a channel that signals when a request is enqueued.
func (rs *RequestScheduler) WaitCh() <-chan struct{} {
	return rs.notify
}

// Stats returns queue metrics.
func (rs *RequestScheduler) Stats() map[string]interface{} {
	rs.mu.Lock()
	defer rs.mu.Unlock()

	counts := map[string]int{
		"critical": 0, "high": 0, "normal": 0, "low": 0, "bulk": 0,
	}
	for _, req := range rs.queue {
		counts[PriorityString(req.Priority)]++
	}

	return map[string]interface{}{
		"total":    rs.queue.Len(),
		"max_size": rs.maxQueueSize,
		"by_priority": counts,
	}
}

// QueueFullError is returned when the queue is at capacity.
type QueueFullError struct {
	Size int
	Max  int
}

func (e *QueueFullError) Error() string {
	return "request queue full"
}
