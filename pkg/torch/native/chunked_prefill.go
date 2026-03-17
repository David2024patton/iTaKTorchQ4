// chunked_prefill.go splits long prompt processing into chunks to interleave
// with decode steps for other requests.
//
// WHAT: When a 4K-token prompt arrives, standard prefill processes all 4K
// tokens before any decode steps can run. This blocks all other requests
// for the entire prefill duration (potentially hundreds of ms).
//
// Chunked prefill breaks the prompt into smaller chunks (e.g., 256 tokens)
// and interleaves decode steps for other requests between chunks:
//   Chunk 1 (tokens 0-255)   -> Decode 3 tokens for request B
//   Chunk 2 (tokens 256-511) -> Decode 2 tokens for request C
//   Chunk 3 (tokens 512-767) -> Decode 1 token for request B
//   ... and so on
//
// GAIN: Dramatically better tail latency for concurrent requests.
// A request that would have waited 500ms now waits at most chunk_time.
package native

import (
	"context"
	"sync"
)

// DefaultChunkSize is the number of tokens processed per prefill chunk.
// 256 is a good balance: large enough for GPU efficiency, small enough
// to not block decode steps for too long.
const DefaultChunkSize = 256

// PrefillChunk represents one segment of a prompt being prefilled.
type PrefillChunk struct {
	SequenceID string
	TokenIDs   []int32
	ChunkIndex int
	TotalChunks int
	StartPos   int // Position in original prompt
}

// ChunkedPrefiller manages chunked prompt processing with decode interleaving.
type ChunkedPrefiller struct {
	mu sync.Mutex

	chunkSize    int
	pendingQueue []*PrefillJob
	activeJob    *PrefillJob

	// Stats.
	totalChunks    int64
	totalPrompts   int64
	interleaveOps  int64
}

// PrefillJob tracks the state of one prompt being chunked.
type PrefillJob struct {
	SequenceID  string
	Tokens      []int32
	ChunksTotal int
	ChunksDone  int
	KVState     interface{} // Opaque KV cache state between chunks
	Ctx         context.Context
	DoneCh      chan struct{}
}

// NewChunkedPrefiller creates a prefiller with the given chunk size.
func NewChunkedPrefiller(chunkSize int) *ChunkedPrefiller {
	if chunkSize <= 0 {
		chunkSize = DefaultChunkSize
	}
	return &ChunkedPrefiller{
		chunkSize:    chunkSize,
		pendingQueue: make([]*PrefillJob, 0),
	}
}

// SubmitPrompt queues a prompt for chunked prefill processing.
// Returns a channel that closes when prefill is complete.
func (cp *ChunkedPrefiller) SubmitPrompt(ctx context.Context, seqID string, tokens []int32) <-chan struct{} {
	cp.mu.Lock()
	defer cp.mu.Unlock()

	totalChunks := (len(tokens) + cp.chunkSize - 1) / cp.chunkSize
	job := &PrefillJob{
		SequenceID:  seqID,
		Tokens:      tokens,
		ChunksTotal: totalChunks,
		Ctx:         ctx,
		DoneCh:      make(chan struct{}),
	}

	cp.pendingQueue = append(cp.pendingQueue, job)
	cp.totalPrompts++
	return job.DoneCh
}

// NextChunk returns the next prefill chunk to process, or nil if no work.
// The scheduler calls this between decode steps to interleave prefill work.
func (cp *ChunkedPrefiller) NextChunk() *PrefillChunk {
	cp.mu.Lock()
	defer cp.mu.Unlock()

	// Resume active job or pick a new one.
	if cp.activeJob == nil {
		if len(cp.pendingQueue) == 0 {
			return nil
		}
		cp.activeJob = cp.pendingQueue[0]
		cp.pendingQueue = cp.pendingQueue[1:]
	}

	job := cp.activeJob

	// Check cancellation.
	if job.Ctx.Err() != nil {
		close(job.DoneCh)
		cp.activeJob = nil
		return cp.NextChunk() // Try next job.
	}

	// Extract the next chunk.
	startPos := job.ChunksDone * cp.chunkSize
	endPos := startPos + cp.chunkSize
	if endPos > len(job.Tokens) {
		endPos = len(job.Tokens)
	}

	chunk := &PrefillChunk{
		SequenceID:  job.SequenceID,
		TokenIDs:    job.Tokens[startPos:endPos],
		ChunkIndex:  job.ChunksDone,
		TotalChunks: job.ChunksTotal,
		StartPos:    startPos,
	}

	job.ChunksDone++
	cp.totalChunks++

	// Check if this was the last chunk.
	if job.ChunksDone >= job.ChunksTotal {
		close(job.DoneCh)
		cp.activeJob = nil
	}

	return chunk
}

// HasPendingWork returns true if there are prefill chunks waiting.
func (cp *ChunkedPrefiller) HasPendingWork() bool {
	cp.mu.Lock()
	defer cp.mu.Unlock()
	return cp.activeJob != nil || len(cp.pendingQueue) > 0
}

// PendingCount returns the number of prompts waiting for prefill.
func (cp *ChunkedPrefiller) PendingCount() int {
	cp.mu.Lock()
	defer cp.mu.Unlock()
	count := len(cp.pendingQueue)
	if cp.activeJob != nil {
		count++
	}
	return count
}

// InterleaveStep is called by the scheduler to record that a decode step
// was interleaved between prefill chunks. Used for metrics.
func (cp *ChunkedPrefiller) InterleaveStep() {
	cp.mu.Lock()
	cp.interleaveOps++
	cp.mu.Unlock()
}

// Stats returns chunked prefill metrics.
func (cp *ChunkedPrefiller) Stats() map[string]interface{} {
	cp.mu.Lock()
	defer cp.mu.Unlock()
	return map[string]interface{}{
		"chunk_size":      cp.chunkSize,
		"total_prompts":   cp.totalPrompts,
		"total_chunks":    cp.totalChunks,
		"interleave_ops":  cp.interleaveOps,
		"pending_prompts": len(cp.pendingQueue),
		"active_job":      cp.activeJob != nil,
	}
}

// ScheduleInterleaved runs the main interleaving loop:
// process one prefill chunk, then allow N decode steps, repeat.
// decodeFunc processes pending decode requests and returns how many were served.
// prefillFunc processes one prefill chunk.
func (cp *ChunkedPrefiller) ScheduleInterleaved(
	ctx context.Context,
	prefillFunc func(chunk *PrefillChunk) error,
	decodeFunc func() (int, error),
	maxDecodesPerChunk int,
) error {
	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		// Process one prefill chunk if available.
		chunk := cp.NextChunk()
		if chunk != nil {
			if err := prefillFunc(chunk); err != nil {
				return err
			}
		}

		// Interleave decode steps.
		for i := 0; i < maxDecodesPerChunk; i++ {
			decoded, err := decodeFunc()
			if err != nil {
				return err
			}
			if decoded > 0 {
				cp.InterleaveStep()
			}
			if decoded == 0 {
				break // No more decode work.
			}
		}

		// If nothing to do, yield.
		if chunk == nil {
			decoded, _ := decodeFunc()
			if decoded == 0 {
				return nil // No work at all.
			}
		}
	}
}
