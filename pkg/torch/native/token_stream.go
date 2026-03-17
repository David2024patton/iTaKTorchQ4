// token_stream.go implements a streaming protocol for real-time token delivery.
//
// WHAT: Instead of waiting for the full response, stream tokens to clients
// as they're generated. Each token is delivered immediately via a channel-based
// protocol that maps to SSE (Server-Sent Events) or WebSocket at the HTTP layer.
//
// FEATURES:
//   - Per-request token channels with backpressure
//   - Metadata per token (logprob, finish reason)
//   - Multi-consumer: one generation can stream to multiple listeners
//   - Usage tracking (prompt + completion tokens)
//   - Timeout and cancellation support
package native

import (
	"context"
	"sync"
	"sync/atomic"
	"time"
)

// StreamTokenType identifies the type of stream event.
type StreamTokenType int

const (
	StreamTokenGenerated  StreamTokenType = iota // Normal generated token
	StreamTokenFinish                             // Generation complete
	StreamTokenError                              // Error occurred
	StreamTokenHeartbeat                          // Keep-alive (no token)
)

// StreamToken is one event in the token stream.
type StreamToken struct {
	Type       StreamTokenType
	TokenID    int32
	TokenText  string
	LogProb    float64
	FinishReason string // "stop", "length", "error"
	Error      error
	Index      int    // Position in generated sequence
	Timestamp  time.Time
}

// TokenStream manages streaming for one request.
type TokenStream struct {
	mu         sync.RWMutex
	requestID  string
	ch         chan StreamToken
	consumers  []chan StreamToken
	bufferSize int

	// State.
	started   bool
	finished  bool
	cancelled int32

	// Usage tracking.
	promptTokens     int
	completionTokens int

	// Context for cancellation.
	ctx    context.Context
	cancel context.CancelFunc
}

// NewTokenStream creates a streaming channel for one request.
func NewTokenStream(requestID string, bufferSize int) *TokenStream {
	ctx, cancel := context.WithCancel(context.Background())

	ts := &TokenStream{
		requestID:  requestID,
		ch:         make(chan StreamToken, bufferSize),
		bufferSize: bufferSize,
		ctx:        ctx,
		cancel:     cancel,
	}

	return ts
}

// Channel returns the primary token channel for reading.
func (ts *TokenStream) Channel() <-chan StreamToken {
	return ts.ch
}

// Subscribe creates a new consumer channel that receives all tokens.
// Multiple consumers can subscribe (e.g., SSE + logging + metrics).
func (ts *TokenStream) Subscribe() <-chan StreamToken {
	ts.mu.Lock()
	defer ts.mu.Unlock()

	consumer := make(chan StreamToken, ts.bufferSize)
	ts.consumers = append(ts.consumers, consumer)
	return consumer
}

// Emit sends a token to all consumers.
func (ts *TokenStream) Emit(token StreamToken) bool {
	if atomic.LoadInt32(&ts.cancelled) != 0 {
		return false
	}

	token.Timestamp = time.Now()

	ts.mu.RLock()
	defer ts.mu.RUnlock()

	// Send to primary channel (with backpressure).
	select {
	case ts.ch <- token:
	case <-ts.ctx.Done():
		return false
	}

	// Fan out to all consumer channels.
	for _, consumer := range ts.consumers {
		select {
		case consumer <- token:
		default:
			// Consumer is slow: drop token (backpressure).
		}
	}

	if token.Type == StreamTokenGenerated {
		ts.completionTokens++
	}

	return true
}

// EmitToken is a convenience method for emitting a generated token.
func (ts *TokenStream) EmitToken(tokenID int32, text string, logProb float64, index int) bool {
	return ts.Emit(StreamToken{
		Type:      StreamTokenGenerated,
		TokenID:   tokenID,
		TokenText: text,
		LogProb:   logProb,
		Index:     index,
	})
}

// Finish signals that generation is complete.
func (ts *TokenStream) Finish(reason string) {
	ts.mu.Lock()
	defer ts.mu.Unlock()

	ts.finished = true
	finalToken := StreamToken{
		Type:         StreamTokenFinish,
		FinishReason: reason,
		Timestamp:    time.Now(),
	}

	// Send to all channels.
	select {
	case ts.ch <- finalToken:
	default:
	}
	for _, consumer := range ts.consumers {
		select {
		case consumer <- finalToken:
		default:
		}
	}

	// Close channels.
	close(ts.ch)
	for _, consumer := range ts.consumers {
		close(consumer)
	}
}

// Cancel aborts the stream.
func (ts *TokenStream) Cancel() {
	atomic.StoreInt32(&ts.cancelled, 1)
	ts.cancel()
	ts.Finish("cancelled")
}

// IsCancelled checks if the stream was cancelled.
func (ts *TokenStream) IsCancelled() bool {
	return atomic.LoadInt32(&ts.cancelled) != 0
}

// SetPromptTokens records the number of prompt tokens for usage tracking.
func (ts *TokenStream) SetPromptTokens(n int) {
	ts.promptTokens = n
}

// Usage returns token usage statistics (OpenAI-compatible format).
func (ts *TokenStream) Usage() map[string]int {
	return map[string]int{
		"prompt_tokens":     ts.promptTokens,
		"completion_tokens": ts.completionTokens,
		"total_tokens":      ts.promptTokens + ts.completionTokens,
	}
}

// StreamManager manages multiple concurrent token streams.
type StreamManager struct {
	mu      sync.RWMutex
	streams map[string]*TokenStream
	nextID  int64
}

// NewStreamManager creates a stream manager.
func NewStreamManager() *StreamManager {
	return &StreamManager{
		streams: make(map[string]*TokenStream),
	}
}

// Create starts a new token stream.
func (sm *StreamManager) Create(requestID string) *TokenStream {
	sm.mu.Lock()
	defer sm.mu.Unlock()

	stream := NewTokenStream(requestID, 128)
	sm.streams[requestID] = stream
	return stream
}

// Get retrieves an active stream by request ID.
func (sm *StreamManager) Get(requestID string) (*TokenStream, bool) {
	sm.mu.RLock()
	defer sm.mu.RUnlock()
	s, ok := sm.streams[requestID]
	return s, ok
}

// Remove removes a completed stream.
func (sm *StreamManager) Remove(requestID string) {
	sm.mu.Lock()
	defer sm.mu.Unlock()
	delete(sm.streams, requestID)
}

// ActiveCount returns the number of active streams.
func (sm *StreamManager) ActiveCount() int {
	sm.mu.RLock()
	defer sm.mu.RUnlock()
	return len(sm.streams)
}
