// request_trace.go implements per-request distributed tracing.
//
// WHAT: Each inference request gets a unique trace ID that follows it through
// every stage: tokenization, KV cache lookup, forward passes, sampling,
// detokenization. This enables pinpointing bottlenecks and debugging
// slow requests.
//
// COMPATIBLE: Trace data can be exported as OpenTelemetry-compatible spans
// or logged as structured JSON for Jaeger/Zipkin/Grafana Tempo.
package torch

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"sync"
	"time"
)

// TraceID is a unique identifier for a request trace.
type TraceID string

// SpanID is a unique identifier for a span within a trace.
type SpanID string

// GenerateTraceID creates a random 16-byte trace ID.
func GenerateTraceID() TraceID {
	b := make([]byte, 16)
	rand.Read(b)
	return TraceID(hex.EncodeToString(b))
}

// GenerateSpanID creates a random 8-byte span ID.
func GenerateSpanID() SpanID {
	b := make([]byte, 8)
	rand.Read(b)
	return SpanID(hex.EncodeToString(b))
}

// Span represents one operation within a trace.
type Span struct {
	TraceID   TraceID                `json:"trace_id"`
	SpanID    SpanID                 `json:"span_id"`
	ParentID  SpanID                 `json:"parent_id,omitempty"`
	Name      string                 `json:"name"`
	StartTime time.Time              `json:"start_time"`
	EndTime   time.Time              `json:"end_time,omitempty"`
	Duration  time.Duration          `json:"duration_ms,omitempty"`
	Attrs     map[string]interface{} `json:"attrs,omitempty"`
	Status    string                 `json:"status"` // "ok", "error"
	children  []*Span
}

// End marks the span as complete.
func (s *Span) End() {
	s.EndTime = time.Now()
	s.Duration = s.EndTime.Sub(s.StartTime)
	if s.Status == "" {
		s.Status = "ok"
	}
}

// SetError marks the span as failed.
func (s *Span) SetError(err error) {
	s.Status = "error"
	if s.Attrs == nil {
		s.Attrs = make(map[string]interface{})
	}
	s.Attrs["error"] = err.Error()
}

// SetAttr adds a key-value attribute to the span.
func (s *Span) SetAttr(key string, value interface{}) {
	if s.Attrs == nil {
		s.Attrs = make(map[string]interface{})
	}
	s.Attrs[key] = value
}

// RequestTrace tracks all operations for a single request.
type RequestTrace struct {
	mu    sync.Mutex
	root  *Span
	spans []*Span
}

// NewRequestTrace creates a new trace for a request.
func NewRequestTrace(operationName string) *RequestTrace {
	traceID := GenerateTraceID()
	root := &Span{
		TraceID:   traceID,
		SpanID:    GenerateSpanID(),
		Name:      operationName,
		StartTime: time.Now(),
		Attrs:     make(map[string]interface{}),
	}

	return &RequestTrace{
		root:  root,
		spans: []*Span{root},
	}
}

// TraceID returns this trace's ID.
func (t *RequestTrace) TraceID() TraceID {
	return t.root.TraceID
}

// StartSpan creates a child span under the root.
func (t *RequestTrace) StartSpan(name string) *Span {
	t.mu.Lock()
	defer t.mu.Unlock()

	span := &Span{
		TraceID:   t.root.TraceID,
		SpanID:    GenerateSpanID(),
		ParentID:  t.root.SpanID,
		Name:      name,
		StartTime: time.Now(),
		Attrs:     make(map[string]interface{}),
	}
	t.spans = append(t.spans, span)
	t.root.children = append(t.root.children, span)
	return span
}

// Finish ends the root span and returns the complete trace.
func (t *RequestTrace) Finish() {
	t.root.End()
}

// TotalDuration returns the overall trace duration.
func (t *RequestTrace) TotalDuration() time.Duration {
	if t.root.EndTime.IsZero() {
		return time.Since(t.root.StartTime)
	}
	return t.root.Duration
}

// ToJSON exports the trace as JSON for logging/exporting.
func (t *RequestTrace) ToJSON() ([]byte, error) {
	t.mu.Lock()
	defer t.mu.Unlock()
	return json.Marshal(t.spans)
}

// Summary returns a concise summary of the trace.
func (t *RequestTrace) Summary() string {
	t.mu.Lock()
	defer t.mu.Unlock()

	var summary string
	summary = fmt.Sprintf("Trace %s (%s, %s)\n",
		t.root.TraceID, t.root.Name, t.root.Duration)

	for _, span := range t.spans[1:] { // Skip root.
		status := span.Status
		if status == "" {
			status = "running"
		}
		summary += fmt.Sprintf("  [%s] %s: %s (%v)\n",
			status, span.SpanID, span.Name, span.Duration)
	}

	return summary
}

// ---------- Trace Collector ----------

// TraceCollector aggregates traces for analysis.
type TraceCollector struct {
	mu       sync.Mutex
	traces   []*RequestTrace
	maxStore int
}

// NewTraceCollector creates a collector that stores recent traces.
func NewTraceCollector(maxStore int) *TraceCollector {
	return &TraceCollector{
		traces:   make([]*RequestTrace, 0, maxStore),
		maxStore: maxStore,
	}
}

// Add stores a completed trace.
func (tc *TraceCollector) Add(trace *RequestTrace) {
	tc.mu.Lock()
	defer tc.mu.Unlock()

	if len(tc.traces) >= tc.maxStore {
		// Remove oldest.
		tc.traces = tc.traces[1:]
	}
	tc.traces = append(tc.traces, trace)
}

// Recent returns the N most recent traces.
func (tc *TraceCollector) Recent(n int) []*RequestTrace {
	tc.mu.Lock()
	defer tc.mu.Unlock()

	if n > len(tc.traces) {
		n = len(tc.traces)
	}
	start := len(tc.traces) - n
	result := make([]*RequestTrace, n)
	copy(result, tc.traces[start:])
	return result
}

// SlowTraces returns traces exceeding the given duration.
func (tc *TraceCollector) SlowTraces(threshold time.Duration) []*RequestTrace {
	tc.mu.Lock()
	defer tc.mu.Unlock()

	var slow []*RequestTrace
	for _, t := range tc.traces {
		if t.TotalDuration() > threshold {
			slow = append(slow, t)
		}
	}
	return slow
}
