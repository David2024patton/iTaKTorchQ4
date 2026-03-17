package torch

import (
	"strings"
	"testing"
	"time"
)

func TestSpeculativeMetricsString(t *testing.T) {
	// Test 1: Metrics WITH speculative decoding should include [spec] section.
	m := &InferenceMetrics{
		PromptTokens:       50,
		CompletionTokens:   100,
		TotalTokens:        150,
		PromptDuration:     200 * time.Millisecond,
		GenDuration:        800 * time.Millisecond,
		TotalDuration:      1000 * time.Millisecond,
		TokensPerSecond:    125.0,
		SpecDraftTokens:    100,
		SpecAcceptedTokens: 85,
		SpecAcceptRate:     85.0,
		SpecRounds:         10,
	}

	s := m.String()

	if !strings.Contains(s, "[perf]") {
		t.Errorf("expected [perf] section, got: %s", s)
	}
	if !strings.Contains(s, "125.0 tok/s") {
		t.Errorf("expected 125.0 tok/s, got: %s", s)
	}
	if !strings.Contains(s, "spec:") {
		t.Errorf("expected spec section when SpecDraftTokens > 0, got: %s", s)
	}
	if !strings.Contains(s, "85/100") {
		t.Errorf("expected 85/100 accepted, got: %s", s)
	}
	if !strings.Contains(s, "85%%") && !strings.Contains(s, "85%") {
		t.Errorf("expected 85%% acceptance rate, got: %s", s)
	}
	if !strings.Contains(s, "10 rounds") {
		t.Errorf("expected 10 rounds, got: %s", s)
	}

	t.Logf("Spec metrics output: %s", s)
}

func TestSpeculativeMetricsStringNoSpec(t *testing.T) {
	// Test 2: Metrics WITHOUT speculative decoding should NOT include spec section.
	m := &InferenceMetrics{
		PromptTokens:     50,
		CompletionTokens: 100,
		TotalTokens:      150,
		PromptDuration:   200 * time.Millisecond,
		GenDuration:      800 * time.Millisecond,
		TotalDuration:    1000 * time.Millisecond,
		TokensPerSecond:  125.0,
		// SpecDraftTokens is 0 (default) - spec section should be hidden.
	}

	s := m.String()

	if strings.Contains(s, "spec:") {
		t.Errorf("expected NO spec section when SpecDraftTokens == 0, got: %s", s)
	}
	if !strings.Contains(s, "[perf]") {
		t.Errorf("expected [perf] section, got: %s", s)
	}

	t.Logf("No-spec metrics output: %s", s)
}

func TestSystemResourcesString(t *testing.T) {
	r := CaptureResources()
	s := r.String()

	if !strings.Contains(s, "[sys]") {
		t.Errorf("expected [sys] prefix, got: %s", s)
	}
	if !strings.Contains(s, "heap:") {
		t.Errorf("expected heap field, got: %s", s)
	}
	if r.GoRoutines <= 0 {
		t.Errorf("expected goroutines > 0, got: %d", r.GoRoutines)
	}

	t.Logf("System resources: %s", s)
}

func TestEngineStatsRecordRequest(t *testing.T) {
	stats := &EngineStats{}

	m1 := &InferenceMetrics{
		CompletionTokens: 100,
		TokensPerSecond:  500.0,
	}
	stats.RecordRequest(m1)

	if stats.RequestCount != 1 {
		t.Errorf("expected 1 request, got: %d", stats.RequestCount)
	}
	if stats.TotalTokensGen != 100 {
		t.Errorf("expected 100 tokens, got: %d", stats.TotalTokensGen)
	}
	if stats.AvgTokPerSec != 500.0 {
		t.Errorf("expected 500.0 avg tok/s on first request, got: %.1f", stats.AvgTokPerSec)
	}

	// Second request: EMA should weight toward recent.
	m2 := &InferenceMetrics{
		CompletionTokens: 50,
		TokensPerSecond:  600.0,
	}
	stats.RecordRequest(m2)

	if stats.RequestCount != 2 {
		t.Errorf("expected 2 requests, got: %d", stats.RequestCount)
	}
	if stats.TotalTokensGen != 150 {
		t.Errorf("expected 150 tokens, got: %d", stats.TotalTokensGen)
	}
	// EMA: 500*0.8 + 600*0.2 = 520.0
	expectedAvg := 520.0
	if stats.AvgTokPerSec != expectedAvg {
		t.Errorf("expected %.1f avg tok/s after EMA, got: %.1f", expectedAvg, stats.AvgTokPerSec)
	}

	// Snapshot should return a copy.
	snap := stats.Snapshot()
	if snap.RequestCount != 2 {
		t.Errorf("snapshot request count mismatch: got %d", snap.RequestCount)
	}
}
