package torch

import (
	"fmt"
	"runtime"
	"sync"
	"time"
)

// InferenceMetrics tracks performance data for a single inference request.
type InferenceMetrics struct {
	PromptTokens     int           `json:"prompt_tokens"`
	CompletionTokens int           `json:"completion_tokens"`
	TotalTokens      int           `json:"total_tokens"`
	PromptDuration   time.Duration `json:"prompt_duration_ms"`
	GenDuration      time.Duration `json:"gen_duration_ms"`
	TotalDuration    time.Duration `json:"total_duration_ms"`
	TokensPerSecond  float64       `json:"tokens_per_second"`

	// Speculative decoding metrics (zero when not using draft model).
	SpecDraftTokens    int     `json:"spec_draft_tokens,omitempty"`
	SpecAcceptedTokens int     `json:"spec_accepted_tokens,omitempty"`
	SpecAcceptRate     float64 `json:"spec_accept_rate,omitempty"`
	SpecRounds         int     `json:"spec_rounds,omitempty"`
}

// String returns a compact one-line summary for CLI display.
func (m *InferenceMetrics) String() string {
	s := fmt.Sprintf(
		"[perf] %d tok in %s (%.1f tok/s) | prompt: %s | gen: %s",
		m.CompletionTokens,
		m.TotalDuration.Round(time.Millisecond),
		m.TokensPerSecond,
		m.PromptDuration.Round(time.Millisecond),
		m.GenDuration.Round(time.Millisecond),
	)
	if m.SpecDraftTokens > 0 {
		s += fmt.Sprintf(" | spec: %d/%d accepted (%.0f%%), %d rounds",
			m.SpecAcceptedTokens, m.SpecDraftTokens, m.SpecAcceptRate, m.SpecRounds)
	}
	return s
}

// SystemResources captures a snapshot of system resource usage.
type SystemResources struct {
	Timestamp    time.Time `json:"timestamp"`
	GoRoutines   int       `json:"goroutines"`
	HeapAllocMB  float64   `json:"heap_alloc_mb"`
	HeapSysMB    float64   `json:"heap_sys_mb"`
	TotalAllocMB float64   `json:"total_alloc_mb"`
	SysMB        float64   `json:"sys_mb"`
	NumGC        uint32    `json:"num_gc"`
}

// CaptureResources takes a snapshot of current system resources.
func CaptureResources() SystemResources {
	var mem runtime.MemStats
	runtime.ReadMemStats(&mem)

	return SystemResources{
		Timestamp:    time.Now(),
		GoRoutines:   runtime.NumGoroutine(),
		HeapAllocMB:  float64(mem.HeapAlloc) / 1024 / 1024,
		HeapSysMB:    float64(mem.HeapSys) / 1024 / 1024,
		TotalAllocMB: float64(mem.TotalAlloc) / 1024 / 1024,
		SysMB:        float64(mem.Sys) / 1024 / 1024,
		NumGC:        mem.NumGC,
	}
}

// String returns a compact one-line summary.
func (r *SystemResources) String() string {
	return fmt.Sprintf(
		"[sys] heap: %.1fMB | sys: %.1fMB | goroutines: %d | gc: %d",
		r.HeapAllocMB, r.SysMB, r.GoRoutines, r.NumGC,
	)
}

// EngineStats aggregates performance data for the running engine.
type EngineStats struct {
	mu             sync.RWMutex
	ModelLoadTime  time.Duration     `json:"model_load_time_ms"`
	PreLoadRes     SystemResources   `json:"pre_load_resources"`
	PostLoadRes    SystemResources   `json:"post_load_resources"`
	RequestCount   int64             `json:"request_count"`
	TotalTokensGen int64             `json:"total_tokens_generated"`
	AvgTokPerSec   float64           `json:"avg_tokens_per_second"`
	LastMetrics    *InferenceMetrics `json:"last_request_metrics,omitempty"`
}

// RecordRequest updates aggregate stats after an inference request.
func (s *EngineStats) RecordRequest(m *InferenceMetrics) {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.RequestCount++
	s.TotalTokensGen += int64(m.CompletionTokens)
	s.LastMetrics = m

	// Rolling average tok/s.
	if s.RequestCount == 1 {
		s.AvgTokPerSec = m.TokensPerSecond
	} else {
		// Exponential moving average (weight recent requests more).
		s.AvgTokPerSec = s.AvgTokPerSec*0.8 + m.TokensPerSecond*0.2
	}
}

// Snapshot returns a copy of the current stats.
func (s *EngineStats) Snapshot() EngineStats {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return EngineStats{
		ModelLoadTime:  s.ModelLoadTime,
		PreLoadRes:     s.PreLoadRes,
		PostLoadRes:    s.PostLoadRes,
		RequestCount:   s.RequestCount,
		TotalTokensGen: s.TotalTokensGen,
		AvgTokPerSec:   s.AvgTokPerSec,
		LastMetrics:    s.LastMetrics,
	}
}
