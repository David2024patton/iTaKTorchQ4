// metrics_export.go provides a /metrics endpoint in Prometheus exposition format.
//
// WHY: Production LLM deployments need monitoring. Prometheus scrapes this
// endpoint to track token throughput, latency, VRAM usage, and cache hit rates.
// Grafana dashboards can then visualize Torch's performance in real time.
package torch

import (
	"fmt"
	"net/http"
	"strings"
	"sync/atomic"
	"time"
)

// PrometheusMetrics tracks all counters, gauges, and histograms for /metrics.
type PrometheusMetrics struct {
	// Counters (monotonically increasing).
	TotalRequests      atomic.Int64
	TotalTokens        atomic.Int64
	TotalPromptTokens  atomic.Int64
	TotalCompletionTok atomic.Int64
	CacheHits          atomic.Int64
	CacheMisses        atomic.Int64
	TotalErrors        atomic.Int64
	StreamingRequests  atomic.Int64

	// For averages.
	TotalLatencyMs     atomic.Int64
	TotalTokPerSec     atomic.Int64
	LatencySampleCount atomic.Int64

	// Gauges (set to current value).
	ActiveRequests atomic.Int32
	ActiveSlots    atomic.Int32
	LoadedModels   atomic.Int32
	StartTime      time.Time
}

// NewPrometheusMetrics initializes a new metrics collector.
func NewPrometheusMetrics() *PrometheusMetrics {
	return &PrometheusMetrics{
		StartTime: time.Now(),
	}
}

// RecordRequest tracks a completed inference request.
func (pm *PrometheusMetrics) RecordRequest(promptTokens, completionTokens int, latencyMs int64, tokPerSec float64, isStreaming bool, isError bool) {
	pm.TotalRequests.Add(1)
	pm.TotalPromptTokens.Add(int64(promptTokens))
	pm.TotalCompletionTok.Add(int64(completionTokens))
	pm.TotalTokens.Add(int64(promptTokens + completionTokens))
	pm.TotalLatencyMs.Add(latencyMs)
	pm.TotalTokPerSec.Add(int64(tokPerSec * 100)) // Store as fixed-point *100
	pm.LatencySampleCount.Add(1)
	if isStreaming {
		pm.StreamingRequests.Add(1)
	}
	if isError {
		pm.TotalErrors.Add(1)
	}
}

// handleMetrics serves the /metrics endpoint in Prometheus exposition format.
func (s *Server) handleMetrics(w http.ResponseWriter, r *http.Request) {
	if s.metrics == nil {
		w.Header().Set("Content-Type", "text/plain")
		w.Write([]byte("# no metrics available\n"))
		return
	}

	pm := s.metrics
	var sb strings.Builder

	// Uptime.
	uptime := time.Since(pm.StartTime).Seconds()
	writeMetric(&sb, "torch_uptime_seconds", "gauge", "Time since server start", uptime)

	// Request counters.
	writeMetric(&sb, "torch_requests_total", "counter", "Total inference requests", float64(pm.TotalRequests.Load()))
	writeMetric(&sb, "torch_errors_total", "counter", "Total failed requests", float64(pm.TotalErrors.Load()))
	writeMetric(&sb, "torch_streaming_requests_total", "counter", "Total streaming requests", float64(pm.StreamingRequests.Load()))

	// Token counters.
	writeMetric(&sb, "torch_tokens_total", "counter", "Total tokens processed (prompt + completion)", float64(pm.TotalTokens.Load()))
	writeMetric(&sb, "torch_prompt_tokens_total", "counter", "Total prompt tokens", float64(pm.TotalPromptTokens.Load()))
	writeMetric(&sb, "torch_completion_tokens_total", "counter", "Total completion tokens", float64(pm.TotalCompletionTok.Load()))

	// Cache.
	writeMetric(&sb, "torch_cache_hits_total", "counter", "Prefix/response cache hits", float64(pm.CacheHits.Load()))
	writeMetric(&sb, "torch_cache_misses_total", "counter", "Prefix/response cache misses", float64(pm.CacheMisses.Load()))

	// Gauges.
	writeMetric(&sb, "torch_active_requests", "gauge", "Currently processing requests", float64(pm.ActiveRequests.Load()))
	writeMetric(&sb, "torch_active_slots", "gauge", "Active batching slots", float64(pm.ActiveSlots.Load()))
	writeMetric(&sb, "torch_loaded_models", "gauge", "Number of loaded models", float64(pm.LoadedModels.Load()))

	// Averages.
	samples := pm.LatencySampleCount.Load()
	if samples > 0 {
		avgLatency := float64(pm.TotalLatencyMs.Load()) / float64(samples)
		avgTokPerSec := float64(pm.TotalTokPerSec.Load()) / float64(samples) / 100.0
		writeMetric(&sb, "torch_avg_latency_ms", "gauge", "Average request latency in ms", avgLatency)
		writeMetric(&sb, "torch_avg_tokens_per_second", "gauge", "Average tokens per second", avgTokPerSec)
	}

	// Engine-specific stats.
	stats := s.engine.GetStats()
	writeMetric(&sb, "torch_model_load_time_seconds", "gauge", "Model load time", stats.ModelLoadTime.Seconds())

	w.Header().Set("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
	w.Write([]byte(sb.String()))
}

// writeMetric writes a single metric in Prometheus exposition format.
func writeMetric(sb *strings.Builder, name, metricType, help string, value float64) {
	fmt.Fprintf(sb, "# HELP %s %s\n", name, help)
	fmt.Fprintf(sb, "# TYPE %s %s\n", name, metricType)
	fmt.Fprintf(sb, "%s %g\n\n", name, value)
}
