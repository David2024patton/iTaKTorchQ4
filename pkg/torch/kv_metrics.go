// kv_metrics.go tracks KV cache utilization for the continuous batching engine.
package torch

import (
	"fmt"
	"sync"

	"github.com/David2024patton/iTaKTorchQ4/pkg/torch/llama"
)

// KVCacheMetrics tracks KV cache slot utilization.
type KVCacheMetrics struct {
	mu          sync.RWMutex
	TotalSlots  int     `json:"total_slots"`
	ActiveSlots int     `json:"active_slots"`
	IdleSlots   int     `json:"idle_slots"`
	Utilization float64 `json:"utilization_pct"` // 0-100
	ContextSize uint32  `json:"context_size"`
	Overflows   int64   `json:"kv_overflows"` // Total context overflow events
}

// NewKVCacheMetrics creates a tracker for the given context and slot count.
func NewKVCacheMetrics(ctx llama.Context, totalSlots int) *KVCacheMetrics {
	return &KVCacheMetrics{
		TotalSlots:  totalSlots,
		IdleSlots:   totalSlots,
		ContextSize: llama.NCtx(ctx),
	}
}

// Update refreshes the metrics from the slot manager state.
func (m *KVCacheMetrics) Update(active, idle int) {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.ActiveSlots = active
	m.IdleSlots = idle

	if m.TotalSlots > 0 {
		m.Utilization = float64(active) / float64(m.TotalSlots) * 100
	}
}

// RecordOverflow increments the overflow counter.
func (m *KVCacheMetrics) RecordOverflow() {
	m.mu.Lock()
	m.Overflows++
	m.mu.Unlock()
}

// Snapshot returns a copy of the current metrics.
func (m *KVCacheMetrics) Snapshot() KVCacheMetrics {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return KVCacheMetrics{
		TotalSlots:  m.TotalSlots,
		ActiveSlots: m.ActiveSlots,
		IdleSlots:   m.IdleSlots,
		Utilization: m.Utilization,
		ContextSize: m.ContextSize,
		Overflows:   m.Overflows,
	}
}

// String returns a compact summary.
func (m *KVCacheMetrics) String() string {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return fmt.Sprintf(
		"[kv] slots: %d/%d (%.0f%%) | ctx: %d | overflows: %d",
		m.ActiveSlots, m.TotalSlots, m.Utilization, m.ContextSize, m.Overflows,
	)
}
