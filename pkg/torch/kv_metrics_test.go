package torch

import (
	"strings"
	"testing"
)

func TestKVCacheMetricsUpdate(t *testing.T) {
	m := &KVCacheMetrics{
		TotalSlots:  4,
		IdleSlots:   4,
		ContextSize: 8192,
	}

	// Update: 3 active, 1 idle -> 75% utilization.
	m.Update(3, 1)

	if m.ActiveSlots != 3 {
		t.Errorf("expected 3 active, got: %d", m.ActiveSlots)
	}
	if m.IdleSlots != 1 {
		t.Errorf("expected 1 idle, got: %d", m.IdleSlots)
	}
	if m.Utilization != 75.0 {
		t.Errorf("expected 75%% utilization, got: %.1f%%", m.Utilization)
	}
}

func TestKVCacheMetricsOverflow(t *testing.T) {
	m := &KVCacheMetrics{
		TotalSlots:  4,
		ContextSize: 8192,
	}

	m.RecordOverflow()
	m.RecordOverflow()

	if m.Overflows != 2 {
		t.Errorf("expected 2 overflows, got: %d", m.Overflows)
	}
}

func TestKVCacheMetricsSnapshot(t *testing.T) {
	m := &KVCacheMetrics{
		TotalSlots:  4,
		ContextSize: 8192,
	}
	m.Update(2, 2)
	m.RecordOverflow()

	snap := m.Snapshot()

	// Verify snapshot has correct values.
	if snap.ActiveSlots != 2 {
		t.Errorf("snapshot active mismatch: %d", snap.ActiveSlots)
	}
	if snap.Overflows != 1 {
		t.Errorf("snapshot overflow mismatch: %d", snap.Overflows)
	}

	// Verify snapshot is a copy (modifying it doesn't affect original).
	snap.ActiveSlots = 99
	if m.ActiveSlots == 99 {
		t.Error("snapshot should be a copy, not a reference")
	}
}

func TestKVCacheMetricsString(t *testing.T) {
	m := &KVCacheMetrics{
		TotalSlots:  4,
		ContextSize: 8192,
	}
	m.Update(3, 1)
	m.RecordOverflow()
	m.RecordOverflow()

	s := m.String()

	if !strings.Contains(s, "[kv]") {
		t.Errorf("expected [kv] prefix, got: %s", s)
	}
	if !strings.Contains(s, "3/4") {
		t.Errorf("expected 3/4 slots, got: %s", s)
	}
	if !strings.Contains(s, "75%") {
		t.Errorf("expected 75%% utilization, got: %s", s)
	}
	if !strings.Contains(s, "8192") {
		t.Errorf("expected context size 8192, got: %s", s)
	}
	if !strings.Contains(s, "overflows: 2") {
		t.Errorf("expected overflows: 2, got: %s", s)
	}

	t.Logf("KV metrics: %s", s)
}

func TestKVCacheMetricsZeroSlots(t *testing.T) {
	// Edge case: zero total slots should not panic.
	m := &KVCacheMetrics{
		TotalSlots:  0,
		ContextSize: 4096,
	}
	m.Update(0, 0)

	if m.Utilization != 0 {
		t.Errorf("expected 0%% utilization with 0 total slots, got: %.1f%%", m.Utilization)
	}

	s := m.String()
	if !strings.Contains(s, "0/0") {
		t.Errorf("expected 0/0 slots, got: %s", s)
	}
}
