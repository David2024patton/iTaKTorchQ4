// model_version.go implements model version tracking, rollback, and history.
//
// WHAT: In production, you need to track which model version is active,
// when it was deployed, and roll back if a new version performs worse.
// This file provides a version ledger with:
//   - Version history with timestamps and metadata
//   - Active version tracking
//   - Rollback to any previous version
//   - Version comparison (metrics diff)
package torch

import (
	"fmt"
	"sync"
	"time"
)

// ModelVersion describes one deployed model version.
type ModelVersion struct {
	ID          string                 `json:"id"`          // e.g., "v1.2.3" or hash
	ModelPath   string                 `json:"model_path"`
	DeployedAt  time.Time              `json:"deployed_at"`
	DeployedBy  string                 `json:"deployed_by"`  // API key or user
	Description string                 `json:"description"`
	Metadata    map[string]interface{} `json:"metadata"`     // quant level, LoRA, etc.
	Status      string                 `json:"status"`       // "active", "rolled_back", "retired"
	Metrics     *VersionMetrics        `json:"metrics,omitempty"`
}

// VersionMetrics tracks quality metrics for a version.
type VersionMetrics struct {
	RequestCount  int64   `json:"request_count"`
	AvgLatencyMs  float64 `json:"avg_latency_ms"`
	ErrorRate     float64 `json:"error_rate"`
	AvgTokensSec  float64 `json:"avg_tokens_sec"`
	UserSatisfaction float64 `json:"user_satisfaction,omitempty"` // 0-1 if tracked
}

// ModelVersionManager tracks all deployed versions.
type ModelVersionManager struct {
	mu       sync.RWMutex
	versions []*ModelVersion
	activeID string
}

// NewModelVersionManager creates a version manager.
func NewModelVersionManager() *ModelVersionManager {
	return &ModelVersionManager{
		versions: make([]*ModelVersion, 0),
	}
}

// Deploy registers a new model version as active.
func (m *ModelVersionManager) Deploy(version *ModelVersion) {
	m.mu.Lock()
	defer m.mu.Unlock()

	// Retire previous active version.
	for _, v := range m.versions {
		if v.Status == "active" {
			v.Status = "retired"
		}
	}

	version.DeployedAt = time.Now()
	version.Status = "active"
	if version.Metadata == nil {
		version.Metadata = make(map[string]interface{})
	}
	if version.Metrics == nil {
		version.Metrics = &VersionMetrics{}
	}

	m.versions = append(m.versions, version)
	m.activeID = version.ID

	fmt.Printf("[ModelVersion] Deployed %s: %s\n", version.ID, version.Description)
}

// ActiveVersion returns the currently active version.
func (m *ModelVersionManager) ActiveVersion() *ModelVersion {
	m.mu.RLock()
	defer m.mu.RUnlock()

	for _, v := range m.versions {
		if v.ID == m.activeID {
			return v
		}
	}
	return nil
}

// Rollback reverts to a previous version by ID.
func (m *ModelVersionManager) Rollback(versionID string) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	// Find the target version.
	var target *ModelVersion
	for _, v := range m.versions {
		if v.ID == versionID {
			target = v
			break
		}
	}
	if target == nil {
		return fmt.Errorf("version %s not found", versionID)
	}

	// Retire current active.
	for _, v := range m.versions {
		if v.Status == "active" {
			v.Status = "rolled_back"
		}
	}

	target.Status = "active"
	m.activeID = target.ID

	fmt.Printf("[ModelVersion] Rolled back to %s\n", versionID)
	return nil
}

// History returns all versions, newest first.
func (m *ModelVersionManager) History() []*ModelVersion {
	m.mu.RLock()
	defer m.mu.RUnlock()

	result := make([]*ModelVersion, len(m.versions))
	for i, v := range m.versions {
		result[len(m.versions)-1-i] = v // Reverse order
	}
	return result
}

// UpdateMetrics updates the performance metrics for a version.
func (m *ModelVersionManager) UpdateMetrics(versionID string, latencyMs float64, tokens int, isError bool) {
	m.mu.Lock()
	defer m.mu.Unlock()

	for _, v := range m.versions {
		if v.ID != versionID {
			continue
		}
		if v.Metrics == nil {
			v.Metrics = &VersionMetrics{}
		}

		v.Metrics.RequestCount++
		if isError {
			v.Metrics.ErrorRate = float64(v.Metrics.ErrorRate*float64(v.Metrics.RequestCount-1)+1) /
				float64(v.Metrics.RequestCount)
		}

		// Rolling average latency.
		n := float64(v.Metrics.RequestCount)
		v.Metrics.AvgLatencyMs = v.Metrics.AvgLatencyMs*(n-1)/n + latencyMs/n

		// Rolling average tokens/sec.
		if latencyMs > 0 {
			tokSec := float64(tokens) / (latencyMs / 1000.0)
			v.Metrics.AvgTokensSec = v.Metrics.AvgTokensSec*(n-1)/n + tokSec/n
		}

		break
	}
}

// Compare returns the metrics difference between two versions.
func (m *ModelVersionManager) Compare(v1ID, v2ID string) map[string]interface{} {
	m.mu.RLock()
	defer m.mu.RUnlock()

	var m1, m2 *VersionMetrics
	for _, v := range m.versions {
		if v.ID == v1ID && v.Metrics != nil {
			m1 = v.Metrics
		}
		if v.ID == v2ID && v.Metrics != nil {
			m2 = v.Metrics
		}
	}

	if m1 == nil || m2 == nil {
		return map[string]interface{}{"error": "version metrics not available"}
	}

	return map[string]interface{}{
		"latency_diff_ms":   m2.AvgLatencyMs - m1.AvgLatencyMs,
		"error_rate_diff":   m2.ErrorRate - m1.ErrorRate,
		"tokens_sec_diff":   m2.AvgTokensSec - m1.AvgTokensSec,
		"v1_requests":       m1.RequestCount,
		"v2_requests":       m2.RequestCount,
	}
}
