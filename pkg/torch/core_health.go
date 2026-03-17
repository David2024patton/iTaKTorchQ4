package torch

import (
	"fmt"
	"time"

	"github.com/David2024patton/iTaKCore/pkg/health"
)

// TorchHealthChecker implements Core's health.Checker interface for Torch.
// Returns a health.Report with standardized sub-checks for model status
// and GPU availability.
type TorchHealthChecker struct {
	server    *Server
	startTime time.Time
	version   string
}

// NewTorchHealthChecker creates a health checker bound to a Torch server.
func NewTorchHealthChecker(server *Server, version string) *TorchHealthChecker {
	return &TorchHealthChecker{
		server:    server,
		startTime: time.Now(),
		version:   version,
	}
}

// HealthCheck returns a Core-compatible health report for this Torch instance.
func (thc *TorchHealthChecker) HealthCheck() health.Report {
	defer TimeTrace("TorchHealthChecker.HealthCheck")()
	LogTrace("[HealthChecker] Running health check")
	status := health.StatusHealthy
	checks := make(map[string]health.Check)
	metadata := make(map[string]string)

	// Check 1: Is a model loaded?
	if thc.server.engine != nil {
		modelName := thc.server.engine.ModelName()
		if modelName != "" {
			checks["model_loaded"] = health.Check{
				Status:  health.StatusHealthy,
				Message: modelName,
			}
			metadata["model"] = modelName
		} else {
			checks["model_loaded"] = health.Check{
				Status:  health.StatusUnhealthy,
				Message: "no model loaded",
			}
			status = health.StatusDegraded
		}
	} else {
		checks["model_loaded"] = health.Check{
			Status:  health.StatusUnhealthy,
			Message: "engine is nil",
		}
		status = health.StatusUnhealthy
	}

	// Check 2: GPU backend status.
	if thc.server.engine != nil {
		stats := thc.server.engine.GetStats()
		if te, ok := thc.server.engine.(*TorchEngine); ok {
			backend := te.opts.Backend
			if backend == "" {
				backend = "auto"
			}
			metadata["backend"] = backend

			if backend != "cpu" {
				checks["gpu_available"] = health.Check{
					Status:  health.StatusHealthy,
					Message: backend,
				}
			} else {
				checks["gpu_available"] = health.Check{
					Status:  health.StatusDegraded,
					Message: "CPU-only mode",
				}
			}

			// Engine performance metadata.
			if stats.RequestCount > 0 {
				metadata["avg_tok_per_sec"] = fmt.Sprintf("%.1f", stats.AvgTokPerSec)
				metadata["total_requests"] = fmt.Sprintf("%d", stats.RequestCount)
			}
		}
	}

	// Check 3: Registry health (multi-model mode).
	if thc.server.registry != nil {
		rStats := thc.server.registry.Stats()
		checks["model_registry"] = health.Check{
			Status:  health.StatusHealthy,
			Message: fmt.Sprintf("%d/%d models loaded", rStats.LoadedModels, rStats.MaxModels),
		}
		metadata["registry_loads"] = fmt.Sprintf("%d", rStats.TotalLoads)
		metadata["registry_evicts"] = fmt.Sprintf("%d", rStats.TotalEvicts)
	}

	// Check 4: Scheduler health.
	if thc.server.scheduler != nil {
		depth := thc.server.scheduler.QueueDepth()
		if depth > 50 {
			checks["scheduler"] = health.Check{
				Status:  health.StatusDegraded,
				Message: fmt.Sprintf("queue depth: %d (high)", depth),
			}
			if status == health.StatusHealthy {
				status = health.StatusDegraded
			}
		} else {
			checks["scheduler"] = health.Check{
				Status:  health.StatusHealthy,
				Message: fmt.Sprintf("queue depth: %d", depth),
			}
		}
	}

	return health.Report{
		Module:    "torch",
		Status:    status,
		Version:   thc.version,
		Uptime:    time.Since(thc.startTime).Truncate(time.Second).String(),
		Checks:    checks,
		Metadata:  metadata,
		Timestamp: time.Now(),
	}
}
