// health_monitor.go provides real-time system health monitoring for
// GPU resources, inference throughput, and training metrics.
//
// WHAT: A background health monitor that tracks:
//   - GPU temperature, memory usage, utilization
//   - Inference throughput (tok/s), latency (TTFT, ITL)
//   - Training step time, loss, learning rate
//   - System memory, disk I/O
//   - Alert thresholds and automatic throttling
//
// WHY: Production serving needs observability. Without monitoring,
// you can't detect GPU thermal throttling, memory leaks, throughput
// degradation, or failing requests until it's too late.
package native

import (
	"fmt"
	"runtime"
	"sync"
	"time"
)

// HealthStatus represents overall system health.
type HealthStatus int

const (
	HealthOK       HealthStatus = iota
	HealthWarning                // Approaching limits
	HealthCritical               // Immediate action needed
	HealthDegraded               // Reduced performance
)

// MonitorGPUMetrics holds GPU health data.
type MonitorGPUMetrics struct {
	DeviceID      int
	Name          string
	Temperature   float32 // Celsius
	MemoryUsed    int64   // Bytes
	MemoryTotal   int64   // Bytes
	Utilization   float32 // 0-100%
	PowerDraw     float32 // Watts
	ClockSpeed    int     // MHz
	ThrottleCount int64   // Number of thermal throttle events
}

// MonitorInferenceMetrics holds inference performance data.
type MonitorInferenceMetrics struct {
	TokensPerSecond   float64
	TimeToFirstToken  time.Duration // Average TTFT
	InterTokenLatency time.Duration // Average ITL
	ActiveRequests    int
	QueuedRequests    int
	TotalRequests     int64
	FailedRequests    int64
	CacheHitRate      float64 // KV cache / prefix cache hit rate
}

// MonitorTrainingMetrics holds training performance data.
type MonitorTrainingMetrics struct {
	StepTime        time.Duration
	Loss            float64
	LearningRate    float64
	GradNorm        float64
	ThroughputSPS   float64 // Samples per second
	MemoryPeakMB    float64
	Epoch           int
	Step            int
}

// MonitorSystemMetrics holds OS-level metrics.
type MonitorSystemMetrics struct {
	CPUUsage      float64
	RAMUsed       int64
	RAMTotal      int64
	GoRoutines    int
	HeapAllocMB   float64
	GCPauseMs     float64
}

// AlertRule defines a threshold-based alert.
type AlertRule struct {
	Name       string
	Metric     string  // e.g., "gpu_temp", "tok_per_sec"
	Threshold  float64
	Direction  string  // "above" or "below"
	Duration   time.Duration // How long the threshold must be exceeded
	Triggered  bool
	TriggeredAt time.Time
	Callback   func(rule *AlertRule, value float64)
}

// HealthMonitor tracks system health and triggers alerts.
type HealthMonitor struct {
	mu sync.RWMutex

	// Latest metrics.
	MonitorGPUMetrics       []MonitorGPUMetrics
	MonitorInferenceMetrics MonitorInferenceMetrics
	MonitorTrainingMetrics  MonitorTrainingMetrics
	MonitorSystemMetrics    MonitorSystemMetrics

	// Alert rules.
	alerts []*AlertRule

	// History (ring buffer of recent snapshots).
	history     []healthSnapshot
	historyIdx  int
	maxHistory  int

	// Overall status.
	status HealthStatus

	// Control.
	running    bool
	interval   time.Duration
	stopCh     chan struct{}
}

type healthSnapshot struct {
	Timestamp time.Time
	TokPerSec float64
	GPUTemp   float32
	GPUMem    float64 // Percentage
	Status    HealthStatus
}

// HealthMonitorConfig configures the health monitor.
type HealthMonitorConfig struct {
	PollInterval time.Duration // How often to collect metrics (default: 1s)
	MaxHistory   int           // Number of snapshots to keep (default: 3600)
}

// NewHealthMonitor creates a health monitor with default alerts.
func NewHealthMonitor(config HealthMonitorConfig) *HealthMonitor {
	if config.PollInterval == 0 {
		config.PollInterval = time.Second
	}
	if config.MaxHistory == 0 {
		config.MaxHistory = 3600
	}

	hm := &HealthMonitor{
		maxHistory: config.MaxHistory,
		history:    make([]healthSnapshot, config.MaxHistory),
		interval:   config.PollInterval,
		stopCh:     make(chan struct{}),
		status:     HealthOK,
	}

	// Default alert rules.
	hm.AddAlert(&AlertRule{
		Name:      "gpu_temp_critical",
		Metric:    "gpu_temp",
		Threshold: 90,
		Direction: "above",
		Duration:  10 * time.Second,
	})
	hm.AddAlert(&AlertRule{
		Name:      "gpu_mem_high",
		Metric:    "gpu_mem_pct",
		Threshold: 95,
		Direction: "above",
		Duration:  30 * time.Second,
	})
	hm.AddAlert(&AlertRule{
		Name:      "throughput_drop",
		Metric:    "tok_per_sec",
		Threshold: 1,
		Direction: "below",
		Duration:  60 * time.Second,
	})

	return hm
}

// AddAlert registers a new alert rule.
func (hm *HealthMonitor) AddAlert(rule *AlertRule) {
	hm.mu.Lock()
	defer hm.mu.Unlock()
	hm.alerts = append(hm.alerts, rule)
}

// UpdateGPU records GPU metrics.
func (hm *HealthMonitor) UpdateGPU(metrics []MonitorGPUMetrics) {
	hm.mu.Lock()
	defer hm.mu.Unlock()
	hm.MonitorGPUMetrics = metrics
	hm.checkAlerts()
}

// UpdateInference records inference metrics.
func (hm *HealthMonitor) UpdateInference(metrics MonitorInferenceMetrics) {
	hm.mu.Lock()
	defer hm.mu.Unlock()
	hm.MonitorInferenceMetrics = metrics
	hm.checkAlerts()
}

// UpdateTraining records training metrics.
func (hm *HealthMonitor) UpdateTraining(metrics MonitorTrainingMetrics) {
	hm.mu.Lock()
	defer hm.mu.Unlock()
	hm.MonitorTrainingMetrics = metrics
}

// collectSystem polls OS-level metrics.
func (hm *HealthMonitor) collectSystem() {
	var memStats runtime.MemStats
	runtime.ReadMemStats(&memStats)

	hm.MonitorSystemMetrics = MonitorSystemMetrics{
		GoRoutines:  runtime.NumGoroutine(),
		HeapAllocMB: float64(memStats.HeapAlloc) / (1024 * 1024),
		GCPauseMs:   float64(memStats.PauseNs[(memStats.NumGC+255)%256]) / 1e6,
		RAMUsed:     int64(memStats.Sys),
	}
}

// checkAlerts evaluates all alert rules against current metrics.
func (hm *HealthMonitor) checkAlerts() {
	now := time.Now()

	for _, rule := range hm.alerts {
		var value float64

		switch rule.Metric {
		case "gpu_temp":
			if len(hm.MonitorGPUMetrics) > 0 {
				value = float64(hm.MonitorGPUMetrics[0].Temperature)
			}
		case "gpu_mem_pct":
			if len(hm.MonitorGPUMetrics) > 0 && hm.MonitorGPUMetrics[0].MemoryTotal > 0 {
				value = float64(hm.MonitorGPUMetrics[0].MemoryUsed) / float64(hm.MonitorGPUMetrics[0].MemoryTotal) * 100
			}
		case "tok_per_sec":
			value = hm.MonitorInferenceMetrics.TokensPerSecond
		}

		exceeded := false
		switch rule.Direction {
		case "above":
			exceeded = value > rule.Threshold
		case "below":
			exceeded = value < rule.Threshold && value > 0
		}

		if exceeded {
			if !rule.Triggered {
				rule.Triggered = true
				rule.TriggeredAt = now
			}
			// Fire callback if duration exceeded.
			if rule.Callback != nil && now.Sub(rule.TriggeredAt) > rule.Duration {
				rule.Callback(rule, value)
			}
		} else {
			rule.Triggered = false
		}
	}

	// Update overall status.
	hm.status = HealthOK
	for _, rule := range hm.alerts {
		if rule.Triggered {
			hm.status = HealthWarning
			if rule.Metric == "gpu_temp" || rule.Metric == "gpu_mem_pct" {
				hm.status = HealthCritical
			}
			break
		}
	}
}

// Snapshot records current state into history.
func (hm *HealthMonitor) Snapshot() {
	hm.mu.Lock()
	defer hm.mu.Unlock()

	hm.collectSystem()

	snap := healthSnapshot{
		Timestamp: time.Now(),
		TokPerSec: hm.MonitorInferenceMetrics.TokensPerSecond,
		Status:    hm.status,
	}
	if len(hm.MonitorGPUMetrics) > 0 {
		snap.GPUTemp = hm.MonitorGPUMetrics[0].Temperature
		if hm.MonitorGPUMetrics[0].MemoryTotal > 0 {
			snap.GPUMem = float64(hm.MonitorGPUMetrics[0].MemoryUsed) / float64(hm.MonitorGPUMetrics[0].MemoryTotal) * 100
		}
	}

	hm.history[hm.historyIdx%hm.maxHistory] = snap
	hm.historyIdx++
}

// Status returns the current overall health status.
func (hm *HealthMonitor) Status() HealthStatus {
	hm.mu.RLock()
	defer hm.mu.RUnlock()
	return hm.status
}

// Report generates a health report.
func (hm *HealthMonitor) Report() map[string]interface{} {
	hm.mu.RLock()
	defer hm.mu.RUnlock()

	statusNames := map[HealthStatus]string{
		HealthOK:       "OK",
		HealthWarning:  "WARNING",
		HealthCritical: "CRITICAL",
		HealthDegraded: "DEGRADED",
	}

	report := map[string]interface{}{
		"status":       statusNames[hm.status],
		"goroutines":   hm.MonitorSystemMetrics.GoRoutines,
		"heap_mb":      fmt.Sprintf("%.1f", hm.MonitorSystemMetrics.HeapAllocMB),
		"gc_pause_ms":  fmt.Sprintf("%.2f", hm.MonitorSystemMetrics.GCPauseMs),
		"tok_per_sec":  fmt.Sprintf("%.1f", hm.MonitorInferenceMetrics.TokensPerSecond),
		"active_reqs":  hm.MonitorInferenceMetrics.ActiveRequests,
		"queued_reqs":  hm.MonitorInferenceMetrics.QueuedRequests,
		"total_reqs":   hm.MonitorInferenceMetrics.TotalRequests,
		"failed_reqs":  hm.MonitorInferenceMetrics.FailedRequests,
	}

	for i, gpu := range hm.MonitorGPUMetrics {
		prefix := fmt.Sprintf("gpu_%d", i)
		report[prefix+"_temp"] = fmt.Sprintf("%.0f C", gpu.Temperature)
		report[prefix+"_mem_pct"] = fmt.Sprintf("%.1f%%", float64(gpu.MemoryUsed)/float64(gpu.MemoryTotal+1)*100)
		report[prefix+"_util"] = fmt.Sprintf("%.0f%%", gpu.Utilization)
		report[prefix+"_power"] = fmt.Sprintf("%.0f W", gpu.PowerDraw)
	}

	if hm.MonitorTrainingMetrics.Step > 0 {
		report["train_step"] = hm.MonitorTrainingMetrics.Step
		report["train_loss"] = fmt.Sprintf("%.4f", hm.MonitorTrainingMetrics.Loss)
		report["train_lr"] = fmt.Sprintf("%.2e", hm.MonitorTrainingMetrics.LearningRate)
		report["train_step_ms"] = hm.MonitorTrainingMetrics.StepTime.Milliseconds()
	}

	// Active alerts.
	activeAlerts := make([]string, 0)
	for _, rule := range hm.alerts {
		if rule.Triggered {
			activeAlerts = append(activeAlerts, rule.Name)
		}
	}
	report["active_alerts"] = activeAlerts

	return report
}
