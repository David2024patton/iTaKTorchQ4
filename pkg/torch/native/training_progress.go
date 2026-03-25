// training_progress.go provides real-time progress tracking for LLM training.
//
// WHAT: Renders a terminal progress bar with ETA, loss curves, and throughput.
// Also writes a JSON status file that can be polled by external tools (Agent UI,
// web dashboards, etc.) for live monitoring.
//
// FEATURES:
//   - Terminal progress bar with percentage, ETA, and speed
//   - JSON status file updated every step (for web UI polling)
//   - Loss history tracking with trend detection
//   - Phase tracking for multi-phase training (SFT -> GRPO)
package native

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"strings"
	"time"
)

// TrainingProgress tracks and displays training progress.
type TrainingProgress struct {
	// Config.
	TotalSteps    int
	TotalEpochs   int
	StatusFilePath string // JSON status file for external monitoring

	// Current state.
	CurrentStep   int
	CurrentEpoch  int
	Phase         string // "sft", "grpo", "advanced/sft", "advanced/grpo"

	// Metrics history.
	Losses        []float32
	Rewards       []float32
	LearningRates []float32

	// Timing.
	StartTime     time.Time
	LastStepTime  time.Time
	StepTimes     []time.Duration // Last N step durations for ETA smoothing

	// Display config.
	BarWidth      int  // Width of the progress bar in characters
	ShowBar       bool // Show terminal progress bar
}

// NewTrainingProgress creates a progress tracker.
func NewTrainingProgress(totalSteps, totalEpochs int, statusFile string) *TrainingProgress {
	return &TrainingProgress{
		TotalSteps:     totalSteps,
		TotalEpochs:    totalEpochs,
		StatusFilePath: statusFile,
		Phase:          "sft",
		StartTime:      time.Now(),
		LastStepTime:   time.Now(),
		BarWidth:       40,
		ShowBar:        true,
	}
}

// TrainingStatus is the JSON structure written to the status file.
type TrainingStatus struct {
	Phase         string    `json:"phase"`
	Step          int       `json:"step"`
	TotalSteps    int       `json:"total_steps"`
	Epoch         int       `json:"epoch"`
	TotalEpochs   int       `json:"total_epochs"`
	Progress      float64   `json:"progress"`      // 0.0-1.0
	Loss          float32   `json:"loss"`
	AvgLoss       float32   `json:"avg_loss"`
	Reward        float32   `json:"reward,omitempty"`
	LR            float32   `json:"lr"`
	TokensPerSec  float64   `json:"tokens_per_sec"`
	ElapsedSec    float64   `json:"elapsed_sec"`
	ETASec        float64   `json:"eta_sec"`
	ETAFormatted  string    `json:"eta_formatted"`
	StartedAt     time.Time `json:"started_at"`
	UpdatedAt     time.Time `json:"updated_at"`
	LossHistory   []float32 `json:"loss_history,omitempty"`
	LossTrend     string    `json:"loss_trend"` // "decreasing", "stable", "increasing"
}

// Update records a new training step and displays progress.
func (tp *TrainingProgress) Update(step int, loss, lr float32, tokensProcessed int64) {
	now := time.Now()
	stepDur := now.Sub(tp.LastStepTime)
	tp.LastStepTime = now
	tp.CurrentStep = step

	// Track step times (keep last 20 for smooth ETA).
	tp.StepTimes = append(tp.StepTimes, stepDur)
	if len(tp.StepTimes) > 20 {
		tp.StepTimes = tp.StepTimes[len(tp.StepTimes)-20:]
	}

	tp.Losses = append(tp.Losses, loss)
	tp.LearningRates = append(tp.LearningRates, lr)

	// Compute metrics.
	elapsed := now.Sub(tp.StartTime)
	progress := float64(step) / float64(tp.TotalSteps)
	if progress > 1.0 {
		progress = 1.0
	}

	tps := float64(tokensProcessed) / elapsed.Seconds()
	eta := tp.estimateETA(step)

	// Render terminal progress bar.
	if tp.ShowBar {
		tp.renderBar(progress, loss, lr, tps, elapsed, eta)
	}

	// Write JSON status file.
	if tp.StatusFilePath != "" {
		tp.writeStatusFile(progress, loss, lr, tps, elapsed, eta)
	}
}

// UpdateReward records a GRPO reward alongside regular metrics.
func (tp *TrainingProgress) UpdateReward(step int, loss, reward, lr float32, tokensProcessed int64) {
	tp.Rewards = append(tp.Rewards, reward)
	tp.Update(step, loss, lr, tokensProcessed)
}

// SetPhase changes the current training phase label.
func (tp *TrainingProgress) SetPhase(phase string) {
	tp.Phase = phase
	fmt.Printf("\n[Progress] Phase changed to: %s\n", phase)
}

// SetEpoch updates the current epoch.
func (tp *TrainingProgress) SetEpoch(epoch int) {
	tp.CurrentEpoch = epoch
}

// Finish renders the final progress line.
func (tp *TrainingProgress) Finish() {
	elapsed := time.Since(tp.StartTime)
	avgLoss := tp.avgLoss()

	// Show complete bar.
	bar := strings.Repeat("█", tp.BarWidth)
	fmt.Printf("\r[%s] 100%% | %s | loss=%.4f | %v\n",
		tp.Phase, bar, avgLoss, elapsed.Round(time.Second))

	// Summary.
	fmt.Printf("[Progress] Training complete: %d steps in %v\n",
		tp.CurrentStep, elapsed.Round(time.Second))
	if len(tp.Losses) > 1 {
		fmt.Printf("[Progress] Loss: %.4f -> %.4f (%.1f%% %s)\n",
			tp.Losses[0], tp.Losses[len(tp.Losses)-1],
			math.Abs(float64(tp.Losses[0]-tp.Losses[len(tp.Losses)-1])/float64(tp.Losses[0])*100),
			tp.lossTrend())
	}
}

// renderBar prints a terminal progress bar with metrics.
func (tp *TrainingProgress) renderBar(progress float64, loss, lr float32, tps float64, elapsed, eta time.Duration) {
	filled := int(progress * float64(tp.BarWidth))
	if filled > tp.BarWidth {
		filled = tp.BarWidth
	}
	empty := tp.BarWidth - filled

	bar := strings.Repeat("█", filled) + strings.Repeat("░", empty)
	pct := int(progress * 100)

	etaStr := formatDuration(eta)

	// Use \r for in-place update on same line.
	fmt.Printf("\r\033[K[%s] %s %3d%% | step %d/%d | epoch %d/%d | loss=%.4f | lr=%.2e | %.0f tok/s | ETA %s",
		tp.Phase, bar, pct,
		tp.CurrentStep, tp.TotalSteps,
		tp.CurrentEpoch+1, tp.TotalEpochs,
		loss, lr, tps, etaStr)
}

// writeStatusFile writes a JSON status file for external monitoring.
func (tp *TrainingProgress) writeStatusFile(progress float64, loss, lr float32, tps float64, elapsed, eta time.Duration) {
	status := TrainingStatus{
		Phase:        tp.Phase,
		Step:         tp.CurrentStep,
		TotalSteps:   tp.TotalSteps,
		Epoch:        tp.CurrentEpoch + 1,
		TotalEpochs:  tp.TotalEpochs,
		Progress:     progress,
		Loss:         loss,
		AvgLoss:      tp.avgLoss(),
		LR:           lr,
		TokensPerSec: tps,
		ElapsedSec:   elapsed.Seconds(),
		ETASec:       eta.Seconds(),
		ETAFormatted: formatDuration(eta),
		StartedAt:    tp.StartTime,
		UpdatedAt:    time.Now(),
		LossTrend:    tp.lossTrend(),
	}

	// Include last 50 loss values for charting.
	if len(tp.Losses) > 50 {
		status.LossHistory = tp.Losses[len(tp.Losses)-50:]
	} else {
		status.LossHistory = tp.Losses
	}

	if len(tp.Rewards) > 0 {
		status.Reward = tp.Rewards[len(tp.Rewards)-1]
	}

	data, err := json.MarshalIndent(status, "", "  ")
	if err != nil {
		return
	}

	// Write atomically (write to tmp then rename).
	tmpPath := tp.StatusFilePath + ".tmp"
	if err := os.WriteFile(tmpPath, data, 0644); err != nil {
		return
	}
	os.Rename(tmpPath, tp.StatusFilePath)
}

// estimateETA estimates remaining time using smoothed step durations.
func (tp *TrainingProgress) estimateETA(currentStep int) time.Duration {
	remaining := tp.TotalSteps - currentStep
	if remaining <= 0 || len(tp.StepTimes) == 0 {
		return 0
	}

	// Average of recent step times.
	var totalDur time.Duration
	for _, d := range tp.StepTimes {
		totalDur += d
	}
	avgStep := totalDur / time.Duration(len(tp.StepTimes))

	return avgStep * time.Duration(remaining)
}

// avgLoss returns the average of recent losses.
func (tp *TrainingProgress) avgLoss() float32 {
	if len(tp.Losses) == 0 {
		return 0
	}
	n := 10
	if n > len(tp.Losses) {
		n = len(tp.Losses)
	}
	var sum float32
	for _, l := range tp.Losses[len(tp.Losses)-n:] {
		sum += l
	}
	return sum / float32(n)
}

// lossTrend detects if loss is decreasing, stable, or increasing.
func (tp *TrainingProgress) lossTrend() string {
	if len(tp.Losses) < 10 {
		return "warming_up"
	}

	// Compare first half average vs second half average.
	mid := len(tp.Losses) / 2
	var firstSum, secondSum float32
	for _, l := range tp.Losses[:mid] {
		firstSum += l
	}
	for _, l := range tp.Losses[mid:] {
		secondSum += l
	}
	firstAvg := firstSum / float32(mid)
	secondAvg := secondSum / float32(len(tp.Losses)-mid)

	ratio := secondAvg / firstAvg
	if ratio < 0.95 {
		return "decreasing"
	} else if ratio > 1.05 {
		return "increasing"
	}
	return "stable"
}

// formatDuration formats a duration as "Xh Ym Zs" or "Ym Zs".
func formatDuration(d time.Duration) string {
	if d <= 0 {
		return "0s"
	}
	h := int(d.Hours())
	m := int(d.Minutes()) % 60
	s := int(d.Seconds()) % 60

	if h > 0 {
		return fmt.Sprintf("%dh%02dm%02ds", h, m, s)
	}
	if m > 0 {
		return fmt.Sprintf("%dm%02ds", m, s)
	}
	return fmt.Sprintf("%ds", s)
}
