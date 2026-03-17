// gradient_accum.go implements gradient accumulation for virtual batch sizes.
//
// WHAT: On consumer GPUs with limited VRAM, you can't fit a large batch.
// Gradient accumulation processes N micro-batches, sums their gradients,
// then does one optimizer step -- same effect as a batch N times larger.
//
// EXAMPLE: With micro_batch=4 and accum_steps=8, you get an effective
// batch size of 32 while only using memory for 4 samples at a time.
//
// WHY: Larger effective batch sizes produce smoother gradients, better
// convergence, and fewer total training steps needed.
package native

import (
	"fmt"
	"math"
)

// GradientAccumulator manages accumulated gradients across micro-batches.
type GradientAccumulator struct {
	// Accumulated gradient buffers (one per parameter).
	accumulatedGrads map[string][]float32

	// Config.
	accumSteps   int     // Number of micro-batches before optimizer step
	currentStep  int     // Current micro-batch step (0..accumSteps-1)
	scaleFactor  float32 // 1.0 / accumSteps for averaging

	// Gradient clipping.
	maxGradNorm float32 // Maximum gradient norm (0 = no clipping)

	// Stats.
	totalSteps     int64
	totalClips     int64
	avgGradNorm    float64
}

// NewGradientAccumulator creates an accumulator for the given number of steps.
func NewGradientAccumulator(accumSteps int) *GradientAccumulator {
	if accumSteps < 1 {
		accumSteps = 1
	}
	return &GradientAccumulator{
		accumulatedGrads: make(map[string][]float32),
		accumSteps:       accumSteps,
		scaleFactor:      1.0 / float32(accumSteps),
		maxGradNorm:      1.0, // Default: clip at 1.0 like most frameworks
	}
}

// SetMaxGradNorm sets the maximum gradient norm for clipping.
// Set to 0 to disable gradient clipping.
func (ga *GradientAccumulator) SetMaxGradNorm(maxNorm float32) {
	ga.maxGradNorm = maxNorm
}

// Accumulate adds gradients from one micro-batch to the accumulator.
// Returns true if this was the last micro-batch (time to step the optimizer).
func (ga *GradientAccumulator) Accumulate(paramName string, grads []float32) bool {
	// Initialize buffer if first time.
	if _, ok := ga.accumulatedGrads[paramName]; !ok {
		ga.accumulatedGrads[paramName] = make([]float32, len(grads))
	}

	buf := ga.accumulatedGrads[paramName]

	// Accumulate (add scaled gradients).
	for i := range grads {
		buf[i] += grads[i] * ga.scaleFactor
	}

	ga.currentStep++

	// Check if we've accumulated enough micro-batches.
	if ga.currentStep >= ga.accumSteps {
		return true // Ready for optimizer step.
	}
	return false
}

// GetAccumulatedGrad returns the accumulated gradient for a parameter.
// This should be called after Accumulate returns true.
func (ga *GradientAccumulator) GetAccumulatedGrad(paramName string) []float32 {
	return ga.accumulatedGrads[paramName]
}

// ClipGradients applies gradient norm clipping across all accumulated parameters.
// Call this after all parameters have been accumulated but before the optimizer step.
func (ga *GradientAccumulator) ClipGradients() float64 {
	if ga.maxGradNorm <= 0 {
		return 0
	}

	// Compute global gradient norm.
	var totalNormSq float64
	for _, grads := range ga.accumulatedGrads {
		for _, g := range grads {
			totalNormSq += float64(g) * float64(g)
		}
	}
	totalNorm := math.Sqrt(totalNormSq)

	// Track stats.
	ga.avgGradNorm = ga.avgGradNorm*0.99 + totalNorm*0.01 // EMA

	// Clip if needed.
	if totalNorm > float64(ga.maxGradNorm) {
		clipCoef := float64(ga.maxGradNorm) / (totalNorm + 1e-6)
		for _, grads := range ga.accumulatedGrads {
			for i := range grads {
				grads[i] *= float32(clipCoef)
			}
		}
		ga.totalClips++
	}

	return totalNorm
}

// Reset clears accumulated gradients for the next accumulation cycle.
// Call this after the optimizer step.
func (ga *GradientAccumulator) Reset() {
	for name, buf := range ga.accumulatedGrads {
		for i := range buf {
			buf[i] = 0
		}
		ga.accumulatedGrads[name] = buf
	}
	ga.currentStep = 0
	ga.totalSteps++
}

// ShouldStep returns true if enough micro-batches have been accumulated.
func (ga *GradientAccumulator) ShouldStep() bool {
	return ga.currentStep >= ga.accumSteps
}

// EffectiveBatchSize returns the virtual batch size.
func (ga *GradientAccumulator) EffectiveBatchSize(microBatchSize int) int {
	return microBatchSize * ga.accumSteps
}

// Stats returns accumulation metrics.
func (ga *GradientAccumulator) Stats() map[string]interface{} {
	return map[string]interface{}{
		"accum_steps":         ga.accumSteps,
		"current_micro_step":  ga.currentStep,
		"total_optimizer_steps": ga.totalSteps,
		"total_clips":         ga.totalClips,
		"avg_grad_norm":       fmt.Sprintf("%.4f", ga.avgGradNorm),
		"max_grad_norm":       ga.maxGradNorm,
		"params_tracked":      len(ga.accumulatedGrads),
	}
}
