// lr_scheduler.go implements learning rate scheduling strategies for training.
//
// WHAT: A fixed learning rate is suboptimal. Starting high and decaying allows
// faster initial progress without instability. The right schedule can cut
// total training time by 30-50% by safely using higher learning rates.
//
// SCHEDULES:
//   CosineAnnealing: LR follows a cosine curve from max to min. The most
//     common choice for LLM training (used by GPT-3, LLaMA, Chinchilla).
//   WarmupCosine: Linear warmup for N steps, then cosine decay. Prevents
//     early training instability from high gradients on random weights.
//   OneCycleLR: Ramp up to max LR then back down. Finds the optimal LR
//     automatically (Smith 2018). Fastest convergence for fine-tuning.
//   LinearDecay: Simple linear decay. Good baseline.
//   ConstantWithWarmup: Warmup then flat. Good for short fine-tuning runs.
package native

import (
	"math"
)

// LRScheduler computes the learning rate for a given step.
type LRScheduler interface {
	// GetLR returns the learning rate for the given training step.
	GetLR(step int) float32
	// Name returns the scheduler name for logging.
	Name() string
}

// ---------- Cosine Annealing ----------

type cosineScheduler struct {
	maxLR    float32
	minLR    float32
	totalSteps int
}

// CosineAnnealingLR decays LR from maxLR to minLR following a cosine curve.
// This is the standard schedule used by LLaMA, GPT-3, and Chinchilla.
func CosineAnnealingLR(maxLR, minLR float32, totalSteps int) LRScheduler {
	return &cosineScheduler{maxLR: maxLR, minLR: minLR, totalSteps: totalSteps}
}

func (s *cosineScheduler) GetLR(step int) float32 {
	if step >= s.totalSteps {
		return s.minLR
	}
	progress := float64(step) / float64(s.totalSteps)
	cosVal := (1.0 + math.Cos(math.Pi*progress)) / 2.0
	return s.minLR + (s.maxLR-s.minLR)*float32(cosVal)
}

func (s *cosineScheduler) Name() string { return "cosine_annealing" }

// ---------- Warmup + Cosine ----------

type warmupCosineScheduler struct {
	maxLR       float32
	minLR       float32
	warmupSteps int
	totalSteps  int
}

// WarmupCosineLR does linear warmup for warmupSteps, then cosine decay.
// The most popular schedule for pre-training and fine-tuning LLMs.
func WarmupCosineLR(maxLR, minLR float32, warmupSteps, totalSteps int) LRScheduler {
	return &warmupCosineScheduler{
		maxLR: maxLR, minLR: minLR,
		warmupSteps: warmupSteps, totalSteps: totalSteps,
	}
}

func (s *warmupCosineScheduler) GetLR(step int) float32 {
	// Warmup phase: linear ramp from 0 to maxLR.
	if step < s.warmupSteps {
		return s.maxLR * float32(step) / float32(s.warmupSteps)
	}
	// Cosine decay phase.
	decaySteps := s.totalSteps - s.warmupSteps
	if decaySteps <= 0 {
		return s.minLR
	}
	progress := float64(step-s.warmupSteps) / float64(decaySteps)
	if progress > 1.0 {
		progress = 1.0
	}
	cosVal := (1.0 + math.Cos(math.Pi*progress)) / 2.0
	return s.minLR + (s.maxLR-s.minLR)*float32(cosVal)
}

func (s *warmupCosineScheduler) Name() string { return "warmup_cosine" }

// ---------- OneCycle LR ----------

type oneCycleScheduler struct {
	maxLR      float32
	totalSteps int
	pctStart   float32 // Percentage of training spent ramping up (default: 0.3)
	divFactor  float32 // Initial LR = maxLR / divFactor (default: 25)
	finalDiv   float32 // Final LR = maxLR / (divFactor * finalDiv) (default: 1e4)
}

// OneCycleLR implements the 1cycle policy (Smith 2018).
// Ramps LR up to maxLR over pctStart of training, then anneals down.
// Typically achieves the fastest convergence for fine-tuning tasks.
func OneCycleLR(maxLR float32, totalSteps int) LRScheduler {
	return &oneCycleScheduler{
		maxLR:      maxLR,
		totalSteps: totalSteps,
		pctStart:   0.3,
		divFactor:  25.0,
		finalDiv:   1e4,
	}
}

func (s *oneCycleScheduler) GetLR(step int) float32 {
	if step >= s.totalSteps {
		return s.maxLR / (s.divFactor * s.finalDiv)
	}

	rampUpEnd := int(float32(s.totalSteps) * s.pctStart)
	initialLR := s.maxLR / s.divFactor
	finalLR := s.maxLR / (s.divFactor * s.finalDiv)

	if step < rampUpEnd {
		// Phase 1: Ramp up from initialLR to maxLR.
		progress := float64(step) / float64(rampUpEnd)
		cosVal := (1.0 - math.Cos(math.Pi*progress)) / 2.0
		return initialLR + (s.maxLR-initialLR)*float32(cosVal)
	}

	// Phase 2: Anneal from maxLR to finalLR.
	decaySteps := s.totalSteps - rampUpEnd
	progress := float64(step-rampUpEnd) / float64(decaySteps)
	cosVal := (1.0 + math.Cos(math.Pi*progress)) / 2.0
	return finalLR + (s.maxLR-finalLR)*float32(cosVal)
}

func (s *oneCycleScheduler) Name() string { return "one_cycle" }

// ---------- Linear Decay ----------

type linearScheduler struct {
	startLR    float32
	endLR      float32
	totalSteps int
}

// LinearDecayLR linearly decays from startLR to endLR over totalSteps.
func LinearDecayLR(startLR, endLR float32, totalSteps int) LRScheduler {
	return &linearScheduler{startLR: startLR, endLR: endLR, totalSteps: totalSteps}
}

func (s *linearScheduler) GetLR(step int) float32 {
	if step >= s.totalSteps {
		return s.endLR
	}
	progress := float32(step) / float32(s.totalSteps)
	return s.startLR + (s.endLR-s.startLR)*progress
}

func (s *linearScheduler) Name() string { return "linear_decay" }

// ---------- Constant with Warmup ----------

type constantWarmupScheduler struct {
	lr          float32
	warmupSteps int
}

// ConstantWithWarmupLR does linear warmup then holds constant.
// Simple and effective for short fine-tuning runs.
func ConstantWithWarmupLR(lr float32, warmupSteps int) LRScheduler {
	return &constantWarmupScheduler{lr: lr, warmupSteps: warmupSteps}
}

func (s *constantWarmupScheduler) GetLR(step int) float32 {
	if step < s.warmupSteps {
		return s.lr * float32(step) / float32(s.warmupSteps)
	}
	return s.lr
}

func (s *constantWarmupScheduler) Name() string { return "constant_warmup" }

// ---------- Warmup-Stable-Decay (WSD) ----------
// Used by Chinchilla and newer models. Three phases:
// warmup -> stable (constant max LR) -> decay.

type wsdScheduler struct {
	maxLR        float32
	minLR        float32
	warmupSteps  int
	stableSteps  int
	decaySteps   int
}

// WarmupStableDecayLR implements the 3-phase schedule from Chinchilla/Gemma.
// Warmup for W steps, hold at maxLR for S steps, then cosine decay for D steps.
func WarmupStableDecayLR(maxLR, minLR float32, warmupSteps, stableSteps, decaySteps int) LRScheduler {
	return &wsdScheduler{
		maxLR: maxLR, minLR: minLR,
		warmupSteps: warmupSteps, stableSteps: stableSteps, decaySteps: decaySteps,
	}
}

func (s *wsdScheduler) GetLR(step int) float32 {
	// Phase 1: Warmup.
	if step < s.warmupSteps {
		return s.maxLR * float32(step) / float32(s.warmupSteps)
	}
	// Phase 2: Stable.
	stableEnd := s.warmupSteps + s.stableSteps
	if step < stableEnd {
		return s.maxLR
	}
	// Phase 3: Decay.
	decayProgress := float64(step-stableEnd) / float64(s.decaySteps)
	if decayProgress > 1.0 {
		return s.minLR
	}
	cosVal := (1.0 + math.Cos(math.Pi*decayProgress)) / 2.0
	return s.minLR + (s.maxLR-s.minLR)*float32(cosVal)
}

func (s *wsdScheduler) Name() string { return "warmup_stable_decay" }
