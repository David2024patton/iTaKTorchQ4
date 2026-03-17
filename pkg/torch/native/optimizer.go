// optimizer.go implements AdamW optimizer with LR scheduling for training.
//
// WHY: AdamW is the standard optimizer for LLM training. It combines
// adaptive learning rates (Adam) with decoupled weight decay, which
// prevents the weight decay from being scaled by the adaptive learning rate.
//
// FEATURES:
//   - AdamW with decoupled weight decay
//   - Per-parameter first/second moment tracking
//   - Gradient clipping by max norm
//   - Learning rate warmup + cosine annealing schedule
package native

import (
	"fmt"
	"math"
)

// AdamWConfig holds optimizer hyperparameters.
type AdamWConfig struct {
	LearningRate float32 // Peak learning rate (e.g., 1e-4)
	Beta1        float32 // First moment decay (default: 0.9)
	Beta2        float32 // Second moment decay (default: 0.999)
	Epsilon      float32 // Numerical stability (default: 1e-8)
	WeightDecay  float32 // Decoupled weight decay (default: 0.01)
	MaxGradNorm  float32 // Gradient clipping threshold (default: 1.0)
	WarmupSteps  int     // LR warmup steps
	TotalSteps   int     // Total training steps (for cosine decay)
}

// DefaultAdamWConfig returns standard LLM fine-tuning hyperparameters.
func DefaultAdamWConfig(totalSteps int) AdamWConfig {
	return AdamWConfig{
		LearningRate: 1e-4,
		Beta1:        0.9,
		Beta2:        0.999,
		Epsilon:      1e-8,
		WeightDecay:  0.01,
		MaxGradNorm:  1.0,
		WarmupSteps:  totalSteps / 10, // 10% warmup
		TotalSteps:   totalSteps,
	}
}

// AdamW implements the AdamW optimizer.
type AdamW struct {
	config AdamWConfig
	params []*GradTensor // All trainable parameters

	// Per-parameter state.
	m    [][]float32 // First moment estimates
	v    [][]float32 // Second moment estimates
	step int         // Global step counter
}

// NewAdamW creates an AdamW optimizer for the given parameters.
func NewAdamW(params []*GradTensor, config AdamWConfig) *AdamW {
	opt := &AdamW{
		config: config,
		params: params,
		m:      make([][]float32, len(params)),
		v:      make([][]float32, len(params)),
	}

	for i, p := range params {
		opt.m[i] = make([]float32, len(p.Data))
		opt.v[i] = make([]float32, len(p.Data))
	}

	fmt.Printf("[AdamW] Initialized with %d parameter groups, lr=%.2e, wd=%.4f\n",
		len(params), config.LearningRate, config.WeightDecay)
	return opt
}

// Step performs one optimization step: clip gradients, update moments, update parameters.
func (opt *AdamW) Step() {
	opt.step++
	lr := opt.currentLR()

	// Gradient clipping across all parameters.
	opt.clipGradients()

	beta1 := opt.config.Beta1
	beta2 := opt.config.Beta2
	eps := opt.config.Epsilon
	wd := opt.config.WeightDecay

	// Bias correction factors.
	bc1 := float32(1.0 - math.Pow(float64(beta1), float64(opt.step)))
	bc2 := float32(1.0 - math.Pow(float64(beta2), float64(opt.step)))

	for i, p := range opt.params {
		if p.Grad == nil {
			continue
		}

		for j := range p.Data {
			g := p.Grad[j]

			// Update biased first/second moment estimates.
			opt.m[i][j] = beta1*opt.m[i][j] + (1-beta1)*g
			opt.v[i][j] = beta2*opt.v[i][j] + (1-beta2)*g*g

			// Bias-corrected moments.
			mHat := opt.m[i][j] / bc1
			vHat := opt.v[i][j] / bc2

			// AdamW update: decoupled weight decay applied directly to weights.
			p.Data[j] -= lr * (mHat/(float32(math.Sqrt(float64(vHat)))+eps) + wd*p.Data[j])
		}
	}
}

// clipGradients applies max-norm gradient clipping across all parameters.
func (opt *AdamW) clipGradients() {
	if opt.config.MaxGradNorm <= 0 {
		return
	}

	// Compute total gradient norm.
	var totalNormSq float64
	for _, p := range opt.params {
		if p.Grad == nil {
			continue
		}
		for _, g := range p.Grad {
			totalNormSq += float64(g) * float64(g)
		}
	}

	totalNorm := float32(math.Sqrt(totalNormSq))
	if totalNorm <= opt.config.MaxGradNorm {
		return
	}

	// Scale all gradients down.
	scale := opt.config.MaxGradNorm / totalNorm
	for _, p := range opt.params {
		if p.Grad == nil {
			continue
		}
		for j := range p.Grad {
			p.Grad[j] *= scale
		}
	}
}

// currentLR computes the current learning rate with warmup + cosine decay.
func (opt *AdamW) currentLR() float32 {
	step := opt.step
	warmup := opt.config.WarmupSteps
	total := opt.config.TotalSteps
	peakLR := opt.config.LearningRate

	if total <= 0 {
		return peakLR
	}

	if step <= warmup && warmup > 0 {
		// Linear warmup.
		return peakLR * float32(step) / float32(warmup)
	}

	// Cosine annealing to 10% of peak LR.
	progress := float64(step-warmup) / float64(total-warmup)
	if progress > 1.0 {
		progress = 1.0
	}
	minLR := peakLR * 0.1
	return minLR + (peakLR-minLR)*float32(1+math.Cos(math.Pi*progress))/2
}

// ZeroGrad resets gradients for all parameters.
func (opt *AdamW) ZeroGrad() {
	for _, p := range opt.params {
		p.ZeroGrad()
	}
}

// GetLR returns the current learning rate (for logging).
func (opt *AdamW) GetLR() float32 {
	return opt.currentLR()
}
