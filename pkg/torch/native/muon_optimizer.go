// muon_optimizer.go implements the Muon (Momentum Orthogonalization) optimizer.
//
// WHAT: Muon is the optimizer used by Moonshot AI's Kimi K2 model. It
// combines momentum with periodic orthogonalization of the update direction,
// preventing the optimizer from collapsing into low-rank subspaces.
//
// HOW:
//   1. Standard momentum update: m = beta * m + (1-beta) * grad
//   2. Periodically orthogonalize the momentum via Newton-Schulz iteration:
//      M = M * (I + 0.5*(I - M^T*M))  repeated K times
//   3. This keeps the update directions well-conditioned and prevents
//      mode collapse during training.
//
// WHY: Adam's second moment (v) can cause the optimizer to get "stuck" in
// valleys of the loss landscape. Muon's orthogonalization ensures diverse
// update directions, leading to faster convergence and better generalization.
// Kimi K2 showed 10-20% faster convergence vs Adam on MoE architectures.
//
// REFERENCE: Moonshot AI, Jordan et al. "Muon: An optimizer for hidden layers"
package native

import (
	"math"
)

// MuonConfig configures the Muon optimizer.
type MuonConfig struct {
	LR              float32 // Learning rate (default: 0.02)
	Momentum        float32 // Momentum coefficient (default: 0.95)
	NSIterations    int     // Newton-Schulz iterations for orthogonalization (default: 5)
	OrthogInterval  int     // Steps between orthogonalization (default: 1)
	WeightDecay     float32 // Decoupled weight decay (default: 0.0)
}

// DefaultMuonConfig returns Kimi K2-style settings.
func DefaultMuonConfig() MuonConfig {
	return MuonConfig{
		LR:             0.02,
		Momentum:       0.95,
		NSIterations:   5,
		OrthogInterval: 1,
		WeightDecay:    0.0,
	}
}

// MuonState holds per-parameter optimizer state.
type MuonState struct {
	M    []float32 // Momentum buffer
	Step int
}

// MuonOptimizer implements the Muon optimizer.
type MuonOptimizer struct {
	config MuonConfig
	states map[string]*MuonState
}

// NewMuonOptimizer creates a Muon optimizer.
func NewMuonOptimizer(config MuonConfig) *MuonOptimizer {
	return &MuonOptimizer{
		config: config,
		states: make(map[string]*MuonState),
	}
}

// Step performs one Muon update for a 2D weight matrix.
// For 1D params (bias, layernorm), falls back to standard momentum SGD.
func (opt *MuonOptimizer) Step(name string, params, grad []float32, rows, cols int) {
	state, ok := opt.states[name]
	if !ok {
		state = &MuonState{
			M: make([]float32, len(grad)),
		}
		opt.states[name] = state
	}
	state.Step++

	// Momentum update: m = beta * m + (1 - beta) * grad.
	beta := opt.config.Momentum
	for i := range grad {
		state.M[i] = beta*state.M[i] + (1-beta)*grad[i]
	}

	// For 2D weight matrices, apply Newton-Schulz orthogonalization.
	var update []float32
	if rows > 1 && cols > 1 && state.Step%opt.config.OrthogInterval == 0 {
		update = opt.newtonSchulzOrthog(state.M, rows, cols)
	} else {
		update = make([]float32, len(state.M))
		copy(update, state.M)
	}

	// Apply update with optional weight decay.
	lr := opt.config.LR
	wd := opt.config.WeightDecay
	for i := range params {
		if wd > 0 {
			params[i] -= lr * wd * params[i] // Decoupled weight decay.
		}
		params[i] -= lr * update[i]
	}
}

// newtonSchulzOrthog applies Newton-Schulz iteration to orthogonalize
// the momentum matrix. This keeps update directions well-conditioned.
//
// The iteration converges to the nearest orthogonal matrix:
//   X_{k+1} = X_k * (I + 0.5 * (I - X_k^T * X_k))
//
// For rectangular matrices where rows != cols, we use the economy form.
func (opt *MuonOptimizer) newtonSchulzOrthog(m []float32, rows, cols int) []float32 {
	// Work with the smaller dimension for efficiency.
	// If rows < cols, transpose -> orthog -> transpose back.
	transposed := rows < cols
	if transposed {
		m = muonTranspose(m, rows, cols)
		rows, cols = cols, rows
	}

	// Normalize: scale so largest singular value is ~1.
	norm := muonFrobenius(m, rows, cols)
	if norm < 1e-8 {
		return m
	}
	scale := math.Sqrt(float64(cols)) / norm

	x := make([]float32, rows*cols)
	for i := range m {
		x[i] = m[i] * float32(scale)
	}

	// Newton-Schulz iterations.
	for iter := 0; iter < opt.config.NSIterations; iter++ {
		// Compute X^T * X: [cols, cols].
		xtx := muonMatMulTransA(x, x, rows, cols, cols)

		// Compute I - X^T * X.
		for i := 0; i < cols; i++ {
			for j := 0; j < cols; j++ {
				xtx[i*cols+j] = -xtx[i*cols+j]
				if i == j {
					xtx[i*cols+j] += 1.0
				}
			}
		}

		// Scale by 0.5 and add I: factor = I + 0.5 * (I - X^T X).
		for i := 0; i < cols; i++ {
			for j := 0; j < cols; j++ {
				xtx[i*cols+j] *= 0.5
				if i == j {
					xtx[i*cols+j] += 1.0
				}
			}
		}

		// X_new = X * factor.
		x = muonMatMul(x, xtx, rows, cols, cols)
	}

	// Undo the normalization scaling.
	invScale := float32(1.0 / scale)
	for i := range x {
		x[i] *= invScale
	}

	if transposed {
		x = muonTranspose(x, rows, cols)
	}

	return x
}

// matrixFrobenius computes the Frobenius norm of a matrix.
func muonFrobenius(m []float32, rows, cols int) float64 {
	var sum float64
	for i := 0; i < rows*cols; i++ {
		sum += float64(m[i]) * float64(m[i])
	}
	return math.Sqrt(sum)
}

// transposeMatrix transposes [rows, cols] -> [cols, rows].
func muonTranspose(m []float32, rows, cols int) []float32 {
	t := make([]float32, rows*cols)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			t[j*rows+i] = m[i*cols+j]
		}
	}
	return t
}

// matMulTransA computes A^T * B where A is [rows, colsA], B is [rows, colsB].
// Result: [colsA, colsB].
func muonMatMulTransA(a, b []float32, rows, colsA, colsB int) []float32 {
	result := make([]float32, colsA*colsB)
	for i := 0; i < colsA; i++ {
		for j := 0; j < colsB; j++ {
			var sum float64
			for k := 0; k < rows; k++ {
				sum += float64(a[k*colsA+i]) * float64(b[k*colsB+j])
			}
			result[i*colsB+j] = float32(sum)
		}
	}
	return result
}

// matMul computes A * B where A is [rowsA, colsA], B is [colsA, colsB].
// Result: [rowsA, colsB].
func muonMatMul(a, b []float32, rowsA, colsA, colsB int) []float32 {
	result := make([]float32, rowsA*colsB)
	for i := 0; i < rowsA; i++ {
		for j := 0; j < colsB; j++ {
			var sum float64
			for k := 0; k < colsA; k++ {
				sum += float64(a[i*colsA+k]) * float64(b[k*colsB+j])
			}
			result[i*colsB+j] = float32(sum)
		}
	}
	return result
}

// Stats returns Muon optimizer metrics.
func (opt *MuonOptimizer) Stats() map[string]interface{} {
	var totalParams int64
	for _, s := range opt.states {
		totalParams += int64(len(s.M))
	}
	return map[string]interface{}{
		"params_tracked":  len(opt.states),
		"total_elements":  totalParams,
		"memory_bytes":    totalParams * 4, // Momentum only (no second moment).
		"ns_iterations":   opt.config.NSIterations,
		"momentum":        opt.config.Momentum,
	}
}
