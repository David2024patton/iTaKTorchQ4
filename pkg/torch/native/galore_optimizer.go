// galore_optimizer.go implements GaLore (Gradient Low-Rank Projection) for
// memory-efficient LLM training.
//
// WHAT: GaLore (Zhao et al., 2024) projects gradients into a low-rank
// subspace before applying Adam, reducing optimizer state memory by 8x.
// Unlike LoRA (which adds low-rank adapters), GaLore trains the FULL model
// but stores optimizer states in a compressed low-rank form.
//
// HOW:
//   1. Periodically compute SVD of the gradient matrix (every T steps)
//   2. Project gradient into the top-R singular vectors: G_low = P^T @ G
//   3. Run Adam on the low-rank gradient G_low (much smaller state)
//   4. Project back: update = P @ G_low_updated
//
// WHY: For a d x d weight matrix:
//   Standard Adam: 2 * d * d states (m and v) = 8 bytes/param
//   GaLore rank-R: 2 * d * R states          = 8R/d bytes/param
//   With R = d/8: 87.5% memory reduction
//
// GAIN: Train full-rank models with LoRA-level memory. A 7B model that
// normally needs 56GB for Adam states can train in 7GB.
//
// REFERENCE: LlamaFactory, HuggingFace PEFT implement GaLore.
package native

import (
	"math"
)

// GaLoreConfig configures the GaLore optimizer.
type GaLoreConfig struct {
	Rank          int     // Low-rank dimension (default: 128)
	UpdateInterval int   // Steps between projection matrix updates (default: 200)
	Scale         float32 // Gradient scaling factor (default: 1.0)
	LR            float32 // Learning rate
	Beta1         float32 // Adam beta1 (default: 0.9)
	Beta2         float32 // Adam beta2 (default: 0.999)
	Eps           float32 // Adam epsilon (default: 1e-8)
}

// DefaultGaLoreConfig returns standard GaLore settings.
func DefaultGaLoreConfig() GaLoreConfig {
	return GaLoreConfig{
		Rank:           128,
		UpdateInterval: 200,
		Scale:          1.0,
		LR:             1e-4,
		Beta1:          0.9,
		Beta2:          0.999,
		Eps:            1e-8,
	}
}

// GaLoreState holds the low-rank projection and compressed optimizer state.
type GaLoreState struct {
	// Projection matrix P: [dim, rank] (top-R right singular vectors).
	Projector []float32

	// Low-rank Adam states.
	M []float32 // First moment [rank] or [rows*rank]
	V []float32 // Second moment [rank] or [rows*rank]

	// Dimensions.
	Rows int
	Cols int
	Rank int

	// Step counter.
	Step int
}

// GaLoreOptimizer manages GaLore-compressed training for multiple parameters.
type GaLoreOptimizer struct {
	config GaLoreConfig
	states map[string]*GaLoreState
}

// NewGaLoreOptimizer creates a GaLore optimizer.
func NewGaLoreOptimizer(config GaLoreConfig) *GaLoreOptimizer {
	return &GaLoreOptimizer{
		config: config,
		states: make(map[string]*GaLoreState),
	}
}

// Step performs one optimizer update for a 2D weight matrix.
// grad shape: [rows, cols], params shape: [rows, cols].
func (opt *GaLoreOptimizer) Step(name string, params, grad []float32, rows, cols int) {
	rank := opt.config.Rank
	if rank > cols {
		rank = cols
	}
	if rank > rows {
		rank = rows
	}

	// Get or create state.
	state, ok := opt.states[name]
	if !ok {
		state = &GaLoreState{
			Rows: rows,
			Cols: cols,
			Rank: rank,
			M:    make([]float32, rows*rank),
			V:    make([]float32, rows*rank),
		}
		opt.states[name] = state
	}
	state.Step++

	// Update projection matrix periodically via SVD approximation.
	if state.Projector == nil || state.Step%opt.config.UpdateInterval == 0 {
		state.Projector = computeProjection(grad, rows, cols, rank)
	}

	// Project gradient to low-rank space: G_low = G @ P  [rows, rank]
	gLow := projectGradient(grad, state.Projector, rows, cols, rank)

	// Adam update in low-rank space.
	bc1 := 1.0 - math.Pow(float64(opt.config.Beta1), float64(state.Step))
	bc2 := 1.0 - math.Pow(float64(opt.config.Beta2), float64(state.Step))

	for i := range gLow {
		state.M[i] = opt.config.Beta1*state.M[i] + (1-opt.config.Beta1)*gLow[i]
		state.V[i] = opt.config.Beta2*state.V[i] + (1-opt.config.Beta2)*gLow[i]*gLow[i]

		mHat := float64(state.M[i]) / bc1
		vHat := float64(state.V[i]) / bc2
		gLow[i] = float32(mHat / (math.Sqrt(vHat) + float64(opt.config.Eps)))
	}

	// Project back to full space: update = G_low_updated @ P^T  [rows, cols]
	update := unprojectGradient(gLow, state.Projector, rows, cols, rank)

	// Apply update.
	scale := opt.config.LR * opt.config.Scale
	for i := range params {
		params[i] -= scale * update[i]
	}
}

// computeProjection computes an approximate top-R projection via power iteration.
// This approximates the right singular vectors of the gradient matrix.
// Full SVD is O(m*n*min(m,n)); power iteration is O(m*n*R*iters).
func computeProjection(grad []float32, rows, cols, rank int) []float32 {
	// Initialize random projection matrix [cols, rank].
	proj := make([]float32, cols*rank)
	for i := range proj {
		// Simple deterministic initialization based on index.
		proj[i] = float32(math.Sin(float64(i)*0.01)) * 0.1
	}

	// Power iteration: 3 iterations is usually sufficient.
	for iter := 0; iter < 3; iter++ {
		// Compute G^T @ G @ P  (cols x rank).
		// Step 1: Q = G @ P  (rows x rank)
		q := make([]float32, rows*rank)
		for i := 0; i < rows; i++ {
			for r := 0; r < rank; r++ {
				var sum float64
				for j := 0; j < cols; j++ {
					sum += float64(grad[i*cols+j]) * float64(proj[j*rank+r])
				}
				q[i*rank+r] = float32(sum)
			}
		}

		// Step 2: P_new = G^T @ Q  (cols x rank)
		newProj := make([]float32, cols*rank)
		for j := 0; j < cols; j++ {
			for r := 0; r < rank; r++ {
				var sum float64
				for i := 0; i < rows; i++ {
					sum += float64(grad[i*cols+j]) * float64(q[i*rank+r])
				}
				newProj[j*rank+r] = float32(sum)
			}
		}

		// QR orthogonalization (simplified Gram-Schmidt).
		for r := 0; r < rank; r++ {
			// Orthogonalize against previous columns.
			for prev := 0; prev < r; prev++ {
				var dot, normSq float64
				for j := 0; j < cols; j++ {
					dot += float64(newProj[j*rank+r]) * float64(newProj[j*rank+prev])
					normSq += float64(newProj[j*rank+prev]) * float64(newProj[j*rank+prev])
				}
				if normSq > 1e-10 {
					coef := dot / normSq
					for j := 0; j < cols; j++ {
						newProj[j*rank+r] -= float32(coef) * newProj[j*rank+prev]
					}
				}
			}
			// Normalize.
			var norm float64
			for j := 0; j < cols; j++ {
				norm += float64(newProj[j*rank+r]) * float64(newProj[j*rank+r])
			}
			norm = math.Sqrt(norm)
			if norm > 1e-10 {
				invNorm := float32(1.0 / norm)
				for j := 0; j < cols; j++ {
					newProj[j*rank+r] *= invNorm
				}
			}
		}

		proj = newProj
	}

	return proj
}

// projectGradient projects gradient to low-rank: G_low = G @ P  [rows, rank].
func projectGradient(grad, proj []float32, rows, cols, rank int) []float32 {
	result := make([]float32, rows*rank)
	for i := 0; i < rows; i++ {
		for r := 0; r < rank; r++ {
			var sum float64
			for j := 0; j < cols; j++ {
				sum += float64(grad[i*cols+j]) * float64(proj[j*rank+r])
			}
			result[i*rank+r] = float32(sum)
		}
	}
	return result
}

// unprojectGradient projects back: update = G_low @ P^T  [rows, cols].
func unprojectGradient(gLow, proj []float32, rows, cols, rank int) []float32 {
	result := make([]float32, rows*cols)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			var sum float64
			for r := 0; r < rank; r++ {
				sum += float64(gLow[i*rank+r]) * float64(proj[j*rank+r])
			}
			result[i*cols+j] = float32(sum)
		}
	}
	return result
}

// MemoryUsage returns bytes used by GaLore optimizer states.
func (opt *GaLoreOptimizer) MemoryUsage() int64 {
	var total int64
	for _, state := range opt.states {
		total += int64(len(state.M)*4 + len(state.V)*4) // Low-rank M and V.
		total += int64(len(state.Projector) * 4)          // Projection matrix.
	}
	return total
}

// FP32AdamEquivalent returns what standard Adam would cost.
func (opt *GaLoreOptimizer) FP32AdamEquivalent() int64 {
	var total int64
	for _, state := range opt.states {
		total += int64(state.Rows*state.Cols*4*2) // FP32 M + V at full rank.
	}
	return total
}

// Stats returns optimizer metrics.
func (opt *GaLoreOptimizer) Stats() map[string]interface{} {
	return map[string]interface{}{
		"rank":             opt.config.Rank,
		"update_interval":  opt.config.UpdateInterval,
		"params_tracked":   len(opt.states),
		"memory_bytes":     opt.MemoryUsage(),
		"fp32_equivalent":  opt.FP32AdamEquivalent(),
		"compression":      float64(opt.FP32AdamEquivalent()) / float64(opt.MemoryUsage()+1),
	}
}
