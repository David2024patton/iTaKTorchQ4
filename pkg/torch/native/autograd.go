// autograd.go implements reverse-mode automatic differentiation for training.
//
// WHY: Training requires computing gradients of the loss with respect to every
// trainable parameter. Instead of hand-deriving backward formulas for each
// combination of operations, autograd builds a computation graph during the
// forward pass and automatically propagates gradients backward.
//
// HOW: Each operation (MatMul, Add, SiLU, etc.) creates a GradTensor that
// stores a backward function. Calling Backward() on the loss tensor walks
// the graph in reverse topological order, calling each backward function
// to accumulate gradients into the .Grad field of input tensors.
//
// DESIGN: Inspired by PyTorch's autograd but simplified for LLM fine-tuning.
// Only supports the operations needed for transformer forward/backward.
package native

import (
	"math"
)

// GradTensor wraps a Tensor with gradient tracking for automatic differentiation.
type GradTensor struct {
	Data         []float32   // Forward pass values
	Grad         []float32   // Accumulated gradients (same shape as Data)
	Shape        []int       // Tensor dimensions
	RequiresGrad bool        // If true, gradients are computed for this tensor
	GradFn       *GradFn     // Backward function (nil for leaf tensors)
	children     []*GradTensor // Inputs that fed into this tensor
	name         string      // Debug name
}

// GradFn stores the backward function and references to input tensors.
type GradFn struct {
	backward func(outGrad []float32) // Propagates gradients to inputs
	name     string                  // Operation name for debugging
}

// NewGradTensor creates a gradient-tracking tensor from raw data.
func NewGradTensor(data []float32, shape []int, requiresGrad bool) *GradTensor {
	gt := &GradTensor{
		Data:         data,
		Shape:        shape,
		RequiresGrad: requiresGrad,
	}
	if requiresGrad {
		gt.Grad = make([]float32, len(data))
	}
	return gt
}

// NewGradTensorFrom wraps an existing Tensor.
func NewGradTensorFrom(t *Tensor, requiresGrad bool) *GradTensor {
	return NewGradTensor(t.Data, t.Shape, requiresGrad)
}

// Backward computes gradients for all tensors in the computation graph.
// Call this on the loss tensor (should be a scalar or 1-element tensor).
func (gt *GradTensor) Backward() {
	// Seed the output gradient with 1.0 (dL/dL = 1).
	if gt.Grad == nil {
		gt.Grad = make([]float32, len(gt.Data))
	}
	for i := range gt.Grad {
		gt.Grad[i] = 1.0
	}

	// Topological sort: collect all nodes in reverse order.
	visited := make(map[*GradTensor]bool)
	var order []*GradTensor
	var topo func(node *GradTensor)
	topo = func(node *GradTensor) {
		if visited[node] {
			return
		}
		visited[node] = true
		for _, child := range node.children {
			topo(child)
		}
		order = append(order, node)
	}
	topo(gt)

	// Walk in reverse topological order (output -> inputs).
	for i := len(order) - 1; i >= 0; i-- {
		node := order[i]
		if node.GradFn != nil && node.Grad != nil {
			node.GradFn.backward(node.Grad)
		}
	}
}

// ZeroGrad resets all gradients to zero.
func (gt *GradTensor) ZeroGrad() {
	if gt.Grad != nil {
		for i := range gt.Grad {
			gt.Grad[i] = 0
		}
	}
}

// ---------- Differentiable Operations ----------

// GradMatMul computes C = A @ B with gradient tracking.
// A: [M, K], B: [K, N] -> C: [M, N]
func GradMatMul(a, b *GradTensor) *GradTensor {
	M := a.Shape[0]
	K := a.Shape[1]
	N := b.Shape[1]

	// Forward: C[m,n] = sum_k A[m,k] * B[k,n]
	outData := make([]float32, M*N)
	for m := 0; m < M; m++ {
		for n := 0; n < N; n++ {
			var sum float32
			for k := 0; k < K; k++ {
				sum += a.Data[m*K+k] * b.Data[k*N+n]
			}
			outData[m*N+n] = sum
		}
	}

	out := &GradTensor{
		Data:     outData,
		Shape:    []int{M, N},
		Grad:     make([]float32, M*N),
		children: []*GradTensor{a, b},
	}

	out.GradFn = &GradFn{
		name: "MatMul",
		backward: func(outGrad []float32) {
			// dL/dA = outGrad @ B^T
			if a.RequiresGrad {
				for m := 0; m < M; m++ {
					for k := 0; k < K; k++ {
						var sum float32
						for n := 0; n < N; n++ {
							sum += outGrad[m*N+n] * b.Data[k*N+n]
						}
						a.Grad[m*K+k] += sum
					}
				}
			}
			// dL/dB = A^T @ outGrad
			if b.RequiresGrad {
				for k := 0; k < K; k++ {
					for n := 0; n < N; n++ {
						var sum float32
						for m := 0; m < M; m++ {
							sum += a.Data[m*K+k] * outGrad[m*N+n]
						}
						b.Grad[k*N+n] += sum
					}
				}
			}
		},
	}

	return out
}

// GradMatVecMul computes y = A @ x with gradient tracking.
// A: [M, K], x: [K] -> y: [M]
func GradMatVecMul(a, x *GradTensor) *GradTensor {
	M := a.Shape[0]
	K := a.Shape[1]

	outData := make([]float32, M)
	for m := 0; m < M; m++ {
		outData[m] = DotProduct(a.Data[m*K:(m+1)*K], x.Data)
	}

	out := &GradTensor{
		Data:     outData,
		Shape:    []int{M},
		Grad:     make([]float32, M),
		children: []*GradTensor{a, x},
	}

	out.GradFn = &GradFn{
		name: "MatVecMul",
		backward: func(outGrad []float32) {
			// dL/dA[m,k] = outGrad[m] * x[k]
			if a.RequiresGrad {
				for m := 0; m < M; m++ {
					for k := 0; k < K; k++ {
						a.Grad[m*K+k] += outGrad[m] * x.Data[k]
					}
				}
			}
			// dL/dx[k] = sum_m outGrad[m] * A[m,k]
			if x.RequiresGrad {
				for k := 0; k < K; k++ {
					var sum float32
					for m := 0; m < M; m++ {
						sum += outGrad[m] * a.Data[m*K+k]
					}
					x.Grad[k] += sum
				}
			}
		},
	}

	return out
}

// GradAdd computes c = a + b element-wise with gradient tracking.
func GradAdd(a, b *GradTensor) *GradTensor {
	outData := make([]float32, len(a.Data))
	for i := range outData {
		outData[i] = a.Data[i] + b.Data[i]
	}

	out := &GradTensor{
		Data:     outData,
		Shape:    append([]int(nil), a.Shape...),
		Grad:     make([]float32, len(outData)),
		children: []*GradTensor{a, b},
	}

	out.GradFn = &GradFn{
		name: "Add",
		backward: func(outGrad []float32) {
			// dL/da = outGrad, dL/db = outGrad (identity)
			if a.RequiresGrad {
				for i := range outGrad {
					a.Grad[i] += outGrad[i]
				}
			}
			if b.RequiresGrad {
				for i := range outGrad {
					b.Grad[i] += outGrad[i]
				}
			}
		},
	}

	return out
}

// GradSiLU computes SiLU(x) = x * sigmoid(x) with gradient tracking.
func GradSiLU(x *GradTensor) *GradTensor {
	outData := make([]float32, len(x.Data))
	sigmas := make([]float32, len(x.Data)) // Cache sigmoid values for backward
	for i, v := range x.Data {
		sig := float32(1.0 / (1.0 + math.Exp(-float64(v))))
		sigmas[i] = sig
		outData[i] = v * sig
	}

	out := &GradTensor{
		Data:     outData,
		Shape:    append([]int(nil), x.Shape...),
		Grad:     make([]float32, len(outData)),
		children: []*GradTensor{x},
	}

	out.GradFn = &GradFn{
		name: "SiLU",
		backward: func(outGrad []float32) {
			if !x.RequiresGrad {
				return
			}
			// d/dx[x * sig(x)] = sig(x) + x * sig(x) * (1 - sig(x))
			//                   = sig(x) * (1 + x * (1 - sig(x)))
			for i := range outGrad {
				sig := sigmas[i]
				grad := sig * (1 + x.Data[i]*(1-sig))
				x.Grad[i] += outGrad[i] * grad
			}
		},
	}

	return out
}

// GradMul computes c = a * b element-wise with gradient tracking.
func GradMul(a, b *GradTensor) *GradTensor {
	outData := make([]float32, len(a.Data))
	for i := range outData {
		outData[i] = a.Data[i] * b.Data[i]
	}

	out := &GradTensor{
		Data:     outData,
		Shape:    append([]int(nil), a.Shape...),
		Grad:     make([]float32, len(outData)),
		children: []*GradTensor{a, b},
	}

	out.GradFn = &GradFn{
		name: "Mul",
		backward: func(outGrad []float32) {
			if a.RequiresGrad {
				for i := range outGrad {
					a.Grad[i] += outGrad[i] * b.Data[i]
				}
			}
			if b.RequiresGrad {
				for i := range outGrad {
					b.Grad[i] += outGrad[i] * a.Data[i]
				}
			}
		},
	}

	return out
}

// GradRMSNorm computes RMS normalization with gradient tracking.
// x: [dim], weight: [dim] -> out: [dim]
func GradRMSNorm(x, weight *GradTensor, eps float32) *GradTensor {
	dim := len(x.Data)
	outData := make([]float32, dim)

	// Forward: compute RMS and normalize.
	var sumSq float64
	for _, v := range x.Data {
		sumSq += float64(v) * float64(v)
	}
	rms := float32(math.Sqrt(sumSq/float64(dim) + float64(eps)))
	invRms := float32(1.0) / rms

	for i := 0; i < dim; i++ {
		outData[i] = x.Data[i] * invRms * weight.Data[i]
	}

	out := &GradTensor{
		Data:     outData,
		Shape:    []int{dim},
		Grad:     make([]float32, dim),
		children: []*GradTensor{x, weight},
	}

	out.GradFn = &GradFn{
		name: "RMSNorm",
		backward: func(outGrad []float32) {
			// Gradient of RMSNorm is complex but standard.
			// d/dx_i = (w_i / rms) * (outGrad_i - x_i * sum_j(outGrad_j * w_j * x_j) / (dim * rms^2))
			if x.RequiresGrad {
				var dotProd float64
				for j := 0; j < dim; j++ {
					dotProd += float64(outGrad[j]) * float64(weight.Data[j]) * float64(x.Data[j])
				}
				coeff := float32(dotProd / (float64(dim) * float64(rms) * float64(rms)))

				for i := 0; i < dim; i++ {
					x.Grad[i] += (weight.Data[i]*outGrad[i] - x.Data[i]*coeff) * invRms
				}
			}
			// d/dw_i = x_i / rms * outGrad_i
			if weight.RequiresGrad {
				for i := 0; i < dim; i++ {
					weight.Grad[i] += x.Data[i] * invRms * outGrad[i]
				}
			}
		},
	}

	return out
}

// GradCrossEntropy computes cross-entropy loss with gradient tracking.
// logits: [vocabSize], target: index -> scalar loss
func GradCrossEntropy(logits *GradTensor, target int) *GradTensor {
	n := len(logits.Data)

	// Forward: stable softmax + cross entropy.
	maxLogit := logits.Data[0]
	for _, l := range logits.Data[1:] {
		if l > maxLogit {
			maxLogit = l
		}
	}

	probs := make([]float32, n)
	var sumExp float64
	for i, l := range logits.Data {
		probs[i] = float32(math.Exp(float64(l - maxLogit)))
		sumExp += float64(probs[i])
	}
	for i := range probs {
		probs[i] /= float32(sumExp)
	}

	loss := float32(-math.Log(float64(probs[target]) + 1e-10))

	out := &GradTensor{
		Data:     []float32{loss},
		Shape:    []int{1},
		Grad:     make([]float32, 1),
		children: []*GradTensor{logits},
	}

	out.GradFn = &GradFn{
		name: "CrossEntropy",
		backward: func(outGrad []float32) {
			if !logits.RequiresGrad {
				return
			}
			// d(CE)/d(logits_i) = probs[i] - 1{i == target}
			scale := outGrad[0]
			for i := 0; i < n; i++ {
				grad := probs[i]
				if i == target {
					grad -= 1.0
				}
				logits.Grad[i] += scale * grad
			}
		},
	}

	return out
}

// GradSoftmax computes softmax with gradient tracking.
// Used within attention mechanism.
func GradSoftmax(x *GradTensor) *GradTensor {
	n := len(x.Data)
	outData := make([]float32, n)

	maxVal := x.Data[0]
	for _, v := range x.Data[1:] {
		if v > maxVal {
			maxVal = v
		}
	}

	var sumExp float64
	for i, v := range x.Data {
		outData[i] = float32(math.Exp(float64(v - maxVal)))
		sumExp += float64(outData[i])
	}
	for i := range outData {
		outData[i] /= float32(sumExp)
	}

	out := &GradTensor{
		Data:     outData,
		Shape:    append([]int(nil), x.Shape...),
		Grad:     make([]float32, n),
		children: []*GradTensor{x},
	}

	out.GradFn = &GradFn{
		name: "Softmax",
		backward: func(outGrad []float32) {
			if !x.RequiresGrad {
				return
			}
			// d(softmax_i)/d(x_j) = softmax_i * (delta_ij - softmax_j)
			var dotProd float32
			for i := range outData {
				dotProd += outGrad[i] * outData[i]
			}
			for i := range x.Grad {
				x.Grad[i] += outData[i] * (outGrad[i] - dotProd)
			}
		},
	}

	return out
}
