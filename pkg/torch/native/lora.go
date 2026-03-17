// lora.go implements Low-Rank Adaptation (LoRA) for parameter-efficient fine-tuning.
//
// WHY: Full fine-tuning updates billions of parameters. LoRA freezes the base
// model and injects small trainable matrices: W' = W + A @ B where A is [dim, rank]
// and B is [rank, dim]. With rank 4-16, this trains ~0.1% of parameters while
// achieving 90-95% of full fine-tuning quality.
//
// HOW: For each target weight matrix (usually WQ, WK, WV, WO in attention),
// we create a LoRA adapter with two small matrices A and B. During the forward
// pass, the effective weight is W + alpha/rank * A @ B. During training, only
// A and B receive gradients.
//
// MERGE: After training, LoRA weights can be merged into the base model:
// W_merged = W + alpha/rank * A @ B. This produces a single weight matrix
// with zero inference overhead.
package native

import (
	"encoding/binary"
	"fmt"
	"math"
	"math/rand"
	"os"
)

// LoRAConfig controls LoRA adapter creation.
type LoRAConfig struct {
	Rank    int     // Low-rank dimension (4, 8, 16 are common)
	Alpha   float32 // Scaling factor (typically = rank)
	Dropout float32 // Dropout rate during training (0 = disabled)
}

// DefaultLoRAConfig returns standard LoRA hyperparameters.
func DefaultLoRAConfig() LoRAConfig {
	return LoRAConfig{
		Rank:    8,
		Alpha:   8.0,
		Dropout: 0.05,
	}
}

// LoRAAdapter holds the A and B matrices for one weight tensor.
type LoRAAdapter struct {
	A         *GradTensor // [inputDim, rank] - initialized with small random values
	B         *GradTensor // [rank, outputDim] - initialized to zero
	Scale     float32     // alpha / rank
	InputDim  int
	OutputDim int
	Rank      int
	name      string // e.g., "layer.0.WQ"
}

// NewLoRAAdapter creates a LoRA adapter for a weight matrix of shape [outputDim, inputDim].
func NewLoRAAdapter(inputDim, outputDim, rank int, alpha float32, name string) *LoRAAdapter {
	scale := alpha / float32(rank)

	// A: Kaiming uniform initialization (standard for LoRA).
	aData := make([]float32, inputDim*rank)
	stddev := float32(1.0 / math.Sqrt(float64(inputDim)))
	for i := range aData {
		aData[i] = (rand.Float32()*2 - 1) * stddev
	}

	// B: Initialize to zero (LoRA starts as identity at init).
	bData := make([]float32, rank*outputDim)

	return &LoRAAdapter{
		A:         NewGradTensor(aData, []int{inputDim, rank}, true),
		B:         NewGradTensor(bData, []int{rank, outputDim}, true),
		Scale:     scale,
		InputDim:  inputDim,
		OutputDim: outputDim,
		Rank:      rank,
		name:      name,
	}
}

// Forward computes the LoRA contribution: scale * (x @ A @ B).
// This is added to the base model's output: y = x@W + scale * x@A@B
func (la *LoRAAdapter) Forward(x *GradTensor) *GradTensor {
	// x @ A -> [seqLen, rank]
	xA := GradMatMul(x, la.A)

	// (x @ A) @ B -> [seqLen, outputDim]
	xAB := GradMatMul(xA, la.B)

	// Scale by alpha/rank.
	if la.Scale != 1.0 {
		for i := range xAB.Data {
			xAB.Data[i] *= la.Scale
		}
	}

	return xAB
}

// GetParams returns all trainable parameters in this adapter.
func (la *LoRAAdapter) GetParams() []*GradTensor {
	return []*GradTensor{la.A, la.B}
}

// NumParams returns the total number of trainable parameters.
func (la *LoRAAdapter) NumParams() int {
	return la.InputDim*la.Rank + la.Rank*la.OutputDim
}

// Merge computes the merged weight: delta_W = scale * A @ B.
// Returns a Tensor that can be added to the base weight.
func (la *LoRAAdapter) Merge() *Tensor {
	// A @ B -> [inputDim, outputDim]
	merged := NewTensor([]int{la.InputDim, la.OutputDim})
	for i := 0; i < la.InputDim; i++ {
		for j := 0; j < la.OutputDim; j++ {
			var sum float32
			for r := 0; r < la.Rank; r++ {
				sum += la.A.Data[i*la.Rank+r] * la.B.Data[r*la.OutputDim+j]
			}
			merged.Data[i*la.OutputDim+j] = sum * la.Scale
		}
	}
	return merged
}

// ---------- LoRA Manager ----------

// LoRAManager manages all LoRA adapters for a model.
type LoRAManager struct {
	adapters map[string]*LoRAAdapter // key: "layer.{idx}.{weight_name}"
	config   LoRAConfig
}

// NewLoRAManager creates a LoRA manager and attaches adapters to the model.
func NewLoRAManager(engine *NativeEngine, config LoRAConfig) *LoRAManager {
	mgr := &LoRAManager{
		adapters: make(map[string]*LoRAAdapter),
		config:   config,
	}

	totalParams := 0

	for i, layer := range engine.layers {
		// Attach LoRA to attention weights: WQ, WK, WV, WO.
		targets := map[string]*Tensor{
			"WQ": layer.WQ,
			"WK": layer.WK,
			"WV": layer.WV,
			"WO": layer.WO,
		}

		for name, w := range targets {
			if w == nil || len(w.Shape) < 2 {
				continue
			}
			key := fmt.Sprintf("layer.%d.%s", i, name)
			adapter := NewLoRAAdapter(w.Shape[1], w.Shape[0], config.Rank, config.Alpha, key)
			mgr.adapters[key] = adapter
			totalParams += adapter.NumParams()
		}
	}

	fmt.Printf("[LoRA] Created %d adapters (rank=%d, alpha=%.0f)\n",
		len(mgr.adapters), config.Rank, config.Alpha)
	fmt.Printf("[LoRA] Trainable parameters: %d (%.2f KB)\n",
		totalParams, float64(totalParams*4)/1024)

	return mgr
}

// GetAdapter returns the LoRA adapter for a specific layer/weight.
func (mgr *LoRAManager) GetAdapter(layerIdx int, weightName string) *LoRAAdapter {
	key := fmt.Sprintf("layer.%d.%s", layerIdx, weightName)
	return mgr.adapters[key]
}

// AllParams returns all trainable LoRA parameters (for the optimizer).
func (mgr *LoRAManager) AllParams() []*GradTensor {
	var params []*GradTensor
	for _, adapter := range mgr.adapters {
		params = append(params, adapter.GetParams()...)
	}
	return params
}

// TotalParams returns the total number of trainable parameters across all adapters.
func (mgr *LoRAManager) TotalParams() int {
	total := 0
	for _, adapter := range mgr.adapters {
		total += adapter.NumParams()
	}
	return total
}

// MergeAll permanently merges all LoRA weights into the base model.
// After merging, inference uses the combined weights with zero overhead.
func (mgr *LoRAManager) MergeAll(engine *NativeEngine) {
	for i := range engine.layers {
		targets := map[string]*Tensor{
			"WQ": engine.layers[i].WQ,
			"WK": engine.layers[i].WK,
			"WV": engine.layers[i].WV,
			"WO": engine.layers[i].WO,
		}

		for name, w := range targets {
			adapter := mgr.GetAdapter(i, name)
			if adapter == nil || w == nil {
				continue
			}
			delta := adapter.Merge()
			for j := range w.Data {
				if j < len(delta.Data) {
					w.Data[j] += delta.Data[j]
				}
			}
		}
	}
	fmt.Println("[LoRA] All adapters merged into base model weights")
}

// ---------- Save / Load ----------

// Save writes all LoRA adapter weights to a binary file.
func (mgr *LoRAManager) Save(path string) error {
	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("create %s: %w", path, err)
	}
	defer f.Close()

	numAdapters := uint32(len(mgr.adapters))
	binary.Write(f, binary.LittleEndian, numAdapters)
	binary.Write(f, binary.LittleEndian, uint32(mgr.config.Rank))

	for key, adapter := range mgr.adapters {
		// Write key length + key string.
		keyBytes := []byte(key)
		binary.Write(f, binary.LittleEndian, uint32(len(keyBytes)))
		f.Write(keyBytes)

		// Write dimensions.
		binary.Write(f, binary.LittleEndian, uint32(adapter.InputDim))
		binary.Write(f, binary.LittleEndian, uint32(adapter.OutputDim))

		// Write A and B data.
		binary.Write(f, binary.LittleEndian, adapter.A.Data)
		binary.Write(f, binary.LittleEndian, adapter.B.Data)
	}

	fmt.Printf("[LoRA] Saved %d adapters to %s\n", numAdapters, path)
	return nil
}

// Load reads LoRA adapter weights from a binary file.
func (mgr *LoRAManager) Load(path string) error {
	f, err := os.Open(path)
	if err != nil {
		return fmt.Errorf("open %s: %w", path, err)
	}
	defer f.Close()

	var numAdapters, rank uint32
	binary.Read(f, binary.LittleEndian, &numAdapters)
	binary.Read(f, binary.LittleEndian, &rank)

	for i := uint32(0); i < numAdapters; i++ {
		var keyLen uint32
		binary.Read(f, binary.LittleEndian, &keyLen)
		keyBytes := make([]byte, keyLen)
		f.Read(keyBytes)
		key := string(keyBytes)

		var inputDim, outputDim uint32
		binary.Read(f, binary.LittleEndian, &inputDim)
		binary.Read(f, binary.LittleEndian, &outputDim)

		adapter, exists := mgr.adapters[key]
		if !exists {
			// Create adapter if it doesn't exist.
			adapter = NewLoRAAdapter(int(inputDim), int(outputDim), int(rank), mgr.config.Alpha, key)
			mgr.adapters[key] = adapter
		}

		binary.Read(f, binary.LittleEndian, adapter.A.Data)
		binary.Read(f, binary.LittleEndian, adapter.B.Data)
	}

	fmt.Printf("[LoRA] Loaded %d adapters from %s\n", numAdapters, path)
	return nil
}
