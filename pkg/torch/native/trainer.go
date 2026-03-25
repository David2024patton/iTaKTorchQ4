// trainer.go orchestrates the training loop for fine-tuning LLMs.
//
// WHY: This ties together the autograd, optimizer, LoRA, AttnRes, and data
// pipeline into a complete training system. One call to Train() runs the
// entire fine-tuning process with logging, checkpointing, and metrics.
//
// MODES:
//   - LoRA fine-tuning: trains low-rank adapters on attention weights
//   - AttnRes fine-tuning: trains depth-wise attention query vectors
//   - Combined: trains both LoRA + AttnRes simultaneously
//
// USAGE:
//   trainer := NewTrainer(engine, TrainerConfig{...})
//   trainer.TrainOnFile("data.txt")
//   trainer.Save("checkpoint.bin")
package native

import (
	"encoding/binary"
	"fmt"
	"math"
	"math/rand"
	"os"
	"time"
)

// TrainerConfig holds all training hyperparameters.
type TrainerConfig struct {
	// Training parameters.
	Epochs    int     // Number of passes over the data
	BatchSize int     // Sequences per batch
	SeqLen    int     // Tokens per sequence
	LR        float32 // Learning rate

	// What to train.
	EnableLoRA    bool // Train LoRA adapter weights
	EnableAttnRes bool // Train AttnRes pseudo-query vectors
	LoRARank      int  // LoRA rank (default: 8)

	// Checkpointing.
	CheckpointDir  string // Directory for saving checkpoints
	CheckpointFreq int    // Save every N steps (0 = only at end)

	// Logging.
	LogFreq int // Print metrics every N steps
}

// DefaultTrainerConfig returns sensible defaults for LLM fine-tuning.
func DefaultTrainerConfig() TrainerConfig {
	return TrainerConfig{
		Epochs:        3,
		BatchSize:     1, // Single sequence at a time (memory efficient)
		SeqLen:        128,
		LR:            1e-4,
		EnableLoRA:    true,
		EnableAttnRes: true,
		LoRARank:      8,
		LogFreq:       10,
	}
}

// Trainer manages the full training loop.
type Trainer struct {
	engine *NativeEngine
	config TrainerConfig

	// Training components.
	lora      *LoRAManager
	optimizer *AdamW
	params    []*GradTensor // All trainable parameters

	// Progress tracking.
	Progress  *TrainingProgress

	// Metrics.
	globalStep    int
	totalTokens   int64
	losses        []float32
	startTime     time.Time
}

// NewTrainer creates a trainer and initializes all components.
func NewTrainer(engine *NativeEngine, config TrainerConfig) *Trainer {
	t := &Trainer{
		engine: engine,
		config: config,
	}

	var allParams []*GradTensor

	// Set up LoRA adapters.
	if config.EnableLoRA {
		loraConfig := DefaultLoRAConfig()
		if config.LoRARank > 0 {
			loraConfig.Rank = config.LoRARank
			loraConfig.Alpha = float32(config.LoRARank)
		}
		t.lora = NewLoRAManager(engine, loraConfig)
		allParams = append(allParams, t.lora.AllParams()...)
	}

	// Set up AttnRes query vectors as trainable parameters.
	if config.EnableAttnRes {
		if !engine.AttnResConfig.Enabled {
			engine.EnableAttnRes()
		}
		for i := range engine.layers {
			attnQ := NewGradTensorFrom(engine.layers[i].AttnResQuery, true)
			ffnQ := NewGradTensorFrom(engine.layers[i].FFNResQuery, true)
			// Point the GradTensor data at the actual engine tensors
			// so updates are reflected in inference.
			attnQ.Data = engine.layers[i].AttnResQuery.Data
			ffnQ.Data = engine.layers[i].FFNResQuery.Data
			allParams = append(allParams, attnQ, ffnQ)
		}
		fmt.Printf("[Trainer] AttnRes query vectors: %d parameters\n",
			2*len(engine.layers)*engine.hiddenDim)
	}

	t.params = allParams

	// Summary.
	totalParams := 0
	for _, p := range allParams {
		totalParams += len(p.Data)
	}

	baseParams := 0
	for _, layer := range engine.layers {
		if layer.WQ != nil {
			baseParams += len(layer.WQ.Data) * 7 // Q,K,V,O,Gate,Up,Down
		}
	}

	fmt.Printf("[Trainer] Trainable: %d params (%.2f MB)\n",
		totalParams, float64(totalParams*4)/(1024*1024))
	fmt.Printf("[Trainer] Frozen:    %d params (%.2f MB)\n",
		baseParams, float64(baseParams*4)/(1024*1024))
	if baseParams > 0 {
		fmt.Printf("[Trainer] Training %.3f%% of total parameters\n",
			float64(totalParams)/float64(baseParams+totalParams)*100)
	}

	return t
}

// AllParams returns all trainable parameters (for gradient accumulation).
func (t *Trainer) AllParams() []*GradTensor {
	return t.params
}

// TrainOnFile loads a text dataset and runs training.
func (t *Trainer) TrainOnFile(path string) error {
	dataset, err := LoadTextDataset(path, t.config.SeqLen, t.engine.vocabSize)
	if err != nil {
		return fmt.Errorf("load dataset: %w", err)
	}
	return t.TrainOnDataset(dataset)
}

// TrainOnTexts creates a dataset from strings and runs training.
func (t *Trainer) TrainOnTexts(texts []string) error {
	var combined string
	for _, text := range texts {
		combined += text + " "
	}
	dataset := LoadTextString(combined, t.config.SeqLen, t.engine.vocabSize)
	return t.TrainOnDataset(dataset)
}

// TrainOnDataset runs the training loop on a prepared dataset.
func (t *Trainer) TrainOnDataset(dataset *TrainingDataset) error {
	if len(dataset.Sequences) == 0 {
		return fmt.Errorf("empty dataset")
	}

	// Initialize optimizer.
	totalSteps := t.config.Epochs * len(dataset.Sequences) / t.config.BatchSize
	optConfig := DefaultAdamWConfig(totalSteps)
	optConfig.LearningRate = t.config.LR
	t.optimizer = NewAdamW(t.params, optConfig)

	t.startTime = time.Now()
	fmt.Printf("[Trainer] Starting training: %d epochs, %d sequences, %d total steps\n",
		t.config.Epochs, len(dataset.Sequences), totalSteps)
	fmt.Println("[Trainer] ----------------------------------------")

	for epoch := 0; epoch < t.config.Epochs; epoch++ {
		epochStart := time.Now()
		epochLoss := float32(0)
		epochTokens := 0
		batchCount := 0

		iter := NewBatchIterator(dataset, t.config.BatchSize)

		for batch := iter.Next(); batch != nil; batch = iter.Next() {
			// 1. Zero gradients.
			t.optimizer.ZeroGrad()

			// 2. Forward pass + loss for each sequence in batch.
			var batchLoss float32
			for seqIdx := 0; seqIdx < len(batch.Inputs); seqIdx++ {
				input := batch.Inputs[seqIdx]
				targets := batch.Targets[seqIdx]

				// Run forward pass through the model.
				logits := t.engine.forward(input)

				// Compute cross-entropy loss on the last position.
				lastTarget := targets[len(targets)-1]
				loss := crossEntropyLoss(logits, lastTarget)
				batchLoss += loss

				// Compute gradients for trainable parameters.
				// Use the autograd cross-entropy for gradient computation.
				logitGrad := NewGradTensor(logits, []int{len(logits)}, true)
				lossGrad := GradCrossEntropy(logitGrad, lastTarget)
				lossGrad.Backward()

				epochTokens += len(input)
			}

			batchLoss /= float32(len(batch.Inputs))

			// 3. Approximate gradients for LoRA/AttnRes params using loss signal.
			t.approximateGradients(batch, batchLoss)

			// 4. Optimizer step.
			t.optimizer.Step()

			epochLoss += batchLoss
			batchCount++
			t.globalStep++
			t.totalTokens += int64(epochTokens)

			// Logging.
			if t.config.LogFreq > 0 && t.globalStep%t.config.LogFreq == 0 {
				elapsed := time.Since(t.startTime)
				tps := float64(t.totalTokens) / elapsed.Seconds()
				lr := t.optimizer.GetLR()

				// Use progress bar if available.
				if t.Progress != nil {
					t.Progress.SetEpoch(epoch)
					t.Progress.Update(t.globalStep, batchLoss, lr, t.totalTokens)
				} else {
					fmt.Printf("[Step %d] loss=%.4f lr=%.2e tps=%.0f elapsed=%v\n",
						t.globalStep, batchLoss, lr, tps, elapsed.Round(time.Second))
				}
			}

			// Checkpoint.
			if t.config.CheckpointFreq > 0 && t.globalStep%t.config.CheckpointFreq == 0 {
				t.saveCheckpoint()
			}
		}

		avgLoss := epochLoss / float32(batchCount)
		t.losses = append(t.losses, avgLoss)
		epochDur := time.Since(epochStart)
		fmt.Printf("\n[Epoch %d/%d] avg_loss=%.4f tokens=%d duration=%v\n",
			epoch+1, t.config.Epochs, avgLoss, epochTokens, epochDur.Round(time.Second))
	}

	totalDur := time.Since(t.startTime)
	fmt.Println("[Trainer] ----------------------------------------")
	fmt.Printf("[Trainer] Training complete in %v\n", totalDur.Round(time.Second))
	fmt.Printf("[Trainer] Total tokens: %d, steps: %d\n", t.totalTokens, t.globalStep)
	if len(t.losses) > 1 {
		fmt.Printf("[Trainer] Loss: %.4f -> %.4f\n", t.losses[0], t.losses[len(t.losses)-1])
	}

	return nil
}

// approximateGradients uses a perturbation-based approach to estimate gradients
// for the trainable parameters, since the full autograd path through the
// inference engine would require rewriting the entire forward pass.
func (t *Trainer) approximateGradients(batch *Batch, baseLoss float32) {
	eps := float32(1e-4)

	// For each parameter, perturb a random subset of coordinates.
	for _, param := range t.params {
		if !param.RequiresGrad || param.Grad == nil {
			continue
		}

		// Stochastic coordinate descent: update a few random coords per step.
		numCoords := len(param.Data) / 32
		if numCoords < 4 {
			numCoords = 4
		}
		if numCoords > 64 {
			numCoords = 64
		}

		for c := 0; c < numCoords; c++ {
			idx := int(math.Abs(float64(rand.Int()))) % len(param.Data)

			// Perturb.
			original := param.Data[idx]
			param.Data[idx] = original + eps

			// Measure perturbed loss.
			var perturbedLoss float32
			for _, input := range batch.Inputs {
				logits := t.engine.forward(input)
				target := batch.Targets[0][len(batch.Targets[0])-1]
				perturbedLoss += crossEntropyLoss(logits, target)
			}
			perturbedLoss /= float32(len(batch.Inputs))

			// Finite-difference gradient.
			param.Grad[idx] = (perturbedLoss - baseLoss) / eps

			// Restore.
			param.Data[idx] = original
		}
	}
}

// ---------- Checkpointing ----------

// saveCheckpoint saves the current training state.
func (t *Trainer) saveCheckpoint() {
	if t.config.CheckpointDir == "" {
		return
	}

	os.MkdirAll(t.config.CheckpointDir, 0755)
	path := fmt.Sprintf("%s/checkpoint_step%d.bin", t.config.CheckpointDir, t.globalStep)

	f, err := os.Create(path)
	if err != nil {
		fmt.Printf("[Trainer] Checkpoint save failed: %v\n", err)
		return
	}
	defer f.Close()

	// Write step and loss.
	binary.Write(f, binary.LittleEndian, uint32(t.globalStep))
	binary.Write(f, binary.LittleEndian, float32(t.losses[len(t.losses)-1]))

	// Write all parameter data.
	for _, p := range t.params {
		binary.Write(f, binary.LittleEndian, p.Data)
	}

	fmt.Printf("[Trainer] Saved checkpoint: %s\n", path)
}

// Save saves the final trained weights (LoRA + AttnRes).
func (t *Trainer) Save(dir string) error {
	os.MkdirAll(dir, 0755)

	if t.lora != nil {
		if err := t.lora.Save(dir + "/lora_adapters.bin"); err != nil {
			return err
		}
	}

	if t.config.EnableAttnRes {
		attnTrainer := &AttnResTrainer{engine: t.engine}
		if err := attnTrainer.Save(dir + "/attnres_queries.bin"); err != nil {
			return err
		}
	}

	fmt.Printf("[Trainer] All weights saved to %s/\n", dir)
	return nil
}

// Load restores trained weights from a directory.
func (t *Trainer) Load(dir string) error {
	if t.lora != nil {
		if err := t.lora.Load(dir + "/lora_adapters.bin"); err != nil {
			fmt.Printf("[Trainer] No LoRA weights found: %v\n", err)
		}
	}

	if t.config.EnableAttnRes {
		if err := LoadAttnResQueries(t.engine, dir+"/attnres_queries.bin"); err != nil {
			fmt.Printf("[Trainer] No AttnRes weights found: %v\n", err)
		}
	}

	return nil
}

// MergeAndExport merges LoRA weights into the base model permanently.
// After this, the model runs at full speed with no adapter overhead.
func (t *Trainer) MergeAndExport() {
	if t.lora != nil {
		t.lora.MergeAll(t.engine)
	}
}
