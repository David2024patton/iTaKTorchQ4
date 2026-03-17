// attn_res_finetune.go implements lightweight fine-tuning for AttnRes query vectors.
//
// WHY: AttnRes pseudo-query vectors are learned weights. A pre-trained model
// (Llama, Qwen, etc.) was trained with standard residuals, so its query vectors
// need to be trained to learn which layers are useful. This module trains ONLY
// the query vectors while keeping all base model weights frozen.
//
// PARAMETERS TRAINED: 2 * numLayers * hiddenDim floats
//   Example: 32-layer, 4096-dim model = 2 * 32 * 4096 = 262,144 params (~1 MB)
//   vs. total model size of billions of params. That's ~0.003%.
//
// HOW: Uses finite-difference gradient approximation (no backprop needed).
// For each query vector element, we perturb it by epsilon, measure the change
// in cross-entropy loss, and compute the gradient. SGD with momentum updates
// the query vectors.
//
// USAGE:
//   trainer := NewAttnResTrainer(engine, 0.001, 50)
//   trainer.Train(calibrationTexts)
//   trainer.Save("attnres_queries.bin")
//   // Later: trainer.Load("attnres_queries.bin")
package native

import (
	"encoding/binary"
	"fmt"
	"math"
	"math/rand"
	"os"
	"time"
)

// AttnResTrainer fine-tunes only the AttnRes pseudo-query vectors.
type AttnResTrainer struct {
	engine *NativeEngine

	// Hyperparameters.
	LearningRate float32
	Momentum     float32
	Epochs       int
	Epsilon      float32 // Finite-difference step size

	// Momentum buffers (same shape as query vectors).
	attnMomentum [][]float32 // [layer][hiddenDim]
	ffnMomentum  [][]float32 // [layer][hiddenDim]

	// Training metrics.
	losses []float32
}

// NewAttnResTrainer creates a trainer for AttnRes query vectors.
//
// Parameters:
//   - engine: the NativeEngine with loaded model weights
//   - lr: learning rate (0.001 is a good starting point)
//   - epochs: number of passes over the calibration data
func NewAttnResTrainer(engine *NativeEngine, lr float32, epochs int) *AttnResTrainer {
	numLayers := len(engine.layers)
	dim := engine.hiddenDim

	t := &AttnResTrainer{
		engine:       engine,
		LearningRate: lr,
		Momentum:     0.9,
		Epochs:       epochs,
		Epsilon:      1e-4,
		attnMomentum: make([][]float32, numLayers),
		ffnMomentum:  make([][]float32, numLayers),
	}

	for i := 0; i < numLayers; i++ {
		t.attnMomentum[i] = make([]float32, dim)
		t.ffnMomentum[i] = make([]float32, dim)
	}

	// Ensure AttnRes is enabled on the engine.
	if !engine.AttnResConfig.Enabled {
		engine.EnableAttnRes()
	}

	return t
}

// Train runs the fine-tuning loop over calibration texts.
// Each text is tokenized and used for next-token prediction loss.
// Only the pseudo-query vectors are updated; all other weights are frozen.
func (t *AttnResTrainer) Train(texts []string) {
	if len(texts) == 0 {
		fmt.Println("[AttnRes Train] No calibration texts provided")
		return
	}

	totalParams := 2 * len(t.engine.layers) * t.engine.hiddenDim
	fmt.Printf("[AttnRes Train] Training %d query parameters (%.2f KB)\n",
		totalParams, float64(totalParams*4)/1024)
	fmt.Printf("[AttnRes Train] Base model weights: FROZEN\n")
	fmt.Printf("[AttnRes Train] Epochs: %d, LR: %.4f, Momentum: %.2f\n",
		t.Epochs, t.LearningRate, t.Momentum)

	// Tokenize all calibration texts.
	var allTokenSeqs [][]int
	for _, text := range texts {
		tokens := simpleTokenize(text, t.engine.vocabSize)
		if len(tokens) > 256 {
			tokens = tokens[:256] // Cap length for training speed.
		}
		if len(tokens) >= 2 {
			allTokenSeqs = append(allTokenSeqs, tokens)
		}
	}

	if len(allTokenSeqs) == 0 {
		fmt.Println("[AttnRes Train] No valid training sequences")
		return
	}

	startTime := time.Now()

	for epoch := 0; epoch < t.Epochs; epoch++ {
		epochLoss := float32(0)
		epochSamples := 0

		// Shuffle training data each epoch.
		rand.Shuffle(len(allTokenSeqs), func(i, j int) {
			allTokenSeqs[i], allTokenSeqs[j] = allTokenSeqs[j], allTokenSeqs[i]
		})

		for _, tokens := range allTokenSeqs {
			// Compute baseline loss.
			baseLoss := t.computeLoss(tokens)

			// Update each query vector using finite-difference gradients.
			t.updateQueryVectors(tokens, baseLoss)

			epochLoss += baseLoss
			epochSamples++
		}

		avgLoss := epochLoss / float32(epochSamples)
		t.losses = append(t.losses, avgLoss)

		elapsed := time.Since(startTime)
		fmt.Printf("[AttnRes Train] Epoch %d/%d  loss=%.4f  elapsed=%v\n",
			epoch+1, t.Epochs, avgLoss, elapsed.Round(time.Second))
	}

	fmt.Printf("[AttnRes Train] !! Training complete in %v\n", time.Since(startTime).Round(time.Second))
	if len(t.losses) > 1 {
		improvement := (t.losses[0] - t.losses[len(t.losses)-1]) / t.losses[0] * 100
		fmt.Printf("[AttnRes Train] Loss reduction: %.4f -> %.4f (%.1f%% improvement)\n",
			t.losses[0], t.losses[len(t.losses)-1], improvement)
	}
}

// computeLoss runs a forward pass and returns the average cross-entropy loss.
func (t *AttnResTrainer) computeLoss(tokens []int) float32 {
	if len(tokens) < 2 {
		return 0
	}

	var totalLoss float64
	count := 0

	// Teacher forcing: for each prefix, predict the next token.
	// Use a sliding window for efficiency.
	windowSize := 64
	if windowSize > len(tokens) {
		windowSize = len(tokens)
	}

	for i := 1; i < windowSize; i++ {
		prefix := tokens[:i]
		target := tokens[i]

		// Forward pass to get logits.
		logits := t.engine.forward(prefix)

		// Cross-entropy loss: -log(softmax(logits)[target])
		loss := crossEntropyLoss(logits, target)
		if !math.IsNaN(float64(loss)) && !math.IsInf(float64(loss), 0) {
			totalLoss += float64(loss)
			count++
		}
	}

	if count == 0 {
		return 0
	}
	return float32(totalLoss / float64(count))
}

// updateQueryVectors computes finite-difference gradients and applies SGD+momentum.
func (t *AttnResTrainer) updateQueryVectors(tokens []int, baseLoss float32) {
	dim := t.engine.hiddenDim

	// Stochastic coordinate descent: update a random subset each step.
	// Full gradient over all dims would be too slow.
	coordsPerLayer := dim / 16 // Update ~6% of coordinates each step
	if coordsPerLayer < 8 {
		coordsPerLayer = 8
	}

	for layerIdx := range t.engine.layers {
		layer := &t.engine.layers[layerIdx]

		// Update AttnResQuery.
		t.updateSingleQuery(tokens, baseLoss, layer.AttnResQuery.Data,
			t.attnMomentum[layerIdx], coordsPerLayer)

		// Update FFNResQuery.
		t.updateSingleQuery(tokens, baseLoss, layer.FFNResQuery.Data,
			t.ffnMomentum[layerIdx], coordsPerLayer)
	}
}

// updateSingleQuery applies stochastic coordinate descent to one query vector.
func (t *AttnResTrainer) updateSingleQuery(tokens []int, baseLoss float32,
	queryData []float32, momentum []float32, numCoords int) {

	dim := len(queryData)

	for c := 0; c < numCoords; c++ {
		// Pick a random coordinate.
		idx := rand.Intn(dim)

		// Perturb and measure.
		original := queryData[idx]
		queryData[idx] = original + t.Epsilon
		perturbedLoss := t.computeLoss(tokens)

		// Finite-difference gradient: dL/dq = (L(q+eps) - L(q)) / eps
		grad := (perturbedLoss - baseLoss) / t.Epsilon

		// Restore original value.
		queryData[idx] = original

		// SGD with momentum.
		momentum[idx] = t.Momentum*momentum[idx] + (1-t.Momentum)*grad
		queryData[idx] -= t.LearningRate * momentum[idx]
	}
}

// crossEntropyLoss computes -log(softmax(logits)[target]).
func crossEntropyLoss(logits []float32, target int) float32 {
	if target < 0 || target >= len(logits) {
		return 0
	}

	// Numerically stable softmax.
	maxLogit := logits[0]
	for _, l := range logits[1:] {
		if l > maxLogit {
			maxLogit = l
		}
	}

	var sumExp float64
	for _, l := range logits {
		sumExp += math.Exp(float64(l - maxLogit))
	}

	logSoftmax := float64(logits[target]-maxLogit) - math.Log(sumExp)
	return float32(-logSoftmax)
}

// ---------- Save / Load ----------

// Save writes the trained query vectors to a binary file.
// Format: [numLayers uint32][hiddenDim uint32][attnQuery0...][ffnQuery0...][attnQuery1...]...
func (t *AttnResTrainer) Save(path string) error {
	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("create %s: %w", path, err)
	}
	defer f.Close()

	numLayers := uint32(len(t.engine.layers))
	hiddenDim := uint32(t.engine.hiddenDim)

	binary.Write(f, binary.LittleEndian, numLayers)
	binary.Write(f, binary.LittleEndian, hiddenDim)

	for i := range t.engine.layers {
		binary.Write(f, binary.LittleEndian, t.engine.layers[i].AttnResQuery.Data)
		binary.Write(f, binary.LittleEndian, t.engine.layers[i].FFNResQuery.Data)
	}

	totalBytes := 8 + int(numLayers)*int(hiddenDim)*2*4
	fmt.Printf("[AttnRes] Saved query vectors to %s (%.1f KB)\n", path, float64(totalBytes)/1024)
	return nil
}

// Load reads trained query vectors from a binary file.
func (t *AttnResTrainer) Load(path string) error {
	f, err := os.Open(path)
	if err != nil {
		return fmt.Errorf("open %s: %w", path, err)
	}
	defer f.Close()

	var numLayers, hiddenDim uint32
	binary.Read(f, binary.LittleEndian, &numLayers)
	binary.Read(f, binary.LittleEndian, &hiddenDim)

	if int(numLayers) != len(t.engine.layers) {
		return fmt.Errorf("layer count mismatch: file has %d, engine has %d", numLayers, len(t.engine.layers))
	}
	if int(hiddenDim) != t.engine.hiddenDim {
		return fmt.Errorf("hidden dim mismatch: file has %d, engine has %d", hiddenDim, t.engine.hiddenDim)
	}

	for i := range t.engine.layers {
		binary.Read(f, binary.LittleEndian, t.engine.layers[i].AttnResQuery.Data)
		binary.Read(f, binary.LittleEndian, t.engine.layers[i].FFNResQuery.Data)
	}

	fmt.Printf("[AttnRes] Loaded query vectors from %s (%d layers, dim=%d)\n", path, numLayers, hiddenDim)

	// Ensure AttnRes is enabled.
	if !t.engine.AttnResConfig.Enabled {
		t.engine.EnableAttnRes()
	}

	return nil
}

// LoadAttnResQueries is a convenience function to load pre-trained query vectors
// onto an engine without creating a full trainer.
func LoadAttnResQueries(engine *NativeEngine, path string) error {
	trainer := &AttnResTrainer{engine: engine}
	return trainer.Load(path)
}
