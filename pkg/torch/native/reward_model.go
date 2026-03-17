// reward_model.go implements a reward model for scoring LLM responses.
//
// WHAT: A reward model takes (prompt, response) pairs and outputs a scalar
// reward score. Higher scores = better responses. This is used by GRPO
// to rank sampled responses and by RLHF for the reward signal.
//
// ARCHITECTURE: The reward model is a transformer with a scalar head:
//   input -> transformer layers -> last hidden state -> linear -> scalar reward
//
// TRAINING: Trained on preference pairs (chosen > rejected) using a
// Bradley-Terry ranking loss: L = -log(sigma(r_chosen - r_rejected)).
//
// This is the same loss as DPO's implicit reward, but here the reward
// model is explicit and reusable across multiple policy training runs.
package native

import (
	"math"
)

// RewardModelConfig configures the reward model.
type RewardModelConfig struct {
	HiddenDim   int     // Hidden dimension of the final projection
	DropoutRate  float32 // Dropout for regularization (default: 0.0)
	Normalize    bool    // Whether to normalize rewards to zero mean
	ClipReward   float32 // Max absolute reward value (0 = no clip)
}

// DefaultRewardModelConfig returns standard settings.
func DefaultRewardModelConfig() RewardModelConfig {
	return RewardModelConfig{
		HiddenDim:  128,
		Normalize:  true,
		ClipReward: 5.0,
	}
}

// RewardHead is the scalar projection head on top of a transformer.
type RewardHead struct {
	config RewardModelConfig

	// Weights: linear projection from model hidden dim to scalar.
	W1     []float32 // [hiddenDim, modelDim] first projection
	B1     []float32 // [hiddenDim] bias
	W2     []float32 // [1, hiddenDim] scalar head
	B2     float32   // scalar bias

	// Reward statistics for normalization.
	runningMean float64
	runningVar  float64
	numSamples  int64
}

// NewRewardHead creates a reward projection head.
// modelDim is the hidden size of the base transformer model.
func NewRewardHead(modelDim int, config RewardModelConfig) *RewardHead {
	head := &RewardHead{
		config: config,
		W1:     make([]float32, config.HiddenDim*modelDim),
		B1:     make([]float32, config.HiddenDim),
		W2:     make([]float32, config.HiddenDim),
		runningVar: 1.0,
	}

	// Xavier initialization.
	scale := float32(math.Sqrt(2.0 / float64(modelDim+config.HiddenDim)))
	for i := range head.W1 {
		head.W1[i] = float32(math.Sin(float64(i)*0.017)) * scale
	}
	scale2 := float32(math.Sqrt(2.0 / float64(config.HiddenDim+1)))
	for i := range head.W2 {
		head.W2[i] = float32(math.Sin(float64(i)*0.031)) * scale2
	}

	return head
}

// Score computes a scalar reward from the last hidden state of a transformer.
// hiddenState: [modelDim] - the hidden state at the last token position.
func (h *RewardHead) Score(hiddenState []float32) float32 {
	modelDim := len(hiddenState)
	hiddenDim := h.config.HiddenDim

	// Layer 1: linear + GELU activation.
	hidden := make([]float32, hiddenDim)
	for i := 0; i < hiddenDim; i++ {
		var sum float64
		for j := 0; j < modelDim; j++ {
			sum += float64(h.W1[i*modelDim+j]) * float64(hiddenState[j])
		}
		hidden[i] = gelu(float32(sum) + h.B1[i])
	}

	// Layer 2: linear to scalar.
	var reward float64
	for i := 0; i < hiddenDim; i++ {
		reward += float64(h.W2[i]) * float64(hidden[i])
	}
	reward += float64(h.B2)

	r := float32(reward)

	// Update running statistics.
	h.numSamples++
	delta := reward - h.runningMean
	h.runningMean += delta / float64(h.numSamples)
	delta2 := reward - h.runningMean
	h.runningVar += (delta*delta2 - h.runningVar) / float64(h.numSamples)

	// Normalize to zero mean, unit variance.
	if h.config.Normalize && h.numSamples > 10 {
		std := math.Sqrt(h.runningVar + 1e-8)
		r = float32((reward - h.runningMean) / std)
	}

	// Clip extreme rewards.
	if h.config.ClipReward > 0 {
		if r > h.config.ClipReward {
			r = h.config.ClipReward
		}
		if r < -h.config.ClipReward {
			r = -h.config.ClipReward
		}
	}

	return r
}

// gelu computes GELU activation.
func gelu(x float32) float32 {
	return float32(0.5 * float64(x) * (1.0 + math.Tanh(math.Sqrt(2.0/math.Pi)*(float64(x)+0.044715*float64(x)*float64(x)*float64(x)))))
}

// RewardTrainer trains the reward head on preference pairs.
type RewardTrainer struct {
	head *RewardHead

	// Stats.
	totalSteps int64
	avgLoss    float64
	accuracy   float64 // Correct ranking accuracy
}

// NewRewardTrainer creates a reward model trainer.
func NewRewardTrainer(head *RewardHead) *RewardTrainer {
	return &RewardTrainer{head: head}
}

// RankingLoss computes the Bradley-Terry loss for a batch of preference pairs.
// Returns loss and per-pair accuracy (did chosen score > rejected?).
func (t *RewardTrainer) RankingLoss(
	chosenHiddenStates [][]float32,
	rejectedHiddenStates [][]float32,
) (float32, float32) {
	batchSize := len(chosenHiddenStates)
	var totalLoss float64
	correct := 0

	for i := 0; i < batchSize; i++ {
		rChosen := t.head.Score(chosenHiddenStates[i])
		rRejected := t.head.Score(rejectedHiddenStates[i])

		// Bradley-Terry loss: -log(sigmoid(r_chosen - r_rejected)).
		margin := float64(rChosen - rRejected)
		loss := -logSigmoidF64(margin)
		totalLoss += loss

		if rChosen > rRejected {
			correct++
		}
	}

	avgLoss := float32(totalLoss / float64(batchSize))
	acc := float32(correct) / float32(batchSize)

	// Update stats.
	t.totalSteps++
	t.avgLoss = t.avgLoss*0.99 + float64(avgLoss)*0.01
	t.accuracy = t.accuracy*0.99 + float64(acc)*0.01

	return avgLoss, acc
}

// ScoreBatch scores multiple responses for a single prompt (used by GRPO).
func (h *RewardHead) ScoreBatch(hiddenStates [][]float32) []float32 {
	scores := make([]float32, len(hiddenStates))
	for i, hs := range hiddenStates {
		scores[i] = h.Score(hs)
	}
	return scores
}

// Stats returns reward model metrics.
func (t *RewardTrainer) Stats() map[string]interface{} {
	return map[string]interface{}{
		"total_steps": t.totalSteps,
		"avg_loss":    t.avgLoss,
		"accuracy":    t.accuracy,
	}
}
