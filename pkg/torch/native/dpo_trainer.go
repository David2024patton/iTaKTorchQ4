// dpo_trainer.go implements Direct Preference Optimization for LLM alignment.
//
// WHAT: DPO (Rafailov et al., 2023) trains models to prefer "chosen" responses
// over "rejected" responses WITHOUT a separate reward model. It directly
// optimizes the policy using a closed-form solution to the RLHF objective.
//
// WHY: Traditional RLHF requires training a reward model, then using PPO to
// optimize against it (complex, unstable). DPO collapses this into a single
// supervised loss on preference pairs, making alignment as simple as SFT.
//
// LOSS: L_DPO = -log(sigma(beta * (log pi(y_w|x)/pi_ref(y_w|x) - log pi(y_l|x)/pi_ref(y_l|x))))
//   pi     = policy model (being trained)
//   pi_ref = reference model (frozen copy of initial policy)
//   y_w    = preferred/chosen response
//   y_l    = rejected response
//   beta   = temperature (controls deviation from reference)
//
// REFERENCE: Axolotl, TRL, LlamaFactory all implement this pattern.
package native

import (
	"math"
)

// DPOConfig configures Direct Preference Optimization training.
type DPOConfig struct {
	Beta           float32 // Temperature controlling policy deviation (default: 0.1)
	LabelSmoothing float32 // Optional smoothing (default: 0.0)
	LossType       string  // "sigmoid" (standard) or "hinge" or "ipo"
	ReferenceFreeze bool   // Whether reference model is frozen (always true in standard DPO)
}

// DefaultDPOConfig returns standard DPO settings.
func DefaultDPOConfig() DPOConfig {
	return DPOConfig{
		Beta:           0.1,
		LabelSmoothing: 0.0,
		LossType:       "sigmoid",
		ReferenceFreeze: true,
	}
}

// DPOTrainer manages preference optimization training.
type DPOTrainer struct {
	config DPOConfig

	// Stats.
	totalSteps    int64
	avgLoss       float64
	avgChosenReward  float64
	avgRejectedReward float64
	avgMargin     float64
}

// NewDPOTrainer creates a DPO trainer.
func NewDPOTrainer(config DPOConfig) *DPOTrainer {
	return &DPOTrainer{config: config}
}

// PreferencePair holds one training example for DPO.
type PreferencePair struct {
	Prompt          []int32 // Shared prompt tokens
	ChosenResponse  []int32 // Preferred response tokens
	RejectedResponse []int32 // Non-preferred response tokens
}

// DPOBatchResult holds the loss and metrics from one DPO step.
type DPOBatchResult struct {
	Loss            float32
	ChosenRewards   []float32 // Per-example rewards for chosen
	RejectedRewards []float32 // Per-example rewards for rejected
	Margins         []float32 // chosen_reward - rejected_reward
}

// ComputeLoss computes the DPO loss for a batch of preference pairs.
//
// Inputs are log-probabilities from the policy and reference models:
//   policyChosenLogps:    log P_pi(y_w | x) for each example
//   policyRejectedLogps:  log P_pi(y_l | x) for each example
//   refChosenLogps:       log P_ref(y_w | x) for each example
//   refRejectedLogps:     log P_ref(y_l | x) for each example
func (t *DPOTrainer) ComputeLoss(
	policyChosenLogps []float32,
	policyRejectedLogps []float32,
	refChosenLogps []float32,
	refRejectedLogps []float32,
) DPOBatchResult {
	batchSize := len(policyChosenLogps)
	result := DPOBatchResult{
		ChosenRewards:   make([]float32, batchSize),
		RejectedRewards: make([]float32, batchSize),
		Margins:         make([]float32, batchSize),
	}

	var totalLoss float64

	for i := 0; i < batchSize; i++ {
		// Log-ratio: how much the policy diverges from reference for each response.
		chosenLogRatio := policyChosenLogps[i] - refChosenLogps[i]
		rejectedLogRatio := policyRejectedLogps[i] - refRejectedLogps[i]

		// Implicit rewards.
		result.ChosenRewards[i] = t.config.Beta * chosenLogRatio
		result.RejectedRewards[i] = t.config.Beta * rejectedLogRatio
		result.Margins[i] = result.ChosenRewards[i] - result.RejectedRewards[i]

		// DPO loss.
		logits := t.config.Beta * (chosenLogRatio - rejectedLogRatio)

		var loss float64
		switch t.config.LossType {
		case "hinge":
			// Hinge loss: max(0, 1 - logits)
			if float64(logits) < 1.0 {
				loss = 1.0 - float64(logits)
			}
		case "ipo":
			// Identity Preference Optimization: (logits - 1/(2*beta))^2
			target := 1.0 / (2.0 * float64(t.config.Beta))
			diff := float64(logits) - target
			loss = diff * diff
		default:
			// Standard sigmoid loss: -log(sigmoid(logits))
			loss = -logSigmoid(float64(logits))
		}

		// Label smoothing.
		if t.config.LabelSmoothing > 0 {
			smoothLoss := -logSigmoid(-float64(logits)) // Reverse label loss.
			loss = (1.0-float64(t.config.LabelSmoothing))*loss + float64(t.config.LabelSmoothing)*smoothLoss
		}

		totalLoss += loss
	}

	result.Loss = float32(totalLoss / float64(batchSize))

	// Update running stats.
	t.totalSteps++
	t.avgLoss = t.avgLoss*0.99 + float64(result.Loss)*0.01
	for i := range result.ChosenRewards {
		t.avgChosenReward = t.avgChosenReward*0.99 + float64(result.ChosenRewards[i])*0.01
		t.avgRejectedReward = t.avgRejectedReward*0.99 + float64(result.RejectedRewards[i])*0.01
		t.avgMargin = t.avgMargin*0.99 + float64(result.Margins[i])*0.01
	}

	return result
}

// ComputeLogProbs computes per-token log probabilities for a sequence.
// logits: [seqLen, vocabSize], targets: [seqLen].
// Returns the sum of log probabilities (log P(sequence)).
func ComputeLogProbs(logits [][]float32, targets []int32) float32 {
	var totalLogProb float64

	for pos, target := range targets {
		if int(target) >= len(logits[pos]) {
			continue
		}

		// Log-softmax for this position.
		maxVal := logits[pos][0]
		for _, v := range logits[pos][1:] {
			if v > maxVal {
				maxVal = v
			}
		}

		var expSum float64
		for _, v := range logits[pos] {
			expSum += math.Exp(float64(v - maxVal))
		}

		logProb := float64(logits[pos][target]-maxVal) - math.Log(expSum)
		totalLogProb += logProb
	}

	return float32(totalLogProb)
}

// Stats returns DPO training metrics.
func (t *DPOTrainer) Stats() map[string]interface{} {
	return map[string]interface{}{
		"total_steps":       t.totalSteps,
		"avg_loss":          t.avgLoss,
		"avg_chosen_reward": t.avgChosenReward,
		"avg_rejected_reward": t.avgRejectedReward,
		"avg_margin":        t.avgMargin,
		"beta":              t.config.Beta,
		"loss_type":         t.config.LossType,
	}
}

// logSigmoid computes log(sigmoid(x)) = -log(1 + exp(-x)) in a numerically
// stable way.
func logSigmoid(x float64) float64 {
	if x >= 0 {
		return -math.Log(1.0 + math.Exp(-x))
	}
	return x - math.Log(1.0+math.Exp(x))
}
