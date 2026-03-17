// orpo_trainer.go implements Odds Ratio Preference Optimization.
//
// WHAT: ORPO (Hong et al., 2024) combines supervised fine-tuning with
// preference alignment in a SINGLE training pass. Unlike DPO, it doesn't
// need a reference model at all - it modifies the SFT loss to include
// an odds-ratio penalty that discourages rejected responses.
//
// WHY: DPO needs a frozen reference model (doubles memory). ORPO achieves
// similar alignment quality with just the policy model by computing the
// odds ratio between chosen and rejected directly in the loss function.
//
// LOSS: L_ORPO = L_SFT(chosen) + lambda * L_OR
//   L_SFT  = standard cross-entropy on chosen response
//   L_OR   = -log(sigma(log(odds_chosen / odds_rejected)))
//   odds   = P(y) / (1 - P(y))  for a response y
//
// GAIN: Same GPU memory as SFT, better quality than SFT alone,
// competitive with DPO without needing a reference model.
package native

import (
	"math"
)

// ORPOConfig configures Odds Ratio Preference Optimization.
type ORPOConfig struct {
	Lambda float32 // Weight for the odds ratio loss (default: 1.0)
	Beta   float32 // Temperature for log-odds (default: 0.1)
}

// DefaultORPOConfig returns standard ORPO settings.
func DefaultORPOConfig() ORPOConfig {
	return ORPOConfig{
		Lambda: 1.0,
		Beta:   0.1,
	}
}

// ORPOTrainer manages ORPO training.
type ORPOTrainer struct {
	config ORPOConfig

	// Running stats.
	totalSteps      int64
	avgLoss         float64
	avgSFTLoss      float64
	avgORLoss       float64
	avgChosenProb   float64
	avgRejectedProb float64
}

// NewORPOTrainer creates an ORPO trainer.
func NewORPOTrainer(config ORPOConfig) *ORPOTrainer {
	return &ORPOTrainer{config: config}
}

// ORPOResult holds the output of one ORPO training step.
type ORPOResult struct {
	Loss        float32 // Total loss (SFT + lambda * OR)
	SFTLoss     float32 // Cross-entropy on chosen
	ORLoss      float32 // Odds ratio preference loss
	ChosenProb  float32 // Average probability of chosen tokens
	RejectedProb float32 // Average probability of rejected tokens
}

// ComputeLoss computes the ORPO loss for a batch.
//
// chosenLogProbs: log P(token | context) for each token in chosen responses.
// rejectedLogProbs: log P(token | context) for each token in rejected responses.
// chosenTargets: target token IDs for SFT loss on chosen responses.
// chosenLogits: raw logits [seqLen][vocabSize] for SFT cross-entropy.
func (t *ORPOTrainer) ComputeLoss(
	chosenLogProbs []float32,
	rejectedLogProbs []float32,
	chosenLogits [][]float32,
	chosenTargets []int32,
) ORPOResult {
	result := ORPOResult{}

	// Part 1: SFT loss on chosen response (standard cross-entropy).
	var sftLoss float64
	for pos, target := range chosenTargets {
		if int(target) >= len(chosenLogits[pos]) {
			continue
		}
		// log-softmax at this position.
		maxVal := chosenLogits[pos][0]
		for _, v := range chosenLogits[pos][1:] {
			if v > maxVal {
				maxVal = v
			}
		}
		var expSum float64
		for _, v := range chosenLogits[pos] {
			expSum += math.Exp(float64(v - maxVal))
		}
		logProb := float64(chosenLogits[pos][target]-maxVal) - math.Log(expSum)
		sftLoss -= logProb
	}
	if len(chosenTargets) > 0 {
		sftLoss /= float64(len(chosenTargets))
	}
	result.SFTLoss = float32(sftLoss)

	// Part 2: Odds ratio loss.
	// Compute average log-prob for chosen and rejected.
	var chosenAvgProb, rejectedAvgProb float64
	for _, lp := range chosenLogProbs {
		chosenAvgProb += float64(lp)
	}
	if len(chosenLogProbs) > 0 {
		chosenAvgProb /= float64(len(chosenLogProbs))
	}

	for _, lp := range rejectedLogProbs {
		rejectedAvgProb += float64(lp)
	}
	if len(rejectedLogProbs) > 0 {
		rejectedAvgProb /= float64(len(rejectedLogProbs))
	}

	// Convert to probabilities.
	pChosen := math.Exp(chosenAvgProb)
	pRejected := math.Exp(rejectedAvgProb)

	// Compute odds.
	oddsChosen := pChosen / (1.0 - pChosen + 1e-10)
	oddsRejected := pRejected / (1.0 - pRejected + 1e-10)

	// Log-odds ratio.
	logOddsRatio := math.Log(oddsChosen/(oddsRejected+1e-10) + 1e-10)

	// OR loss: -log(sigmoid(log_odds_ratio))
	orLoss := -logSigmoidF64(logOddsRatio)
	result.ORLoss = float32(orLoss)
	result.ChosenProb = float32(pChosen)
	result.RejectedProb = float32(pRejected)

	// Total loss.
	result.Loss = result.SFTLoss + t.config.Lambda*result.ORLoss

	// Update stats.
	t.totalSteps++
	t.avgLoss = t.avgLoss*0.99 + float64(result.Loss)*0.01
	t.avgSFTLoss = t.avgSFTLoss*0.99 + float64(result.SFTLoss)*0.01
	t.avgORLoss = t.avgORLoss*0.99 + float64(result.ORLoss)*0.01
	t.avgChosenProb = t.avgChosenProb*0.99 + pChosen*0.01
	t.avgRejectedProb = t.avgRejectedProb*0.99 + pRejected*0.01

	return result
}

// Stats returns ORPO training metrics.
func (t *ORPOTrainer) Stats() map[string]interface{} {
	return map[string]interface{}{
		"total_steps":       t.totalSteps,
		"avg_loss":          t.avgLoss,
		"avg_sft_loss":      t.avgSFTLoss,
		"avg_or_loss":       t.avgORLoss,
		"avg_chosen_prob":   t.avgChosenProb,
		"avg_rejected_prob": t.avgRejectedProb,
		"lambda":            t.config.Lambda,
	}
}

// logSigmoidF64 computes log(sigmoid(x)) numerically stable.
func logSigmoidF64(x float64) float64 {
	if x >= 0 {
		return -math.Log(1.0 + math.Exp(-x))
	}
	return x - math.Log(1.0+math.Exp(x))
}
