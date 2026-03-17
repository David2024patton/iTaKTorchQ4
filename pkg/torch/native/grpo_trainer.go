// grpo_trainer.go implements Group Relative Policy Optimization for alignment.
//
// WHAT: GRPO (Shao et al., 2024) is DeepSeek's alternative to PPO for RLHF.
// Instead of learning a value function (critic), it estimates advantages by
// sampling multiple responses per prompt and using group statistics.
//
// WHY: PPO requires a critic model (extra memory + training complexity).
// GRPO eliminates the critic entirely by:
//   1. Sample G responses per prompt from the policy
//   2. Score each response with a reward function
//   3. Normalize rewards within the group (advantage = (r - mean) / std)
//   4. Update policy to increase probability of above-average responses
//
// GAIN: Same alignment quality as PPO with half the memory (no critic model)
// and simpler training loop. Used by DeepSeek-R1, DeepSeek-V3.
//
// REFERENCE: Axolotl, TRL, and DeepSeek implement this pattern.
package native

import (
	"math"
	"math/rand"
)

// GRPOConfig configures Group Relative Policy Optimization.
type GRPOConfig struct {
	GroupSize       int     // Number of responses to sample per prompt (default: 4)
	Beta            float32 // KL penalty coefficient (default: 0.04)
	ClipRange       float32 // PPO-style clipping range (default: 0.2)
	Temperature     float32 // Sampling temperature for response generation (default: 0.7)
	MinGroupStd     float32 // Minimum group std to prevent division by zero (default: 1e-4)
	KLType          string  // "forward" or "reverse" KL divergence (default: "reverse")
}

// DefaultGRPOConfig returns standard GRPO settings.
func DefaultGRPOConfig() GRPOConfig {
	return GRPOConfig{
		GroupSize:   4,
		Beta:        0.04,
		ClipRange:   0.2,
		Temperature: 0.7,
		MinGroupStd: 1e-4,
		KLType:      "reverse",
	}
}

// GRPOTrainer manages group relative policy optimization.
type GRPOTrainer struct {
	config GRPOConfig

	// Stats.
	totalSteps     int64
	avgLoss        float64
	avgReward      float64
	avgKL          float64
	avgAdvantage   float64
}

// NewGRPOTrainer creates a GRPO trainer.
func NewGRPOTrainer(config GRPOConfig) *GRPOTrainer {
	return &GRPOTrainer{config: config}
}

// GRPOGroup holds sampled responses and their rewards for one prompt.
type GRPOGroup struct {
	PromptTokens []int32            // Shared prompt
	Responses    [][]int32          // G sampled responses
	Rewards      []float32          // Reward for each response
	LogProbs     []float32          // Policy log-probs for each response
	RefLogProbs  []float32          // Reference model log-probs for each response
}

// ComputeAdvantages normalizes rewards within the group to get advantages.
// Returns per-response advantages with zero mean and unit variance.
func (t *GRPOTrainer) ComputeAdvantages(rewards []float32) []float32 {
	n := len(rewards)
	advantages := make([]float32, n)

	// Compute group mean and std.
	var sum, sumSq float64
	for _, r := range rewards {
		sum += float64(r)
		sumSq += float64(r) * float64(r)
	}
	mean := sum / float64(n)
	variance := sumSq/float64(n) - mean*mean
	std := math.Sqrt(variance)
	if std < float64(t.config.MinGroupStd) {
		std = float64(t.config.MinGroupStd)
	}

	// Normalize.
	for i, r := range rewards {
		advantages[i] = float32((float64(r) - mean) / std)
	}

	return advantages
}

// ComputeLoss computes the GRPO policy gradient loss for a group.
//
// The loss encourages the policy to increase probability for responses
// with positive advantages (above-average rewards) and decrease for negative.
//
// L = -E[min(ratio * advantage, clip(ratio, 1-eps, 1+eps) * advantage)] + beta * KL
func (t *GRPOTrainer) ComputeLoss(group *GRPOGroup) GRPOResult {
	advantages := t.ComputeAdvantages(group.Rewards)
	n := len(group.Responses)

	var totalLoss, totalKL float64
	ratios := make([]float32, n)

	for i := 0; i < n; i++ {
		// Importance sampling ratio: pi(y)/pi_old(y).
		logRatio := group.LogProbs[i] - group.RefLogProbs[i]
		ratio := float32(math.Exp(float64(logRatio)))
		ratios[i] = ratio

		// Clipped objective (PPO-style).
		adv := advantages[i]
		unclipped := ratio * adv
		clipped := clampF32(ratio, 1.0-t.config.ClipRange, 1.0+t.config.ClipRange) * adv

		// Take the pessimistic bound.
		var surr float32
		if unclipped < clipped {
			surr = unclipped
		} else {
			surr = clipped
		}
		totalLoss -= float64(surr) // Negative because we maximize.

		// KL divergence penalty.
		var kl float64
		switch t.config.KLType {
		case "forward":
			// Forward KL: E_ref[log(ref/pi)]
			kl = -float64(logRatio)
		default:
			// Reverse KL (default): E_pi[log(pi/ref)]
			kl = float64(logRatio)
		}
		totalKL += kl
	}

	avgLoss := float32(totalLoss / float64(n))
	avgKL := float32(totalKL / float64(n))

	// Total loss = policy loss + KL penalty.
	finalLoss := avgLoss + t.config.Beta*avgKL

	// Update running stats.
	t.totalSteps++
	t.avgLoss = t.avgLoss*0.99 + float64(finalLoss)*0.01
	t.avgKL = t.avgKL*0.99 + float64(avgKL)*0.01

	var rewSum float64
	for _, r := range group.Rewards {
		rewSum += float64(r)
	}
	t.avgReward = t.avgReward*0.99 + rewSum/float64(n)*0.01

	var advSum float64
	for _, a := range advantages {
		advSum += float64(a)
	}
	t.avgAdvantage = t.avgAdvantage*0.99 + advSum/float64(n)*0.01

	return GRPOResult{
		Loss:       finalLoss,
		KLPenalty:  avgKL,
		Advantages: advantages,
		Ratios:     ratios,
	}
}

// GRPOResult holds the output of one GRPO training step.
type GRPOResult struct {
	Loss       float32
	KLPenalty  float32
	Advantages []float32
	Ratios     []float32
}

// SampleResponses generates G responses for a prompt using temperature sampling.
// In practice, this calls the inference engine; here we define the interface.
// generateFn should return token IDs for one sampled response.
func (t *GRPOTrainer) SampleResponses(
	prompt []int32,
	generateFn func(prompt []int32, temperature float32) []int32,
) [][]int32 {
	responses := make([][]int32, t.config.GroupSize)
	for i := 0; i < t.config.GroupSize; i++ {
		// Add slight temperature variation for diversity.
		temp := t.config.Temperature + float32(rand.NormFloat64())*0.05
		if temp < 0.1 {
			temp = 0.1
		}
		responses[i] = generateFn(prompt, temp)
	}
	return responses
}

// Stats returns GRPO training metrics.
func (t *GRPOTrainer) Stats() map[string]interface{} {
	return map[string]interface{}{
		"total_steps":    t.totalSteps,
		"avg_loss":       t.avgLoss,
		"avg_reward":     t.avgReward,
		"avg_kl":         t.avgKL,
		"avg_advantage":  t.avgAdvantage,
		"group_size":     t.config.GroupSize,
		"beta":           t.config.Beta,
		"clip_range":     t.config.ClipRange,
	}
}

// clampF32 clamps a float32 to [lo, hi].
func clampF32(v, lo, hi float32) float32 {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}
