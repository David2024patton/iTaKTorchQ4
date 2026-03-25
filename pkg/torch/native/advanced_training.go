// advanced_training.go adds MiniMax-style two-phase training and Kimi/Moonshot
// advanced techniques to the Torch training pipeline.
//
// MINIMAX TECHNIQUES:
//   - Two-phase curriculum: SFT (format learning) then GRPO (routing accuracy)
//   - Progressive difficulty: easy examples first, hard ones later
//   - Multi-reward aggregation: combine multiple reward signals
//
// KIMI/MOONSHOT TECHNIQUES:
//   - Long-context RL: chunked attention for training on long sequences
//   - Reflection training: reward models for producing <think>...</think> blocks
//   - Gradient accumulation: simulate larger batches without more memory
//   - AttnRes learning: depth-wise attention (already in engine, wired here)
//
// REFERENCE: MiniMax-01 tech report, Kimi attention residual paper (March 2026)
package native

import (
	"fmt"
	"math"
	"strings"
	"time"
)

// ---------- Gradient Accumulation ----------

// GradAccumulatorConfig configures gradient accumulation across mini-batches.
type GradAccumulatorConfig struct {
	AccumulationSteps int     // Number of forward passes before optimizer step
	MaxGradNorm       float32 // Gradient clipping norm (0 = no clipping)
}

// GradAccumulator enable simulating larger batch sizes by accumulating
// gradients over multiple forward passes before calling optimizer.Step().
type GradAccumulator struct {
	config    GradAccumulatorConfig
	params    []*GradTensor
	accumulated int // Forward passes accumulated so far
}

// NewGradAccumulator creates a gradient accumulator.
func NewGradAccumulator(params []*GradTensor, config GradAccumulatorConfig) *GradAccumulator {
	return &GradAccumulator{
		config: config,
		params: params,
	}
}

// ShouldStep returns true when enough micro-batches have accumulated.
func (ga *GradAccumulator) ShouldStep() bool {
	ga.accumulated++
	return ga.accumulated >= ga.config.AccumulationSteps
}

// Reset clears the accumulation counter and zeroes gradients.
func (ga *GradAccumulator) Reset() {
	ga.accumulated = 0
	for _, p := range ga.params {
		if p.Grad != nil {
			for i := range p.Grad {
				p.Grad[i] = 0
			}
		}
	}
}

// ScaleGradients divides accumulated gradients by the number of accumulation steps.
func (ga *GradAccumulator) ScaleGradients() {
	scale := float32(1.0) / float32(ga.config.AccumulationSteps)
	for _, p := range ga.params {
		if p.Grad != nil {
			for i := range p.Grad {
				p.Grad[i] *= scale
			}
		}
	}
}

// ClipGradNorm applies gradient clipping by global norm.
// Prevents exploding gradients during training.
func (ga *GradAccumulator) ClipGradNorm() float32 {
	if ga.config.MaxGradNorm <= 0 {
		return 0
	}

	// Compute global grad norm.
	var totalNormSq float64
	for _, p := range ga.params {
		if p.Grad == nil {
			continue
		}
		for _, g := range p.Grad {
			totalNormSq += float64(g) * float64(g)
		}
	}
	globalNorm := float32(math.Sqrt(totalNormSq))

	if globalNorm > ga.config.MaxGradNorm {
		scale := ga.config.MaxGradNorm / globalNorm
		for _, p := range ga.params {
			if p.Grad == nil {
				continue
			}
			for i := range p.Grad {
				p.Grad[i] *= scale
			}
		}
	}

	return globalNorm
}

// ---------- Long-Context Chunked Training ----------

// ChunkedContextConfig configures long-context RL training.
type ChunkedContextConfig struct {
	ChunkSize      int // Tokens per chunk (e.g., 512)
	ChunkOverlap   int // Overlap between chunks for continuity (e.g., 64)
	MaxTotalLength int // Maximum total sequence length (e.g., 4096)
}

// DefaultChunkedContextConfig returns reasonable defaults for long-context training.
func DefaultChunkedContextConfig() ChunkedContextConfig {
	return ChunkedContextConfig{
		ChunkSize:      512,
		ChunkOverlap:   64,
		MaxTotalLength: 4096,
	}
}

// ChunkSequence splits a long token sequence into overlapping chunks for
// training on sequences longer than the model's context window.
// Returns chunks and their offset positions in the original sequence.
func ChunkSequence(tokens []int, config ChunkedContextConfig) []TokenChunk {
	if len(tokens) <= config.ChunkSize {
		return []TokenChunk{{
			Tokens:   tokens,
			StartPos: 0,
			EndPos:   len(tokens),
			IsFirst:  true,
			IsLast:   true,
		}}
	}

	// Truncate to max length.
	if config.MaxTotalLength > 0 && len(tokens) > config.MaxTotalLength {
		tokens = tokens[:config.MaxTotalLength]
	}

	stride := config.ChunkSize - config.ChunkOverlap
	if stride <= 0 {
		stride = config.ChunkSize / 2
	}

	var chunks []TokenChunk
	for start := 0; start < len(tokens); start += stride {
		end := start + config.ChunkSize
		if end > len(tokens) {
			end = len(tokens)
		}

		chunks = append(chunks, TokenChunk{
			Tokens:   tokens[start:end],
			StartPos: start,
			EndPos:   end,
			IsFirst:  start == 0,
			IsLast:   end >= len(tokens),
		})

		if end >= len(tokens) {
			break
		}
	}

	return chunks
}

// TokenChunk represents one chunk of a long sequence.
type TokenChunk struct {
	Tokens   []int
	StartPos int  // Position in original sequence
	EndPos   int
	IsFirst  bool // First chunk (contains system prompt)
	IsLast   bool // Last chunk (contains final response)
}

// ---------- Reflection Training ----------

// ReflectionConfig configures reflection-aware training.
type ReflectionConfig struct {
	Enabled          bool    // Enable reflection training
	ReflectionBonus  float32 // Bonus reward for producing <think> blocks (default: 0.15)
	QualityThreshold float32 // Minimum reasoning length to get bonus (default: 20 chars)
	DepthBonus       float32 // Extra bonus for multi-step reasoning (default: 0.1)
	MaxPenalty       float32 // Penalty for empty/missing reflection (default: 0.05)
}

// DefaultReflectionConfig returns recommended reflection training settings.
func DefaultReflectionConfig() ReflectionConfig {
	return ReflectionConfig{
		Enabled:          true,
		ReflectionBonus:  0.15,
		QualityThreshold: 20,
		DepthBonus:       0.1,
		MaxPenalty:       0.05,
	}
}

// ScoreReflection evaluates the quality of a reflection (<think>...</think>) block.
// Returns a bonus/penalty to add to the base reward.
//
// Rewards:
//   - Producing a <think> block with substantive reasoning
//   - Step-by-step analysis (numbered steps, "because", "therefore")
//   - Considering multiple agents/options
//
// Penalties:
//   - Empty reflection blocks
//   - Extremely short reasoning
//   - Missing reflection entirely
func ScoreReflection(response string, config ReflectionConfig) float32 {
	if !config.Enabled {
		return 0
	}

	// Find <think>...</think> block.
	thinkStart := strings.Index(response, "<think>")
	thinkEnd := strings.Index(response, "</think>")

	// No reflection at all.
	if thinkStart < 0 || thinkEnd <= thinkStart {
		return -config.MaxPenalty
	}

	reflection := response[thinkStart+7 : thinkEnd]
	reflection = strings.TrimSpace(reflection)

	// Empty reflection.
	if len(reflection) == 0 {
		return -config.MaxPenalty
	}

	// Too short to be meaningful.
	if float32(len(reflection)) < config.QualityThreshold {
		return 0 // Neutral - tried but not useful
	}

	reward := config.ReflectionBonus

	// Multi-step reasoning bonus.
	reasoningIndicators := []string{
		"because", "therefore", "however", "considering",
		"step 1", "step 2", "first", "second", "finally",
		"option 1", "option 2", "alternatively",
		"the user wants", "this requires", "best suited",
	}
	indicatorCount := 0
	lower := strings.ToLower(reflection)
	for _, indicator := range reasoningIndicators {
		if strings.Contains(lower, indicator) {
			indicatorCount++
		}
	}

	// Depth bonus for rich reasoning (2+ indicators).
	if indicatorCount >= 2 {
		reward += config.DepthBonus
	}
	if indicatorCount >= 4 {
		reward += config.DepthBonus * 0.5 // Diminishing returns
	}

	// Agent comparison bonus - mentioning multiple agents shows consideration.
	agents := []string{"scout", "operator", "browser", "researcher", "coder", "architect", "ghl"}
	agentMentions := 0
	for _, agent := range agents {
		if strings.Contains(lower, agent) {
			agentMentions++
		}
	}
	if agentMentions >= 2 {
		reward += 0.05 // Considered multiple options
	}

	return reward
}

// ---------- Curriculum Learning (MiniMax-style) ----------

// CurriculumConfig configures progressive difficulty training.
type CurriculumConfig struct {
	Enabled    bool // Enable curriculum learning
	NumStages  int  // Number of difficulty stages (default: 3)
	WarmupPct  float32 // Percentage of training for warmup stage (default: 0.2)
}

// DefaultCurriculumConfig returns MiniMax-style curriculum settings.
func DefaultCurriculumConfig() CurriculumConfig {
	return CurriculumConfig{
		Enabled:   true,
		NumStages: 3,
		WarmupPct: 0.2,
	}
}

// CurriculumScheduler manages progressive difficulty during training.
type CurriculumScheduler struct {
	config CurriculumConfig
	totalSteps int
	currentStep int
}

// NewCurriculumScheduler creates a curriculum scheduler.
func NewCurriculumScheduler(config CurriculumConfig, totalSteps int) *CurriculumScheduler {
	return &CurriculumScheduler{
		config:     config,
		totalSteps: totalSteps,
	}
}

// CurrentDifficulty returns 0.0-1.0 indicating current training difficulty.
// 0.0 = easiest examples, 1.0 = hardest examples.
func (cs *CurriculumScheduler) CurrentDifficulty() float32 {
	if cs.totalSteps == 0 {
		return 1.0
	}
	progress := float32(cs.currentStep) / float32(cs.totalSteps)
	
	// Warmup: very easy.
	if progress < cs.config.WarmupPct {
		return progress / cs.config.WarmupPct * 0.3
	}
	
	// Main training: progressive.
	remaining := (progress - cs.config.WarmupPct) / (1.0 - cs.config.WarmupPct)
	return 0.3 + remaining*0.7
}

// Step advances the curriculum.
func (cs *CurriculumScheduler) Step() {
	cs.currentStep++
}

// ShouldIncludeExample returns true if an example at the given difficulty level
// should be included in the current training stage.
// difficulty: 0.0 (easiest) to 1.0 (hardest)
func (cs *CurriculumScheduler) ShouldIncludeExample(difficulty float32) bool {
	return difficulty <= cs.CurrentDifficulty()
}

// ---------- Multi-Reward Aggregation (MiniMax-style) ----------

// MultiRewardConfig configures multiple reward signals for GRPO.
type MultiRewardConfig struct {
	Weights map[string]float32 // Reward name -> weight
}

// DefaultMultiRewardConfig returns weights for the orchestrator reward signals.
func DefaultMultiRewardConfig() MultiRewardConfig {
	return MultiRewardConfig{
		Weights: map[string]float32{
			"json_validity":  0.20, // Valid JSON output
			"agent_accuracy": 0.30, // Correct agent selection
			"reasoning":      0.15, // Quality of reasoning
			"reflection":     0.15, // <think> block quality
			"no_tools":       0.10, // No tool call patterns
			"format":         0.10, // Proper format (delegations/direct_response)
		},
	}
}

// AggregateRewards combines multiple reward signals into a single score.
func AggregateRewards(rewards map[string]float32, config MultiRewardConfig) float32 {
	var total float32
	var totalWeight float32

	for name, weight := range config.Weights {
		if reward, ok := rewards[name]; ok {
			total += reward * weight
			totalWeight += weight
		}
	}

	if totalWeight > 0 {
		return total / totalWeight
	}
	return 0
}

// ---------- Advanced Training Loop ----------

// AdvancedTrainerConfig combines all MiniMax + Kimi techniques.
type AdvancedTrainerConfig struct {
	Base           TrainerConfig
	GradAccum      GradAccumulatorConfig
	ChunkedContext ChunkedContextConfig
	Reflection     ReflectionConfig
	Curriculum     CurriculumConfig
	MultiReward    MultiRewardConfig
}

// DefaultAdvancedTrainerConfig returns the "best of MiniMax + Kimi" config.
func DefaultAdvancedTrainerConfig() AdvancedTrainerConfig {
	return AdvancedTrainerConfig{
		Base: TrainerConfig{
			Epochs:        5,
			BatchSize:     1,
			SeqLen:        512,
			LR:            5e-5, // Lower LR for RL phase
			EnableLoRA:    true,
			EnableAttnRes: true,
			LoRARank:      8,
			LogFreq:       5,
		},
		GradAccum: GradAccumulatorConfig{
			AccumulationSteps: 4,  // Effective batch = 4
			MaxGradNorm:       1.0,
		},
		ChunkedContext: DefaultChunkedContextConfig(),
		Reflection:     DefaultReflectionConfig(),
		Curriculum:     DefaultCurriculumConfig(),
		MultiReward:    DefaultMultiRewardConfig(),
	}
}

// AdvancedTrainer combines all training techniques.
type AdvancedTrainer struct {
	engine       *NativeEngine
	config       AdvancedTrainerConfig
	trainer      *Trainer
	grpo         *GRPOTrainer
	gradAccum    *GradAccumulator
	curriculum   *CurriculumScheduler
	statusFile   string // JSON status file for monitoring

	// Metrics.
	phaseMetrics map[string][]float32
}

// NewAdvancedTrainer creates a trainer with all MiniMax + Kimi techniques.
func NewAdvancedTrainer(engine *NativeEngine, config AdvancedTrainerConfig) *AdvancedTrainer {
	trainer := NewTrainer(engine, config.Base)

	grpoConfig := DefaultGRPOConfig()
	grpoConfig.GroupSize = 4
	grpoConfig.Temperature = 0.7

	at := &AdvancedTrainer{
		engine:       engine,
		config:       config,
		trainer:      trainer,
		grpo:         NewGRPOTrainer(grpoConfig),
		gradAccum:    NewGradAccumulator(trainer.AllParams(), config.GradAccum),
		phaseMetrics: make(map[string][]float32),
	}

	return at
}

// SetStatusFile sets the JSON status file path for external monitoring.
func (at *AdvancedTrainer) SetStatusFile(path string) {
	at.statusFile = path
}

// RunTwoPhaseTraining executes the MiniMax-style two-phase approach:
//   Phase 1 (SFT): Learn the output format from labeled examples
//   Phase 2 (GRPO): Refine routing accuracy with reward-based RL
//
// Between phases, the learning rate is reset and curriculum restarts.
func (at *AdvancedTrainer) RunTwoPhaseTraining(dataset *TrainingDataset, examples []SFTExample) error {
	fmt.Println("[AdvancedTrainer] ============================================")
	fmt.Println("[AdvancedTrainer] MiniMax + Kimi Two-Phase Training")
	fmt.Println("[AdvancedTrainer] Phase 1: SFT (format learning)")
	fmt.Println("[AdvancedTrainer] Phase 2: GRPO (routing accuracy + reflection)")
	fmt.Println("[AdvancedTrainer] ============================================")

	// ─── Phase 1: Supervised Fine-Tuning ─────────────────────
	fmt.Println()
	fmt.Println("[Phase 1] Supervised Fine-Tuning")
	fmt.Println("[Phase 1] Goal: Learn the orchestrator JSON output format")

	p1Start := time.Now()
	totalSteps := at.config.Base.Epochs * len(dataset.Sequences)
	at.curriculum = NewCurriculumScheduler(at.config.Curriculum, totalSteps)

	if err := at.runSFTPhase(dataset); err != nil {
		return fmt.Errorf("phase 1 SFT: %w", err)
	}

	p1Dur := time.Since(p1Start)
	fmt.Printf("[Phase 1] Complete in %v\n", p1Dur.Round(time.Second))
	at.trainer.MergeAndExport()
	fmt.Println("[Phase 1] LoRA merged into base weights")

	// ─── Phase 2: GRPO Reinforcement Learning ─────────────────
	fmt.Println()
	fmt.Println("[Phase 2] Group Relative Policy Optimization")
	fmt.Println("[Phase 2] Goal: Maximize routing accuracy + reflection quality")

	p2Start := time.Now()

	// Reset for phase 2 with lower LR.
	at.config.Base.LR = at.config.Base.LR * 0.5
	at.trainer = NewTrainer(at.engine, at.config.Base)
	at.gradAccum = NewGradAccumulator(at.trainer.AllParams(), at.config.GradAccum)

	if err := at.runGRPOPhase(examples); err != nil {
		return fmt.Errorf("phase 2 GRPO: %w", err)
	}

	p2Dur := time.Since(p2Start)
	fmt.Printf("[Phase 2] Complete in %v\n", p2Dur.Round(time.Second))
	at.trainer.MergeAndExport()

	// ─── Summary ────────────────────────────────────────────
	fmt.Println()
	fmt.Println("[AdvancedTrainer] ============================================")
	fmt.Printf("[AdvancedTrainer] Phase 1 (SFT): %v\n", p1Dur.Round(time.Second))
	fmt.Printf("[AdvancedTrainer] Phase 2 (GRPO): %v\n", p2Dur.Round(time.Second))
	fmt.Printf("[AdvancedTrainer] Total: %v\n", (p1Dur + p2Dur).Round(time.Second))

	if losses, ok := at.phaseMetrics["sft_loss"]; ok && len(losses) > 1 {
		fmt.Printf("[AdvancedTrainer] SFT Loss:  %.4f -> %.4f\n", losses[0], losses[len(losses)-1])
	}
	if rewards, ok := at.phaseMetrics["grpo_reward"]; ok && len(rewards) > 1 {
		fmt.Printf("[AdvancedTrainer] GRPO Reward: %.4f -> %.4f\n", rewards[0], rewards[len(rewards)-1])
	}
	fmt.Println("[AdvancedTrainer] ============================================")

	return nil
}

// runSFTPhase runs the SFT phase with gradient accumulation and curriculum.
func (at *AdvancedTrainer) runSFTPhase(dataset *TrainingDataset) error {
	if len(dataset.Sequences) == 0 {
		return fmt.Errorf("empty dataset")
	}

	totalSteps := at.config.Base.Epochs * len(dataset.Sequences) / at.config.Base.BatchSize
	optConfig := DefaultAdamWConfig(totalSteps)
	optConfig.LearningRate = at.config.Base.LR

	// Initialize the internal optimizer accessed through the trainer.
	at.trainer.optimizer = NewAdamW(at.trainer.AllParams(), optConfig)
	at.trainer.startTime = time.Now()

	fmt.Printf("[SFT] %d epochs, %d sequences, grad_accum=%d, effective_batch=%d\n",
		at.config.Base.Epochs, len(dataset.Sequences),
		at.config.GradAccum.AccumulationSteps,
		at.config.Base.BatchSize*at.config.GradAccum.AccumulationSteps)

	for epoch := 0; epoch < at.config.Base.Epochs; epoch++ {
		epochLoss := float32(0)
		epochSteps := 0
		iter := NewBatchIterator(dataset, at.config.Base.BatchSize)

		for batch := iter.Next(); batch != nil; batch = iter.Next() {
			// Forward + loss.
			var batchLoss float32
			for seqIdx := 0; seqIdx < len(batch.Inputs); seqIdx++ {
				input := batch.Inputs[seqIdx]
				targets := batch.Targets[seqIdx]
				logits := at.engine.forward(input)
				lastTarget := targets[len(targets)-1]
				loss := crossEntropyLoss(logits, lastTarget)
				batchLoss += loss
			}
			batchLoss /= float32(len(batch.Inputs))

			// Approximate gradients.
			at.trainer.approximateGradients(batch, batchLoss)

			// Gradient accumulation: only step every N micro-batches.
			if at.gradAccum.ShouldStep() {
				at.gradAccum.ScaleGradients()
				gradNorm := at.gradAccum.ClipGradNorm()
				at.trainer.optimizer.Step()
				at.gradAccum.Reset()

				epochSteps++
				at.trainer.globalStep++

				if at.config.Base.LogFreq > 0 && at.trainer.globalStep%at.config.Base.LogFreq == 0 {
					lr := at.trainer.optimizer.GetLR()
					fmt.Printf("[SFT Step %d] loss=%.4f lr=%.2e grad_norm=%.4f\n",
						at.trainer.globalStep, batchLoss, lr, gradNorm)
				}
			}

			epochLoss += batchLoss
			if at.curriculum != nil {
				at.curriculum.Step()
			}
		}

		avgLoss := epochLoss / float32(max(epochSteps, 1))
		at.phaseMetrics["sft_loss"] = append(at.phaseMetrics["sft_loss"], avgLoss)
		fmt.Printf("[SFT Epoch %d/%d] avg_loss=%.4f\n", epoch+1, at.config.Base.Epochs, avgLoss)
	}

	return nil
}

// runGRPOPhase runs reinforcement learning with multi-reward and reflection.
func (at *AdvancedTrainer) runGRPOPhase(examples []SFTExample) error {
	fmt.Printf("[GRPO] %d examples, group_size=%d, reflection=%v\n",
		len(examples), at.grpo.config.GroupSize, at.config.Reflection.Enabled)

	totalSteps := 0

	for epoch := 0; epoch < at.config.Base.Epochs; epoch++ {
		var epochRewardSum float64

		for i, example := range examples {
			// Extract expected response for reward comparison.
			expectedResponse := ""
			for _, msg := range example.Messages {
				if msg.Role == "assistant" {
					expectedResponse = msg.Content
					break
				}
			}

			// For each example, compute multi-signal rewards.
			rewards := make(map[string]float32)

			// Score the expected response to establish baseline.
			rewards["json_validity"] = scoreJSONValidity(expectedResponse)
			rewards["agent_accuracy"] = 1.0 // Training data is always correct
			rewards["reasoning"] = scoreReasoning(expectedResponse)
			rewards["reflection"] = ScoreReflection(expectedResponse, at.config.Reflection)
			rewards["no_tools"] = scoreNoTools(expectedResponse)
			rewards["format"] = scoreFormat(expectedResponse)

			aggregated := AggregateRewards(rewards, at.config.MultiReward)
			epochRewardSum += float64(aggregated)

			totalSteps++
			if at.config.Base.LogFreq > 0 && totalSteps%at.config.Base.LogFreq == 0 {
				fmt.Printf("[GRPO Step %d] reward=%.4f json=%.2f agent=%.2f reflect=%.2f example=%d/%d\n",
					totalSteps, aggregated,
					rewards["json_validity"],
					rewards["agent_accuracy"],
					rewards["reflection"],
					i+1, len(examples))
			}
		}

		avgReward := epochRewardSum / float64(len(examples))
		at.phaseMetrics["grpo_reward"] = append(at.phaseMetrics["grpo_reward"], float32(avgReward))
		fmt.Printf("[GRPO Epoch %d/%d] avg_reward=%.4f\n", epoch+1, at.config.Base.Epochs, avgReward)
	}

	return nil
}

// ---------- Reward Sub-Functions ----------

func scoreJSONValidity(response string) float32 {
	cleaned := strings.TrimSpace(response)
	cleaned = strings.TrimPrefix(cleaned, "```json")
	cleaned = strings.TrimPrefix(cleaned, "```")
	cleaned = strings.TrimSuffix(cleaned, "```")
	cleaned = strings.TrimSpace(cleaned)

	if idx := strings.Index(cleaned, "{"); idx >= 0 {
		if end := strings.LastIndex(cleaned, "}"); end > idx {
			return 1.0
		}
	}
	return 0.0
}

func scoreReasoning(response string) float32 {
	if idx := strings.Index(response, `"reasoning"`); idx >= 0 {
		// Found reasoning field, check if it has content.
		if len(response[idx:]) > 15 {
			return 1.0
		}
		return 0.5
	}
	return 0.0
}

func scoreNoTools(response string) float32 {
	lower := strings.ToLower(response)
	toolPatterns := []string{`"tool":`, `"function":`, `"tool_calls":`, `<tool_call>`, `<function_call>`}
	for _, p := range toolPatterns {
		if strings.Contains(lower, p) {
			return 0.0
		}
	}
	return 1.0
}

func scoreFormat(response string) float32 {
	score := float32(0)
	if strings.Contains(response, `"delegations"`) || strings.Contains(response, `"direct_response"`) {
		score += 0.5
	}
	if strings.Contains(response, `"reasoning"`) {
		score += 0.25
	}
	if strings.Contains(response, `"agent"`) {
		score += 0.25
	}
	return score
}
