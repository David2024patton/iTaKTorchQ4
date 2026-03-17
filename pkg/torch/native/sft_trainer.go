// sft_trainer.go implements supervised fine-tuning with proper chat formatting.
//
// WHAT: SFT (Supervised Fine-Tuning) is the first step in LLM alignment.
// You take a pre-trained base model and fine-tune it on instruction-response
// pairs with proper chat template formatting.
//
// WHY: A base model just does next-token prediction. SFT teaches it to follow
// instructions. Combined with LoRA, this is the most common way to customize
// a model for a specific use case.
//
// FEATURES:
//   - Multi-turn conversation support (system + user + assistant turns)
//   - Loss masking: only compute loss on assistant responses, not prompts
//   - Packing: fit multiple short conversations into one training sequence
//   - Chat template rendering with configurable format
package native

import (
	"fmt"
	"strings"
)

// SFTMessage represents one turn in a conversation for SFT training.
// Separate from the ChatMessage in engine.go which has JSON tags for the API.
type SFTMessage struct {
	Role    string // "system", "user", "assistant"
	Content string
}

// SFTExample is one training example: a multi-turn conversation.
type SFTExample struct {
	Messages []SFTMessage
}

// ChatFormat defines the token format for a chat template.
type ChatFormat struct {
	Name           string
	BOS            string // Beginning of sequence token
	EOS            string // End of sequence token
	SystemStart    string
	SystemEnd      string
	UserStart      string
	UserEnd        string
	AssistantStart string
	AssistantEnd   string
}

// Common chat formats used by popular models.
var (
	ChatMLFormat = ChatFormat{
		Name:           "chatml",
		BOS:            "<|im_start|>",
		EOS:            "<|im_end|>",
		SystemStart:    "<|im_start|>system\n",
		SystemEnd:      "<|im_end|>\n",
		UserStart:      "<|im_start|>user\n",
		UserEnd:        "<|im_end|>\n",
		AssistantStart: "<|im_start|>assistant\n",
		AssistantEnd:   "<|im_end|>\n",
	}

	LlamaFormat = ChatFormat{
		Name:           "llama",
		BOS:            "<s>",
		EOS:            "</s>",
		SystemStart:    "[INST] <<SYS>>\n",
		SystemEnd:      "\n<</SYS>>\n\n",
		UserStart:      "[INST] ",
		UserEnd:        " [/INST]",
		AssistantStart: "",
		AssistantEnd:   "</s>",
	}

	MistralFormat = ChatFormat{
		Name:           "mistral",
		BOS:            "<s>",
		EOS:            "</s>",
		SystemStart:    "[INST] ",
		SystemEnd:      "\n",
		UserStart:      "[INST] ",
		UserEnd:        " [/INST]",
		AssistantStart: "",
		AssistantEnd:   "</s>",
	}
)

// SFTConfig configures supervised fine-tuning.
type SFTConfig struct {
	Format        ChatFormat
	MaxSeqLen     int     // Maximum sequence length
	MaskPrompts   bool    // Only compute loss on assistant turns (default: true)
	PackSequences bool    // Pack multiple short examples into one sequence
	LossScale     float32 // Scale factor for the loss
}

// DefaultSFTConfig returns standard SFT settings.
func DefaultSFTConfig() SFTConfig {
	return SFTConfig{
		Format:      ChatMLFormat,
		MaxSeqLen:   2048,
		MaskPrompts: true,
		LossScale:   1.0,
	}
}

// SFTTrainer manages supervised fine-tuning.
type SFTTrainer struct {
	config SFTConfig

	// Stats.
	totalSteps   int64
	totalTokens  int64
	avgLoss      float64
}

// NewSFTTrainer creates an SFT trainer.
func NewSFTTrainer(config SFTConfig) *SFTTrainer {
	return &SFTTrainer{config: config}
}

// FormatConversation renders a multi-turn conversation into a tokenizable string.
// Returns the formatted text and a mask indicating which characters are assistant
// responses (for loss masking).
func (t *SFTTrainer) FormatConversation(example SFTExample) (string, []bool) {
	var builder strings.Builder
	var mask []bool

	for _, msg := range example.Messages {
		switch msg.Role {
		case "system":
			text := t.config.Format.SystemStart + msg.Content + t.config.Format.SystemEnd
			builder.WriteString(text)
			// System prompt is not trained on.
			for range text {
				mask = append(mask, false)
			}

		case "user":
			text := t.config.Format.UserStart + msg.Content + t.config.Format.UserEnd
			builder.WriteString(text)
			// User turns are not trained on.
			for range text {
				mask = append(mask, false)
			}

		case "assistant":
			prefix := t.config.Format.AssistantStart
			builder.WriteString(prefix)
			for range prefix {
				mask = append(mask, false)
			}

			// Assistant content IS trained on.
			builder.WriteString(msg.Content)
			for range msg.Content {
				mask = append(mask, true)
			}

			suffix := t.config.Format.AssistantEnd
			builder.WriteString(suffix)
			for range suffix {
				mask = append(mask, !t.config.MaskPrompts)
			}
		}
	}

	return builder.String(), mask
}

// PackExamples packs multiple short conversations into a single sequence
// up to maxSeqLen, separated by EOS tokens. This improves GPU utilization
// by avoiding padding waste.
func (t *SFTTrainer) PackExamples(examples []SFTExample, tokenize func(string) []int32) ([]int32, []bool) {
	var packedTokens []int32
	var packedMask []bool
	maxLen := t.config.MaxSeqLen

	for _, ex := range examples {
		text, charMask := t.FormatConversation(ex)
		tokens := tokenize(text)

		// Check if this example fits.
		if len(packedTokens)+len(tokens) > maxLen {
			break
		}

		// Map character mask to token mask (approximate: 1 token ~ 4 chars).
		tokenMask := make([]bool, len(tokens))
		charsPerToken := len(charMask) / (len(tokens) + 1)
		if charsPerToken < 1 {
			charsPerToken = 1
		}
		for i := range tokenMask {
			charIdx := i * charsPerToken
			if charIdx < len(charMask) {
				tokenMask[i] = charMask[charIdx]
			}
		}

		packedTokens = append(packedTokens, tokens...)
		packedMask = append(packedMask, tokenMask...)
	}

	return packedTokens, packedMask
}

// ComputeLoss computes masked cross-entropy loss.
// Only tokens where mask[i] == true contribute to the loss.
// logits: [seqLen][vocabSize], targets: [seqLen], mask: [seqLen].
func (t *SFTTrainer) ComputeLoss(logits [][]float32, targets []int32, mask []bool) float32 {
	var totalLoss float64
	var count int

	for pos := range targets {
		if pos >= len(mask) || !mask[pos] {
			continue // Skip masked positions (prompts).
		}

		target := int(targets[pos])
		if target >= len(logits[pos]) {
			continue
		}

		// Cross-entropy at this position.
		loss, _ := FusedCrossEntropyBackward(logits[pos], target)
		totalLoss += float64(loss)
		count++
	}

	if count == 0 {
		return 0
	}

	avgLoss := float32(totalLoss / float64(count))

	// Update stats.
	t.totalSteps++
	t.totalTokens += int64(count)
	t.avgLoss = t.avgLoss*0.99 + float64(avgLoss)*0.01

	return avgLoss * t.config.LossScale
}

// Stats returns SFT training metrics.
func (t *SFTTrainer) Stats() map[string]interface{} {
	return map[string]interface{}{
		"total_steps":  t.totalSteps,
		"total_tokens": t.totalTokens,
		"avg_loss":     fmt.Sprintf("%.4f", t.avgLoss),
		"format":       t.config.Format.Name,
		"mask_prompts": t.config.MaskPrompts,
	}
}
