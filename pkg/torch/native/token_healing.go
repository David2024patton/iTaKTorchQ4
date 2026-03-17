// token_healing.go implements token healing to fix broken tokens at prompt boundaries.
//
// WHAT: When a prompt ends mid-token (e.g., "The URL is https://ex" where
// "ex" might not be a valid BPE token), the model's next token prediction
// is degraded because the prompt boundary created an artificial token split.
//
// HOW: Token healing backs up to the last complete token boundary, then
// re-encodes the suffix along with the generated text. This produces
// better continuations, especially for:
//   - Code completion (variable names, string literals)
//   - URL completion
//   - Any structured text
//
// REFERENCE: "Efficient Guided Generation for Large Language Models" (2023)
package native

import (
	"strings"
	"unicode/utf8"
)

// TokenHealer repairs broken tokens at prompt boundaries.
type TokenHealer struct {
	tokenizer *BPETokenizer
}

// NewTokenHealer creates a healer using the given tokenizer.
func NewTokenHealer(tokenizer *BPETokenizer) *TokenHealer {
	return &TokenHealer{tokenizer: tokenizer}
}

// HealPrompt adjusts the prompt to end at a clean token boundary.
// Returns the adjusted prompt and the "rollback" text that should be
// prepended to the generated output.
func (h *TokenHealer) HealPrompt(prompt string) (healedPrompt string, rollback string) {
	if h.tokenizer == nil || prompt == "" {
		return prompt, ""
	}

	// Encode the full prompt.
	tokens := h.tokenizer.Encode(prompt)
	if len(tokens) == 0 {
		return prompt, ""
	}

	// Decode the last token back to text.
	lastToken := tokens[len(tokens)-1]
	lastTokenText := h.tokenizer.Decode([]int{lastToken})

	// Check if the prompt ends cleanly at a token boundary.
	// A "clean" boundary means the prompt text matches exactly what
	// encode-then-decode produces.
	reencoded := h.tokenizer.Decode(tokens)
	if reencoded == prompt {
		return prompt, "" // No healing needed.
	}

	// Back up one token: remove the last token and decode the rest.
	if len(tokens) > 1 {
		healedTokens := tokens[:len(tokens)-1]
		healedPrompt = h.tokenizer.Decode(healedTokens)
		rollback = lastTokenText
		return healedPrompt, rollback
	}

	return prompt, ""
}

// HealOutput applies the rollback to the generated output.
// If rollback is non-empty, the output should start with the rollback text,
// and we verify this by checking the prefix.
func (h *TokenHealer) HealOutput(output string, rollback string) string {
	if rollback == "" {
		return output
	}

	// The model should have generated text starting with the rollback prefix.
	// If it did, strip the rollback from the output (it's already in the prompt).
	if strings.HasPrefix(output, rollback) {
		return output[len(rollback):]
	}

	// If the model didn't generate the expected rollback, return as-is.
	return output
}

// FindTokenBoundary finds the byte position of the last clean token boundary
// in the text, within the last maxLookback bytes.
func (h *TokenHealer) FindTokenBoundary(text string, maxLookback int) int {
	if maxLookback <= 0 || maxLookback > len(text) {
		maxLookback = len(text)
	}

	// Walk backward finding valid UTF-8 character boundaries.
	pos := len(text)
	for i := 0; i < maxLookback && pos > 0; i++ {
		pos--
		for pos > 0 && !utf8.RuneStart(text[pos]) {
			pos--
		}

		// Check if the text up to this point encodes to clean tokens.
		prefix := text[:pos]
		tokens := h.tokenizer.Encode(prefix)
		decoded := h.tokenizer.Decode(tokens)
		if decoded == prefix {
			return pos
		}
	}

	return len(text) // No boundary found, don't heal.
}

// IsCleanBoundary checks if the text ends on a clean token boundary.
func (h *TokenHealer) IsCleanBoundary(text string) bool {
	if h.tokenizer == nil || text == "" {
		return true
	}
	tokens := h.tokenizer.Encode(text)
	decoded := h.tokenizer.Decode(tokens)
	return decoded == text
}
