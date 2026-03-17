package native

import (
	"context"
	"fmt"
	"time"
)

// SpeculativeDecoder wraps two NativeEngines (a large Target model and a small Draft model)
// to perform speculative decoding, significantly accelerating inference.
//
// How it works:
// 1. The small Draft model quickly generates N candidate tokens.
// 2. The large Target model runs a single batched forward pass on those N tokens.
// 3. We compare the Target's predicted probabilities for each token against the Draft's choices.
// 4. We accept the longest matching prefix of drafted tokens.
// 5. We output the accepted tokens + 1 new token guaranteed by the Target model.
type SpeculativeDecoder struct {
	Target *NativeEngine
	Draft  *NativeEngine
}

// NewSpeculativeDecoder creates a decoder that uses a small model to draft tokens
// for a larger, more accurate model.
func NewSpeculativeDecoder(target, draft *NativeEngine) *SpeculativeDecoder {
	return &SpeculativeDecoder{
		Target: target,
		Draft:  draft,
	}
}

// ModelName returns a composite name showing both models.
func (sd *SpeculativeDecoder) ModelName() string {
	return fmt.Sprintf("%s (Draft: %s)", sd.Target.ModelName(), sd.Draft.ModelName())
}

// Complete runs speculative inference on the given messages.
func (sd *SpeculativeDecoder) Complete(ctx context.Context, messages []ChatMessage, params CompletionParams) (string, error) {
	sd.Target.mu.Lock()
	defer sd.Target.mu.Unlock()

	sd.Draft.mu.Lock()
	defer sd.Draft.mu.Unlock()

	start := time.Now()

	// Tokenize prompt (using the Target's vocab limits for simplicity)
	prompt := ""
	for _, m := range messages {
		prompt += m.Content + " "
	}
	tokens := simpleTokenize(prompt, sd.Target.vocabSize)

	maxTokens := params.MaxTokens
	if maxTokens <= 0 {
		maxTokens = 64
	}

	promptDuration := time.Since(start)
	genStart := time.Now()

	var generated []int
	currentContext := make([]int, len(tokens))
	copy(currentContext, tokens)

	draftTokensCount := 3 // The number of speculative tokens to guess ahead (Gamma)

	// Keep going until we hit MaxTokens
	for len(generated) < maxTokens {
		select {
		case <-ctx.Done():
			return tokensToText(generated), ctx.Err()
		default:
		}

		// 1. DRAFT PHASE: Ask the small model to guess the next N tokens quickly.
		drafts := sd.draftAhead(currentContext, draftTokensCount)
		
		// 2. VERIFY PHASE: Pass the context + drafted tokens through the large model.
		// In a real optimized engine, this is a single batched forward pass.
		accepted, nextRequired := sd.verifyDrafts(currentContext, drafts)

		// 3. COMMIT PHASE: Append the accepted drafts + the 1 guaranteed token.
		for _, tok := range accepted {
			generated = append(generated, tok)
			currentContext = append(currentContext, tok)
		}
		
		if len(generated) < maxTokens {
			generated = append(generated, nextRequired)
			currentContext = append(currentContext, nextRequired)
		}
	}

	// Truncate if we overshot MaxTokens
	if len(generated) > maxTokens {
		generated = generated[:maxTokens]
	}

	genDuration := time.Since(genStart)
	
	tokPerSec := 0.0
	if genDuration.Seconds() > 0 {
		tokPerSec = float64(len(generated)) / genDuration.Seconds()
	}

	// Update stats on the Target model
	sd.Target.stats.TotalRequests++
	sd.Target.stats.TotalTokensGen += int64(len(generated))
	sd.Target.stats.LastMetrics = &InferenceMetrics{
		PromptTokens:     len(tokens),
		CompletionTokens: len(generated),
		TotalTokens:      len(tokens) + len(generated),
		PromptDuration:   promptDuration,
		GenDuration:      genDuration,
		TotalDuration:    time.Since(start),
		TokensPerSecond:  tokPerSec,
	}

	return tokensToText(generated), nil
}

// draftAhead uses the small Draft model to autoregressively guess N tokens.
func (sd *SpeculativeDecoder) draftAhead(context []int, numDrafts int) []int {
	drafts := make([]int, 0, numDrafts)
	
	// Start with the current context
	currentCtx := make([]int, len(context))
	copy(currentCtx, context)

	for i := 0; i < numDrafts; i++ {
		// In our simple engine, we re-run the whole context + new draft tokens.
		// (A real engine uses KV caching to only compute the latest column).
		logits := sd.Draft.forward(currentCtx)
		bestTok := argmax(logits)
		
		drafts = append(drafts, bestTok)
		currentCtx = append(currentCtx, bestTok)
	}

	return drafts
}

// verifyDrafts uses the large Target model to check the drafted sequence.
// Returns the accepted prefix of draft tokens, and the single ground-truth next token 
// that should follow that accepted prefix.
func (sd *SpeculativeDecoder) verifyDrafts(context, drafts []int) ([]int, int) {
	accepted := make([]int, 0, len(drafts))
	
	// We need to evaluate the target model at each step to see if it agrees.
	// Normally this is one batched pass: target(context + drafts) -> sequence of logits.
	// Since our NativeEngine API only exposes single-token generation via forward(), 
	// we simulate the batched check by running iteratively. 
	
	currentCtx := make([]int, len(context))
	copy(currentCtx, context)

	var nextTargetTok int

	for _, draftTok := range drafts {
		// What does the large model think the NEXT token should be, given the context so far?
		tgtLogits := sd.Target.forward(currentCtx)
		nextTargetTok = argmax(tgtLogits)

		// Did the draft model guess correctly?
		if nextTargetTok == draftTok {
			// Yes! Accept it and move forward.
			accepted = append(accepted, draftTok)
			currentCtx = append(currentCtx, draftTok)
		} else {
			// No. The chain is broken. We must stop here.
			// Return what we accepted so far, PLUS the token the Target model actually wanted.
			return accepted, nextTargetTok 
		}
	}

	// If we accept ALL drafts, we still need to generate ONE more token from the Target
	// to continue the chain.
	tgtLogits := sd.Target.forward(currentCtx)
	nextTargetTok = argmax(tgtLogits)

	return accepted, nextTargetTok
}

// GetStats delegates to the Target model.
func (sd *SpeculativeDecoder) GetStats() EngineStats {
	return sd.Target.GetStats()
}

// Close releases both engines.
func (sd *SpeculativeDecoder) Close() {
	sd.Target.Close()
	sd.Draft.Close()
}
