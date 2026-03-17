// model_warmup.go implements pre-computation of common prefixes on startup.
//
// WHAT: After loading a model, the first inference request is slow because
// the KV cache is cold. Model warmup runs dummy inference at startup to:
//   1. Populate the KV cache with common system prompts
//   2. JIT compile any lazy-initialized GPU kernels
//   3. Verify the model loads and runs correctly
//   4. Establish baseline performance numbers
//
// RESULT: First real request gets near-peak performance instead of cold-start latency.
package torch

import (
	"context"
	"fmt"
	"time"
)

// WarmupConfig controls startup warmup behavior.
type WarmupConfig struct {
	Enabled       bool
	SystemPrompts []string      // Common system prompts to pre-cache
	WarmupTokens  int           // Tokens to generate per warmup (default: 16)
	MaxDuration   time.Duration // Max time for warmup phase (default: 30s)
}

// DefaultWarmupConfig returns recommended warmup settings.
func DefaultWarmupConfig() WarmupConfig {
	return WarmupConfig{
		Enabled:      true,
		WarmupTokens: 16,
		MaxDuration:  30 * time.Second,
		SystemPrompts: []string{
			"You are a helpful assistant.",
			"You are a helpful AI. Answer questions concisely.",
		},
	}
}

// WarmupResult captures warmup performance.
type WarmupResult struct {
	TotalDuration     time.Duration
	PromptsProcessed  int
	TokensGenerated   int
	AvgTokensPerSec   float64
	FirstTokenLatency time.Duration
	Errors            []string
}

// RunWarmup executes model warmup with the given engine.
func RunWarmup(ctx context.Context, engine Engine, config WarmupConfig) WarmupResult {
	if !config.Enabled {
		return WarmupResult{}
	}

	fmt.Println("[Warmup] Starting model warmup...")
	start := time.Now()
	result := WarmupResult{}

	timeoutCtx, cancel := context.WithTimeout(ctx, config.MaxDuration)
	defer cancel()

	for i, sysPrompt := range config.SystemPrompts {
		if timeoutCtx.Err() != nil {
			result.Errors = append(result.Errors,
				fmt.Sprintf("warmup timed out after %d prompts", i))
			break
		}

		promptStart := time.Now()

		messages := []ChatMessage{
			{Role: "system", Content: sysPrompt},
			{Role: "user", Content: "Hello"},
		}

		params := CompletionParams{
			MaxTokens:   config.WarmupTokens,
			Temperature: 0.0, // Deterministic for warmup.
		}

		_, err := engine.Complete(timeoutCtx, messages, params)
		promptDuration := time.Since(promptStart)

		if err != nil {
			result.Errors = append(result.Errors,
				fmt.Sprintf("prompt %d failed: %v", i, err))
			continue
		}

		result.PromptsProcessed++
		result.TokensGenerated += config.WarmupTokens

		if i == 0 {
			result.FirstTokenLatency = promptDuration
		}

		fmt.Printf("[Warmup] Prompt %d/%d done in %s\n",
			i+1, len(config.SystemPrompts), promptDuration.Round(time.Millisecond))
	}

	result.TotalDuration = time.Since(start)
	if result.TotalDuration.Seconds() > 0 {
		result.AvgTokensPerSec = float64(result.TokensGenerated) / result.TotalDuration.Seconds()
	}

	fmt.Printf("[Warmup] Complete: %d prompts, %d tokens, %.1f tok/s in %s\n",
		result.PromptsProcessed, result.TokensGenerated,
		result.AvgTokensPerSec, result.TotalDuration.Round(time.Millisecond))

	return result
}
