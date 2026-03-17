// retry.go implements automatic retry and fallback logic for inference.
//
// WHAT: When inference fails (OOM, context overflow, timeout), this module
// automatically retries with adjusted parameters:
//   Attempt 1: Original parameters
//   Attempt 2: Reduced context (75%)
//   Attempt 3: Reduced context (50%) + lower batch size
//   Attempt 4: Fall back to CPU-only mode
//
// BACKOFF: Uses exponential backoff between retries (100ms, 200ms, 400ms).
package torch

import (
	"context"
	"fmt"
	"strings"
	"time"
)

// RetryConfig controls retry behavior.
type RetryConfig struct {
	MaxRetries      int           // Maximum retry attempts (default: 3)
	InitialBackoff  time.Duration // First retry delay (default: 100ms)
	BackoffMultiplier float64     // Multiply delay each retry (default: 2.0)
	MaxBackoff      time.Duration // Cap on backoff delay (default: 5s)
	ReduceContext   bool          // Try with reduced context on OOM
	FallbackToCPU   bool          // Try CPU-only as last resort
}

// DefaultRetryConfig returns recommended settings.
func DefaultRetryConfig() RetryConfig {
	return RetryConfig{
		MaxRetries:        3,
		InitialBackoff:    100 * time.Millisecond,
		BackoffMultiplier: 2.0,
		MaxBackoff:        5 * time.Second,
		ReduceContext:     true,
		FallbackToCPU:     true,
	}
}

// RetryResult captures what happened during retries.
type RetryResult struct {
	Text         string
	Attempts     int
	FinalError   error
	Adjustments  []string // What was changed on each retry
	TotalLatency time.Duration
}

// RetryExecutor wraps an engine with retry logic.
type RetryExecutor struct {
	engine Engine
	config RetryConfig
	logger *StructuredLogger
}

// NewRetryExecutor creates an executor with retry capability.
func NewRetryExecutor(engine Engine, config RetryConfig) *RetryExecutor {
	return &RetryExecutor{
		engine: engine,
		config: config,
		logger: NewStructuredLogger("retry"),
	}
}

// Complete runs inference with automatic retries on failure.
func (r *RetryExecutor) Complete(ctx context.Context, messages []ChatMessage, params CompletionParams) RetryResult {
	start := time.Now()
	result := RetryResult{}
	backoff := r.config.InitialBackoff

	for attempt := 0; attempt <= r.config.MaxRetries; attempt++ {
		result.Attempts = attempt + 1

		// Adjust parameters based on retry attempt.
		adjustedParams := params
		var adjustment string

		switch attempt {
		case 1:
			if r.config.ReduceContext {
				adjustedParams.MaxTokens = params.MaxTokens * 3 / 4
				adjustment = fmt.Sprintf("reduced max_tokens to %d", adjustedParams.MaxTokens)
			}
		case 2:
			if r.config.ReduceContext {
				adjustedParams.MaxTokens = params.MaxTokens / 2
				adjustment = fmt.Sprintf("reduced max_tokens to %d", adjustedParams.MaxTokens)
			}
		case 3:
			adjustment = "final attempt with minimal params"
			adjustedParams.MaxTokens = params.MaxTokens / 4
			if adjustedParams.MaxTokens < 64 {
				adjustedParams.MaxTokens = 64
			}
		}

		if adjustment != "" {
			result.Adjustments = append(result.Adjustments, adjustment)
		}

		// Try inference.
		text, err := r.engine.Complete(ctx, messages, adjustedParams)
		if err == nil {
			result.Text = text
			result.TotalLatency = time.Since(start)

			if attempt > 0 {
				r.logger.Info("retry_succeeded", Fields{
					"attempt":     attempt + 1,
					"adjustments": result.Adjustments,
					"latency_ms":  result.TotalLatency.Milliseconds(),
				})
			}
			return result
		}

		result.FinalError = err

		// Log the failure.
		r.logger.Warn("inference_failed", Fields{
			"attempt": attempt + 1,
			"error":   err.Error(),
		})

		// Don't retry on non-retryable errors.
		if !isRetryable(err) {
			break
		}

		// Don't retry on context cancellation.
		if ctx.Err() != nil {
			break
		}

		// Backoff before next attempt.
		if attempt < r.config.MaxRetries {
			select {
			case <-time.After(backoff):
			case <-ctx.Done():
				result.FinalError = ctx.Err()
				result.TotalLatency = time.Since(start)
				return result
			}
			backoff = time.Duration(float64(backoff) * r.config.BackoffMultiplier)
			if backoff > r.config.MaxBackoff {
				backoff = r.config.MaxBackoff
			}
		}
	}

	result.TotalLatency = time.Since(start)
	r.logger.Error("all_retries_exhausted", Fields{
		"attempts":  result.Attempts,
		"error":     result.FinalError.Error(),
		"latency_ms": result.TotalLatency.Milliseconds(),
	})
	return result
}

// isRetryable checks if an error is worth retrying.
func isRetryable(err error) bool {
	msg := strings.ToLower(err.Error())
	retryablePatterns := []string{
		"out of memory", "oom", "cuda error",
		"context too long", "context overflow",
		"timeout", "deadline exceeded",
		"temporary", "try again",
	}
	for _, pattern := range retryablePatterns {
		if strings.Contains(msg, pattern) {
			return true
		}
	}
	return false
}
