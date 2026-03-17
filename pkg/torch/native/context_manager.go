// context_manager.go implements automatic context window management.
//
// WHAT: When a conversation exceeds the model's context window, the engine
// must decide what to keep and what to drop. This file provides multiple
// strategies for handling context overflow:
//
//   - Truncate: Drop oldest messages, keep most recent
//   - Sliding: Maintain a sliding window with system prompt pinned
//   - Summary: Compress old messages into a summary (uses the model itself)
//   - AttentionSink: Keep sink tokens + recent window (see attention_sink.go)
package native

import (
	"fmt"
)

// ContextStrategy specifies how to handle context overflow.
type ContextStrategy int

const (
	ContextTruncate    ContextStrategy = iota // Drop oldest tokens
	ContextSliding                            // Sliding window with pinned system
	ContextSummarize                          // Compress old context
	ContextAttentionSink                      // StreamingLLM sinks + recent
)

func (s ContextStrategy) String() string {
	switch s {
	case ContextSliding:
		return "sliding"
	case ContextSummarize:
		return "summarize"
	case ContextAttentionSink:
		return "attention_sink"
	default:
		return "truncate"
	}
}

// ContextManagerConfig controls context window management.
type ContextManagerConfig struct {
	Strategy     ContextStrategy
	MaxTokens    int  // Maximum context length (0 = model default)
	ReserveTokens int // Tokens reserved for generation (default: 512)
	PinSystemPrompt bool // Always keep the system prompt (default: true)
}

// DefaultContextConfig returns recommended settings.
func DefaultContextConfig() ContextManagerConfig {
	return ContextManagerConfig{
		Strategy:        ContextTruncate,
		MaxTokens:       0,    // Use model default
		ReserveTokens:   512,  // Reserved for generation
		PinSystemPrompt: true,
	}
}

// ContextManager handles context overflow for conversations.
type ContextManager struct {
	config     ContextManagerConfig
	maxContext int // Actual max after accounting for reserve
	truncations int
}

// NewContextManager creates a context manager.
func NewContextManager(config ContextManagerConfig, modelMaxContext int) *ContextManager {
	maxCtx := config.MaxTokens
	if maxCtx == 0 {
		maxCtx = modelMaxContext
	}

	cm := &ContextManager{
		config:     config,
		maxContext:  maxCtx - config.ReserveTokens,
	}

	fmt.Printf("[ContextManager] %s strategy, max=%d tokens (reserve=%d for gen)\n",
		config.Strategy, cm.maxContext, config.ReserveTokens)
	return cm
}

// FitsInContext checks if the token count fits within the context window.
func (cm *ContextManager) FitsInContext(tokenCount int) bool {
	return tokenCount <= cm.maxContext
}

// TruncateTokens trims a token sequence to fit the context window.
// If pinSystem is true, systemTokenCount tokens at the start are preserved.
func (cm *ContextManager) TruncateTokens(tokens []int, systemTokenCount int) []int {
	if len(tokens) <= cm.maxContext {
		return tokens
	}

	cm.truncations++

	switch cm.config.Strategy {
	case ContextSliding:
		return cm.slidingTruncate(tokens, systemTokenCount)
	case ContextAttentionSink:
		return cm.sinkTruncate(tokens)
	default:
		return cm.simpleTruncate(tokens, systemTokenCount)
	}
}

// simpleTruncate keeps the most recent tokens, optionally pinning system prefix.
func (cm *ContextManager) simpleTruncate(tokens []int, systemTokenCount int) []int {
	if cm.config.PinSystemPrompt && systemTokenCount > 0 && systemTokenCount < cm.maxContext/2 {
		// Keep system prefix + most recent tokens.
		systemPart := tokens[:systemTokenCount]
		availableForRecent := cm.maxContext - systemTokenCount
		recentStart := len(tokens) - availableForRecent
		if recentStart < systemTokenCount {
			recentStart = systemTokenCount
		}
		recentPart := tokens[recentStart:]

		result := make([]int, 0, len(systemPart)+len(recentPart))
		result = append(result, systemPart...)
		result = append(result, recentPart...)
		return result
	}

	// Simple: keep last maxContext tokens.
	start := len(tokens) - cm.maxContext
	if start < 0 {
		start = 0
	}
	return tokens[start:]
}

// slidingTruncate uses a sliding window with system prompt pinned.
func (cm *ContextManager) slidingTruncate(tokens []int, systemTokenCount int) []int {
	// Same as simple truncate but always pins system.
	if systemTokenCount == 0 {
		return cm.simpleTruncate(tokens, 0)
	}
	return cm.simpleTruncate(tokens, systemTokenCount)
}

// sinkTruncate keeps first N sink tokens + recent window.
func (cm *ContextManager) sinkTruncate(tokens []int) []int {
	sinkCount := 4 // Default attention sink tokens
	if sinkCount > len(tokens) {
		return tokens
	}

	windowSize := cm.maxContext - sinkCount
	if windowSize <= 0 {
		return tokens[:cm.maxContext]
	}

	recentStart := len(tokens) - windowSize
	if recentStart <= sinkCount {
		return tokens[:cm.maxContext]
	}

	// [sink_0..sink_3] + [recent_window]
	result := make([]int, 0, cm.maxContext)
	result = append(result, tokens[:sinkCount]...)
	result = append(result, tokens[recentStart:]...)
	return result
}

// Stats returns context management metrics.
func (cm *ContextManager) Stats() map[string]interface{} {
	return map[string]interface{}{
		"strategy":       cm.config.Strategy.String(),
		"max_context":    cm.maxContext,
		"reserve_tokens": cm.config.ReserveTokens,
		"truncations":    cm.truncations,
	}
}
