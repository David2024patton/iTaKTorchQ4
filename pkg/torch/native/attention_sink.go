// attention_sink.go implements StreamingLLM-style attention sinks for
// infinite-length context handling.
//
// WHAT: In standard transformers, when the KV cache fills up, the model must
// stop or the oldest tokens must be evicted. But naive eviction breaks
// the model because the first few tokens accumulate disproportionate attention
// mass (the "attention sink" phenomenon).
//
// HOW: StreamingLLM (Xiao et al., 2023) showed that keeping the first 4 tokens
// (sink tokens) plus the most recent W tokens gives nearly identical
// perplexity to full attention, with O(W) memory instead of O(N).
//
// LAYOUT: [sink_0, sink_1, sink_2, sink_3, ..., recent_W-4, ..., recent_W]
//
// This enables infinite-length conversations and streaming without
// context window truncation.
package native

import (
	"fmt"
)

// AttentionSinkConfig controls the streaming attention behavior.
type AttentionSinkConfig struct {
	NumSinkTokens int  // Number of initial tokens to always keep (default: 4)
	WindowSize    int  // Size of the recent token window
	Enabled       bool
}

// DefaultAttentionSinkConfig returns recommended settings.
func DefaultAttentionSinkConfig() AttentionSinkConfig {
	return AttentionSinkConfig{
		NumSinkTokens: 4,
		WindowSize:    2048,
		Enabled:       true,
	}
}

// AttentionSinkManager manages the KV cache eviction for streaming attention.
type AttentionSinkManager struct {
	config      AttentionSinkConfig
	totalTokens int  // Total tokens seen so far
	evictions   int  // Number of eviction operations performed
}

// NewAttentionSinkManager creates a manager for the given config.
func NewAttentionSinkManager(config AttentionSinkConfig) *AttentionSinkManager {
	fmt.Printf("[AttentionSink] Enabled: %d sink tokens, window=%d\n",
		config.NumSinkTokens, config.WindowSize)
	return &AttentionSinkManager{config: config}
}

// ShouldEvict returns true if the KV cache needs eviction.
func (m *AttentionSinkManager) ShouldEvict(currentLen int) bool {
	maxLen := m.config.NumSinkTokens + m.config.WindowSize
	return currentLen > maxLen
}

// ComputeEvictionRange returns which positions to evict from the KV cache.
// Keeps: [0, NumSinkTokens) + [currentLen - WindowSize + NumSinkTokens, currentLen)
// Evicts: [NumSinkTokens, currentLen - WindowSize + NumSinkTokens)
func (m *AttentionSinkManager) ComputeEvictionRange(currentLen int) (evictStart, evictEnd int) {
	if !m.ShouldEvict(currentLen) {
		return 0, 0
	}

	sinkEnd := m.config.NumSinkTokens
	recentStart := currentLen - m.config.WindowSize + m.config.NumSinkTokens

	if recentStart <= sinkEnd {
		return 0, 0 // Nothing to evict
	}

	m.evictions++
	m.totalTokens = currentLen

	return sinkEnd, recentStart
}

// EvictKVCache performs the actual eviction on a flat KV cache slice.
// Removes middle tokens, keeping sinks at the start and recent tokens at the end.
func (m *AttentionSinkManager) EvictKVCache(cache []float32, seqLen, headDim int) []float32 {
	evictStart, evictEnd := m.ComputeEvictionRange(seqLen)
	if evictStart == 0 && evictEnd == 0 {
		return cache
	}

	// Calculate positions in the flat array.
	startIdx := evictStart * headDim
	endIdx := evictEnd * headDim

	// Build new cache: [0:startIdx] + [endIdx:]
	newLen := len(cache) - (endIdx - startIdx)
	result := make([]float32, newLen)

	copy(result[:startIdx], cache[:startIdx])
	copy(result[startIdx:], cache[endIdx:])

	return result
}

// RebuildPositionIDs generates corrected position IDs after eviction.
// Sink tokens keep positions [0, NumSinkTokens).
// Recent tokens get positions [NumSinkTokens, NumSinkTokens + recentCount).
func (m *AttentionSinkManager) RebuildPositionIDs(newLen int) []int {
	positions := make([]int, newLen)
	for i := 0; i < newLen; i++ {
		if i < m.config.NumSinkTokens {
			positions[i] = i // Sink positions stay fixed.
		} else {
			positions[i] = i // Recent positions are contiguous after sinks.
		}
	}
	return positions
}

// Stats returns eviction metrics.
func (m *AttentionSinkManager) Stats() map[string]interface{} {
	return map[string]interface{}{
		"sink_tokens": m.config.NumSinkTokens,
		"window_size": m.config.WindowSize,
		"evictions":   m.evictions,
		"total_seen":  m.totalTokens,
	}
}
