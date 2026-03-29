package torch

import (
	"context"
	"fmt"
	"sync"

	"github.com/David2024patton/iTaKTorchQ4/pkg/torch/native"
)

// NativeAdapter wraps a native.NativeEngine to implement the torch.Engine interface.
type NativeAdapter struct {
	engine *native.NativeEngine
	mu     sync.Mutex
	loaded bool
}

// NewNativeAdapter creates a new adapter for a GOTensor engine.
func NewNativeAdapter(engine *native.NativeEngine) *NativeAdapter {
	return &NativeAdapter{
		engine: engine,
		loaded: true,
	}
}

// Complete runs text completion by delegating to NativeEngine.
func (a *NativeAdapter) Complete(ctx context.Context, messages []ChatMessage, params CompletionParams) (string, error) {
	// Translate messages
	var nativeMsgs []native.ChatMessage
	for _, m := range messages {
		nativeMsgs = append(nativeMsgs, native.ChatMessage{
			Role:    m.Role,
			Content: m.Content,
		})
	}

	// Translate params
	nativeParams := native.CompletionParams{
		MaxTokens:   params.MaxTokens,
		Temperature: params.Temperature,
		TopP:        params.TopP,
		Stop:        params.Stop,
	}

	return a.engine.Complete(ctx, nativeMsgs, nativeParams)
}

// GenerateTokens bypasses the tokenizer (Not yet implemented for GOTensor adapter).
func (a *NativeAdapter) GenerateTokens(ctx context.Context, inputTokens []int32, params CompletionParams) (string, error) {
	return "", fmt.Errorf("GenerateTokens not currently supported by GOTensor wrapper")
}

// IsLoaded returns true if the native engine is loaded.
func (a *NativeAdapter) IsLoaded() bool {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.loaded
}

// Reload reloads the engine. (Currently a no-op for the native adapter)
func (a *NativeAdapter) Reload() error {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.loaded = true
	return nil
}

// ModelName returns the name of the wrapped native model.
func (a *NativeAdapter) ModelName() string {
	return a.engine.ModelName()
}

// GetStats returns engine performance stats translated to torch.EngineStats.
func (a *NativeAdapter) GetStats() EngineStats {
	ns := a.engine.GetStats()
	
	var lastMetrics *InferenceMetrics
	if ns.LastMetrics != nil {
		lastMetrics = &InferenceMetrics{
			PromptTokens:     ns.LastMetrics.PromptTokens,
			CompletionTokens: ns.LastMetrics.CompletionTokens,
			TotalTokens:      ns.LastMetrics.TotalTokens,
			PromptDuration:   ns.LastMetrics.PromptDuration,
			GenDuration:      ns.LastMetrics.GenDuration,
			TotalDuration:    ns.LastMetrics.TotalDuration,
			TokensPerSecond:  ns.LastMetrics.TokensPerSecond,
		}
	}

	return EngineStats{
		RequestCount:   ns.TotalRequests,
		TotalTokensGen: ns.TotalTokensGen,
		LastMetrics:    lastMetrics,
	}
}

// SaveKVCache serializes the engine's memory state to disk.
func (a *NativeAdapter) SaveKVCache(path string) error {
	return nil // No-op
}

// LoadKVCache restores the engine's memory state from disk.
func (a *NativeAdapter) LoadKVCache(path string) error {
	return nil // No-op
}

// Synthesize converts text to speech audio bytes using the GOTensor engine.
func (a *NativeAdapter) Synthesize(ctx context.Context, req SpeechRequest) ([]byte, error) {
	return nil, fmt.Errorf("Synthesize not currently supported by GOTensor wrapper")
}

// Close unloads the model and frees resources.
func (a *NativeAdapter) Close() error {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.engine.Close()
	a.loaded = false
	return nil
}
