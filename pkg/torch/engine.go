package torch

import (
	"context"
	"fmt"
	"sync"
)

// Engine is the interface for any inference backend.
// The CGo llama.cpp backend will implement this, but so can a mock for testing.
type Engine interface {
	// Complete runs text completion and returns the generated text.
	Complete(ctx context.Context, messages []ChatMessage, params CompletionParams) (string, error)
	// GenerateTokens bypasses the tokenizer and runs inference directly on the given token IDs.
	GenerateTokens(ctx context.Context, inputTokens []int32, params CompletionParams) (string, error)
	// IsLoaded returns true if the engine is currently loaded in memory.
	IsLoaded() bool
	// Reload reloads the engine from disk if it was previously closed.
	Reload() error
	// ModelName returns the name of the currently loaded model.
	ModelName() string
	// GetStats returns engine performance stats.
	GetStats() EngineStats
	// SaveKVCache serializes the engine's memory state to disk.
	SaveKVCache(path string) error
	// LoadKVCache restores the engine's memory state from disk.
	LoadKVCache(path string) error
	// Close unloads the model and frees resources.
	Close() error
}

// MockEngine is a test/placeholder engine that returns canned responses.
// Used when CGo/llama.cpp is not available (like on Windows without MinGW).
type MockEngine struct {
	name string
	mu   sync.Mutex
}

// NewMockEngine creates a mock engine for testing without CGo.
func NewMockEngine(name string) *MockEngine {
	return &MockEngine{name: name}
}

func (m *MockEngine) Complete(ctx context.Context, messages []ChatMessage, params CompletionParams) (string, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	if len(messages) == 0 {
		return "", fmt.Errorf("no messages provided")
	}

	// Return a simple response based on the last user message.
	last := messages[len(messages)-1]
	return fmt.Sprintf("[iTaKTorch Mock / %s] Received: %q", m.name, last.Content), nil
}

func (m *MockEngine) GenerateTokens(ctx context.Context, inputTokens []int32, params CompletionParams) (string, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	return fmt.Sprintf("[iTaKTorch Mock / %s] Processed %d raw tokens", m.name, len(inputTokens)), nil
}

func (m *MockEngine) ModelName() string {
	return m.name
}

func (m *MockEngine) GetStats() EngineStats {
	return EngineStats{}
}

func (m *MockEngine) IsLoaded() bool {
	m.mu.Lock()
	defer m.mu.Unlock()
	return true
}

func (m *MockEngine) Reload() error {
	return nil
}

func (m *MockEngine) SaveKVCache(path string) error {
	return nil
}

func (m *MockEngine) LoadKVCache(path string) error {
	return nil
}

func (m *MockEngine) Close() error {
	return nil
}
