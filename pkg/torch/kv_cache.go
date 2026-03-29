package torch

import (
	"fmt"

	"github.com/David2024patton/iTaKTorchQ4/pkg/torch/llama"
)

// SaveKVCache serializes the current context's Key-Value cache to disk.
// This allows for "Context Freezing", where an agent with a massive system prompt
// can save its state and resume instantly later without recalculating the prompt.
func (e *TorchEngine) SaveKVCache(path string) error {
	e.mu.Lock()
	defer e.mu.Unlock()

	if !e.loaded {
		return fmt.Errorf("engine not loaded")
	}

	// Check if state is essentially empty
	size := llama.StateGetSize(e.ctx)
	if size == 0 {
		return fmt.Errorf("failed to get state size or state empty")
	}

	// We use the StateSaveFile binding which writes directly to disk.
	// We pass nil for tokens because we are only freezing the KV cache state,
	// the prompt string/messages history should be maintained by the caller application.
	success := llama.StateSaveFile(e.ctx, path, nil)
	if !success {
		return fmt.Errorf("failed to save KV cache to %s", path)
	}

	return nil
}

// LoadKVCache loads a serialized Key-Value cache from disk into the current context.
func (e *TorchEngine) LoadKVCache(path string) error {
	e.mu.Lock()
	defer e.mu.Unlock()

	if !e.loaded {
		return fmt.Errorf("engine not loaded")
	}

	var tokenCountOut uint64
	success := llama.StateLoadFile(e.ctx, path, nil, 0, &tokenCountOut)
	if !success {
		return fmt.Errorf("failed to load KV cache from %s", path)
	}

	return nil
}
