package torch

import (
	"context"
	"testing"
)

func TestMockEngineComplete(t *testing.T) {
	engine := NewMockEngine("test-model")

	messages := []ChatMessage{
		{Role: "user", Content: "What is 2+2?"},
	}
	result, err := engine.Complete(context.Background(), messages, CompletionParams{})
	if err != nil {
		t.Fatalf("Complete error: %v", err)
	}
	if result == "" {
		t.Error("expected non-empty result")
	}
	t.Logf("MockEngine result: %s", result)
}

func TestMockEngineEmptyMessages(t *testing.T) {
	engine := NewMockEngine("test-model")

	_, err := engine.Complete(context.Background(), []ChatMessage{}, CompletionParams{})
	if err == nil {
		t.Error("expected error for empty messages")
	}
}

func TestMockEngineModelName(t *testing.T) {
	engine := NewMockEngine("qwen3-0.6b")
	if engine.ModelName() != "qwen3-0.6b" {
		t.Errorf("ModelName() = %q, want %q", engine.ModelName(), "qwen3-0.6b")
	}
}

func TestMockEngineClose(t *testing.T) {
	engine := NewMockEngine("test")
	if err := engine.Close(); err != nil {
		t.Errorf("Close() error: %v", err)
	}
}
