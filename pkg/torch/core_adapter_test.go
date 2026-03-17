package torch

import (
	"context"
	"testing"

	"github.com/David2024patton/iTaKCore/pkg/types"
)

func TestCoreAdapterEngineID(t *testing.T) {
	mock := NewMockEngine("test-model")
	adapter := NewTorchAdapter(mock, nil, "vulkan")

	id := adapter.EngineID()
	if id != "torch-vulkan" {
		t.Errorf("EngineID() = %q, want %q", id, "torch-vulkan")
	}

	// Default backend.
	adapter2 := NewTorchAdapter(mock, nil, "")
	if adapter2.EngineID() != "torch-cpu" {
		t.Errorf("EngineID() with empty backend = %q, want %q", adapter2.EngineID(), "torch-cpu")
	}
}

func TestCoreAdapterInfer(t *testing.T) {
	mock := NewMockEngine("qwen3-8b")
	adapter := NewTorchAdapter(mock, nil, "cpu")

	req := types.InferenceRequest{
		Model: "qwen3-8b",
		Messages: []types.ChatMessage{
			{Role: types.RoleUser, Content: "Hello"},
		},
		Temperature: 0.7,
		MaxTokens:   100,
	}

	resp, err := adapter.Infer(context.Background(), req)
	if err != nil {
		t.Fatalf("Infer() error: %v", err)
	}

	if resp.Message.Role != types.RoleAssistant {
		t.Errorf("Message.Role = %q, want %q", resp.Message.Role, types.RoleAssistant)
	}
	if resp.Message.Content == "" {
		t.Error("Message.Content is empty")
	}
	if !resp.Done {
		t.Error("Done should be true for non-streaming")
	}
	if resp.Model != "qwen3-8b" {
		t.Errorf("Model = %q, want %q", resp.Model, "qwen3-8b")
	}
}

func TestCoreAdapterListModelsSingleMode(t *testing.T) {
	mock := NewMockEngine("test-model")
	adapter := NewTorchAdapter(mock, nil, "cuda")

	models, err := adapter.ListModels(context.Background())
	if err != nil {
		t.Fatalf("ListModels() error: %v", err)
	}
	if len(models) != 1 {
		t.Fatalf("ListModels() returned %d models, want 1", len(models))
	}
	if models[0].ID != "test-model" {
		t.Errorf("Model ID = %q, want %q", models[0].ID, "test-model")
	}
	if models[0].Backend != "cuda" {
		t.Errorf("Backend = %q, want %q", models[0].Backend, "cuda")
	}
}

func TestCoreAdapterLoadModelSingleMode(t *testing.T) {
	mock := NewMockEngine("loaded-model")
	adapter := NewTorchAdapter(mock, nil, "cpu")

	// Loading the already-loaded model should be a no-op.
	err := adapter.LoadModel(context.Background(), "loaded-model", types.ModelParams{})
	if err != nil {
		t.Errorf("LoadModel() for loaded model should not error, got: %v", err)
	}

	// Loading a different model in single-model mode should fail.
	err = adapter.LoadModel(context.Background(), "other-model", types.ModelParams{})
	if err == nil {
		t.Error("LoadModel() for different model in single-model mode should error")
	}
}

func TestCoreAdapterUnloadModelSingleMode(t *testing.T) {
	mock := NewMockEngine("test")
	adapter := NewTorchAdapter(mock, nil, "cpu")

	err := adapter.UnloadModel(context.Background(), "test")
	if err == nil {
		t.Error("UnloadModel() in single-model mode should error")
	}
}

func TestCoreMsgToTorch(t *testing.T) {
	coreMsg := types.ChatMessage{
		Role:    types.RoleUser,
		Content: "What is Go?",
	}

	torchMsg := coreMsgToTorch(coreMsg)
	if torchMsg.Role != "user" {
		t.Errorf("Role = %q, want %q", torchMsg.Role, "user")
	}
	if torchMsg.Content != "What is Go?" {
		t.Errorf("Content = %q, want %q", torchMsg.Content, "What is Go?")
	}
}

func TestTorchMsgToCore(t *testing.T) {
	torchMsg := ChatMessage{
		Role:    "assistant",
		Content: "Go is a programming language.",
	}

	coreMsg := torchMsgToCore(torchMsg)
	if coreMsg.Role != types.RoleAssistant {
		t.Errorf("Role = %q, want %q", coreMsg.Role, types.RoleAssistant)
	}
	if coreMsg.Content != "Go is a programming language." {
		t.Errorf("Content = %q, want %q", coreMsg.Content, "Go is a programming language.")
	}
}

func TestTorchModelToCore(t *testing.T) {
	tests := []struct {
		name      string
		input     ModelInfo
		wantParams int64
		wantQuant  string
		wantFamily string
	}{
		{
			name:       "qwen3 8b q4_k_m",
			input:      ModelInfo{ID: "qwen3-8b-instruct-q4_k_m"},
			wantParams: 8_000_000_000,
			wantQuant:  "Q4_K_M",
			wantFamily: "qwen",
		},
		{
			name:       "llama 70b fp16",
			input:      ModelInfo{ID: "llama-3.1-70b-fp16"},
			wantParams: 70_000_000_000,
			wantQuant:  "FP16",
			wantFamily: "llama",
		},
		{
			name:       "unknown model",
			input:      ModelInfo{ID: "custom-model"},
			wantParams: 0,
			wantQuant:  "",
			wantFamily: "",
		},
		{
			name:       "gemma 3b q8_0",
			input:      ModelInfo{ID: "gemma-3b-it-q8_0"},
			wantParams: 3_000_000_000,
			wantQuant:  "Q8_0",
			wantFamily: "gemma",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := torchModelToCore(tt.input)
			if result.Params != tt.wantParams {
				t.Errorf("Params = %d, want %d", result.Params, tt.wantParams)
			}
			if result.Quantized != tt.wantQuant {
				t.Errorf("Quantized = %q, want %q", result.Quantized, tt.wantQuant)
			}
			if result.Family != tt.wantFamily {
				t.Errorf("Family = %q, want %q", result.Family, tt.wantFamily)
			}
		})
	}
}
