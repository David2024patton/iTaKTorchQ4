package torch

import (
	"encoding/json"
	"testing"
)

func TestChatRequestJSON(t *testing.T) {
	raw := `{"model":"test","messages":[{"role":"user","content":"hello"}],"temperature":0.5}`
	var req ChatRequest
	if err := json.Unmarshal([]byte(raw), &req); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if req.Model != "test" {
		t.Errorf("model = %q, want %q", req.Model, "test")
	}
	if len(req.Messages) != 1 {
		t.Fatalf("messages len = %d, want 1", len(req.Messages))
	}
	if req.Messages[0].Role != "user" {
		t.Errorf("role = %q, want %q", req.Messages[0].Role, "user")
	}
	if req.Temperature == nil || *req.Temperature != 0.5 {
		t.Error("temperature should be 0.5")
	}
}

func TestChatResponseJSON(t *testing.T) {
	resp := ChatResponse{
		ID:      "itaktorch-123",
		Object:  "chat.completion",
		Created: 1000,
		Model:   "test-model",
		Choices: []ChatChoice{
			{
				Index:        0,
				Message:      ChatMessage{Role: "assistant", Content: "hello back"},
				FinishReason: "stop",
			},
		},
		Usage: ChatUsage{
			PromptTokens:     10,
			CompletionTokens: 5,
			TotalTokens:      15,
		},
	}

	data, err := json.Marshal(resp)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}

	var parsed ChatResponse
	if err := json.Unmarshal(data, &parsed); err != nil {
		t.Fatalf("unmarshal roundtrip: %v", err)
	}
	if parsed.ID != "itaktorch-123" {
		t.Errorf("id = %q, want %q", parsed.ID, "itaktorch-123")
	}
	if len(parsed.Choices) != 1 {
		t.Fatalf("choices len = %d, want 1", len(parsed.Choices))
	}
	if parsed.Choices[0].Message.Content != "hello back" {
		t.Errorf("content = %q, want %q", parsed.Choices[0].Message.Content, "hello back")
	}
	if parsed.Usage.TotalTokens != 15 {
		t.Errorf("total tokens = %d, want 15", parsed.Usage.TotalTokens)
	}
}

func TestModelInfoJSON(t *testing.T) {
	info := ModelInfo{ID: "qwen3-0.6b", Object: "model", OwnedBy: "itaktorch"}
	data, err := json.Marshal(info)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if string(data) == "" {
		t.Error("empty JSON")
	}
	var parsed ModelInfo
	if err := json.Unmarshal(data, &parsed); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if parsed.ID != "qwen3-0.6b" {
		t.Errorf("id = %q, want %q", parsed.ID, "qwen3-0.6b")
	}
}

func TestCuratedModelsNotEmpty(t *testing.T) {
	models := CuratedModels()
	if len(models) < 3 {
		t.Errorf("expected at least 3 curated models, got %d", len(models))
	}
	for _, m := range models {
		if m.Name == "" {
			t.Error("curated model has empty name")
		}
		if m.URL == "" {
			t.Errorf("curated model %q has empty URL", m.Name)
		}
		if m.Role == "" {
			t.Errorf("curated model %q has empty role", m.Name)
		}
	}
}
