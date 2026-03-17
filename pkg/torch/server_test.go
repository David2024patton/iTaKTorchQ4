package torch

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

func TestServerHealth(t *testing.T) {
	engine := NewMockEngine("test-model")
	server := NewServer(engine, 0)

	// Use httptest to test handler directly.
	req := httptest.NewRequest(http.MethodGet, "/health", nil)
	w := httptest.NewRecorder()
	server.server.Handler.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("health status = %d, want 200", w.Code)
	}

	var resp map[string]interface{}
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatalf("unmarshal health: %v", err)
	}
	if resp["status"] != "ok" {
		t.Errorf("status = %q, want %q", resp["status"], "ok")
	}
	if resp["model"] != "test-model" {
		t.Errorf("model = %q, want %q", resp["model"], "test-model")
	}
	// Verify performance and resources sub-objects exist.
	if _, ok := resp["performance"]; !ok {
		t.Error("missing performance object in health response")
	}
	if _, ok := resp["resources"]; !ok {
		t.Error("missing resources object in health response")
	}
}

func TestServerModels(t *testing.T) {
	engine := NewMockEngine("qwen3-0.6b")
	server := NewServer(engine, 0)

	req := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	w := httptest.NewRecorder()
	server.server.Handler.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("models status = %d, want 200", w.Code)
	}

	var resp ModelsResponse
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatalf("unmarshal models: %v", err)
	}
	if len(resp.Data) != 1 {
		t.Fatalf("models count = %d, want 1", len(resp.Data))
	}
	if resp.Data[0].ID != "qwen3-0.6b" {
		t.Errorf("model id = %q, want %q", resp.Data[0].ID, "qwen3-0.6b")
	}
	if resp.Data[0].OwnedBy != "itaktorch" {
		t.Errorf("owned_by = %q, want %q", resp.Data[0].OwnedBy, "itaktorch")
	}
}

func TestServerChatCompletions(t *testing.T) {
	engine := NewMockEngine("test-model")
	server := NewServer(engine, 0)

	chatReq := ChatRequest{
		Model: "test-model",
		Messages: []ChatMessage{
			{Role: "user", Content: "What is 2+2?"},
		},
	}
	body, _ := json.Marshal(chatReq)

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	server.server.Handler.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("chat status = %d, want 200, body: %s", w.Code, w.Body.String())
	}

	var resp ChatResponse
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatalf("unmarshal chat response: %v", err)
	}
	if resp.Object != "chat.completion" {
		t.Errorf("object = %q, want %q", resp.Object, "chat.completion")
	}
	if len(resp.Choices) != 1 {
		t.Fatalf("choices count = %d, want 1", len(resp.Choices))
	}
	if resp.Choices[0].Message.Role != "assistant" {
		t.Errorf("role = %q, want %q", resp.Choices[0].Message.Role, "assistant")
	}
	if resp.Choices[0].Message.Content == "" {
		t.Error("expected non-empty content")
	}
	if resp.Choices[0].FinishReason != "stop" {
		t.Errorf("finish_reason = %q, want %q", resp.Choices[0].FinishReason, "stop")
	}
	if resp.Model != "test-model" {
		t.Errorf("model = %q, want %q", resp.Model, "test-model")
	}
}

func TestServerChatEmptyMessages(t *testing.T) {
	engine := NewMockEngine("test-model")
	server := NewServer(engine, 0)

	chatReq := ChatRequest{
		Model:    "test-model",
		Messages: []ChatMessage{},
	}
	body, _ := json.Marshal(chatReq)

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(body))
	w := httptest.NewRecorder()
	server.server.Handler.ServeHTTP(w, req)

	if w.Code != http.StatusBadRequest {
		t.Errorf("expected 400 for empty messages, got %d", w.Code)
	}
}

func TestServerChatWrongMethod(t *testing.T) {
	engine := NewMockEngine("test-model")
	server := NewServer(engine, 0)

	req := httptest.NewRequest(http.MethodGet, "/v1/chat/completions", nil)
	w := httptest.NewRecorder()
	server.server.Handler.ServeHTTP(w, req)

	if w.Code != http.StatusMethodNotAllowed {
		t.Errorf("expected 405 for GET on chat, got %d", w.Code)
	}
}

func TestServerChatInvalidJSON(t *testing.T) {
	engine := NewMockEngine("test-model")
	server := NewServer(engine, 0)

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader([]byte("not json")))
	w := httptest.NewRecorder()
	server.server.Handler.ServeHTTP(w, req)

	if w.Code != http.StatusBadRequest {
		t.Errorf("expected 400 for invalid JSON, got %d", w.Code)
	}
}

func TestBuildPrompt(t *testing.T) {
	messages := []ChatMessage{
		{Role: "system", Content: "You are a helpful assistant."},
		{Role: "user", Content: "Hello"},
		{Role: "assistant", Content: "Hi there!"},
		{Role: "user", Content: "What is Go?"},
	}

	prompt := BuildPrompt(messages)
	if prompt == "" {
		t.Error("expected non-empty prompt")
	}
	// Should end with "Assistant: " for the model to continue.
	if prompt[len(prompt)-11:] != "Assistant: " {
		t.Errorf("prompt should end with 'Assistant: ', got: %q", prompt[len(prompt)-20:])
	}
}

func TestServerStartStop(t *testing.T) {
	engine := NewMockEngine("test")
	server := NewServer(engine, 0)

	// Use a random port.
	server.server.Addr = ":0"

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	errCh := make(chan error, 1)
	go func() {
		errCh <- server.Start()
	}()

	// Give server a moment to start.
	time.Sleep(100 * time.Millisecond)

	// Stop server.
	if err := server.Stop(); err != nil {
		t.Fatalf("Stop error: %v", err)
	}

	select {
	case err := <-errCh:
		if err != nil {
			t.Fatalf("Server error: %v", err)
		}
	case <-ctx.Done():
		t.Fatal("Server did not stop in time")
	}
}
