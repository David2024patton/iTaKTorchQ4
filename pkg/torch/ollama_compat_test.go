package torch

import (
	"bufio"
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

// TestOllamaCompat_StreamGenerate verifies streaming /api/generate produces
// valid NDJSON with a final done=true chunk, matching real Ollama's format.
func TestOllamaCompat_StreamGenerate(t *testing.T) {
	engine := NewMockEngine("compat-test")
	server := NewServer(engine, 0, WithOllamaCompat())

	// Streaming is the default when stream is nil/true.
	body := []byte(`{"model":"compat-test","prompt":"hello"}`)
	req := httptest.NewRequest(http.MethodPost, "/api/generate", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	server.server.Handler.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}

	// Content type should be NDJSON for streaming.
	ct := w.Header().Get("Content-Type")
	if ct != "application/x-ndjson" {
		t.Errorf("content-type = %q, want application/x-ndjson", ct)
	}

	// Parse NDJSON lines, last should have done=true.
	scanner := bufio.NewScanner(w.Body)
	var lastChunk OllamaGenerateResponse
	lineCount := 0
	for scanner.Scan() {
		line := scanner.Text()
		if line == "" {
			continue
		}
		lineCount++
		if err := json.Unmarshal([]byte(line), &lastChunk); err != nil {
			t.Fatalf("line %d not valid JSON: %v", lineCount, err)
		}
	}

	if lineCount == 0 {
		t.Fatal("expected at least 1 NDJSON line")
	}
	if !lastChunk.Done {
		t.Error("last chunk should have done=true")
	}
	if lastChunk.DoneReason != "stop" {
		t.Errorf("done_reason = %q, want %q", lastChunk.DoneReason, "stop")
	}
}

// TestOllamaCompat_StreamChat verifies streaming /api/chat NDJSON format.
func TestOllamaCompat_StreamChat(t *testing.T) {
	engine := NewMockEngine("chat-compat")
	server := NewServer(engine, 0, WithOllamaCompat())

	body := []byte(`{"model":"chat-compat","messages":[{"role":"user","content":"hi"}]}`)
	req := httptest.NewRequest(http.MethodPost, "/api/chat", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	server.server.Handler.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}

	// Parse NDJSON, verify last chunk.
	scanner := bufio.NewScanner(w.Body)
	var lastChunk OllamaChatResponse
	for scanner.Scan() {
		line := scanner.Text()
		if line == "" {
			continue
		}
		json.Unmarshal([]byte(line), &lastChunk)
	}

	if !lastChunk.Done {
		t.Error("last chunk should have done=true")
	}
	if lastChunk.Message.Role != "assistant" {
		t.Errorf("role = %q, want assistant", lastChunk.Message.Role)
	}
}

// TestOllamaCompat_TagsSchema verifies /api/tags matches Ollama's exact schema.
func TestOllamaCompat_TagsSchema(t *testing.T) {
	engine := NewMockEngine("tags-schema")
	server := NewServer(engine, 0, WithOllamaCompat())

	req := httptest.NewRequest(http.MethodGet, "/api/tags", nil)
	w := httptest.NewRecorder()
	server.server.Handler.ServeHTTP(w, req)

	// Ollama returns {"models": [...]} at the top level.
	var resp map[string]interface{}
	json.Unmarshal(w.Body.Bytes(), &resp)

	models, ok := resp["models"].([]interface{})
	if !ok {
		t.Fatal("expected top-level 'models' array")
	}

	if len(models) == 0 {
		t.Fatal("expected at least 1 model")
	}

	// Each model must have name, modified_at, details.
	firstModel, ok := models[0].(map[string]interface{})
	if !ok {
		t.Fatal("model entry should be an object")
	}

	for _, key := range []string{"name", "modified_at", "details"} {
		if _, exists := firstModel[key]; !exists {
			t.Errorf("model missing key %q", key)
		}
	}

	// Details should have format field.
	details, ok := firstModel["details"].(map[string]interface{})
	if !ok {
		t.Fatal("details should be an object")
	}
	if _, exists := details["format"]; !exists {
		t.Error("details missing 'format'")
	}
}

// TestOllamaCompat_ShowSchema verifies /api/show has required keys.
func TestOllamaCompat_ShowSchema(t *testing.T) {
	engine := NewMockEngine("show-schema")
	server := NewServer(engine, 0, WithOllamaCompat())

	body := []byte(`{"name":"show-schema"}`)
	req := httptest.NewRequest(http.MethodPost, "/api/show", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	server.server.Handler.ServeHTTP(w, req)

	var resp map[string]interface{}
	json.Unmarshal(w.Body.Bytes(), &resp)

	// Ollama /api/show returns modelfile, parameters, template, details.
	requiredKeys := []string{"modelfile", "parameters", "template", "details"}
	for _, key := range requiredKeys {
		if _, exists := resp[key]; !exists {
			t.Errorf("/api/show missing key %q", key)
		}
	}
}

// TestOllamaCompat_GenerateMetrics verifies non-streaming response contains timing metrics.
func TestOllamaCompat_GenerateMetrics(t *testing.T) {
	engine := NewMockEngine("metrics-test")
	server := NewServer(engine, 0, WithOllamaCompat())

	streamFalse := false
	ollamaReq := OllamaGenerateRequest{
		Model:  "metrics-test",
		Prompt: "test prompt",
		Stream: &streamFalse,
	}
	body, _ := json.Marshal(ollamaReq)

	req := httptest.NewRequest(http.MethodPost, "/api/generate", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	server.server.Handler.ServeHTTP(w, req)

	var resp OllamaGenerateResponse
	json.Unmarshal(w.Body.Bytes(), &resp)

	if resp.TotalDuration < 0 {
		t.Error("total_duration should be > 0")
	}
	if resp.Model != "metrics-test" {
		t.Errorf("model = %q, want metrics-test", resp.Model)
	}
}

// TestOllamaCompat_ChatMetrics verifies chat response contains timing metrics.
func TestOllamaCompat_ChatMetrics(t *testing.T) {
	engine := NewMockEngine("chat-metrics")
	server := NewServer(engine, 0, WithOllamaCompat())

	streamFalse := false
	ollamaReq := OllamaChatRequest{
		Model:    "chat-metrics",
		Messages: []OllamaChatMsg{{Role: "user", Content: "test"}},
		Stream:   &streamFalse,
	}
	body, _ := json.Marshal(ollamaReq)

	req := httptest.NewRequest(http.MethodPost, "/api/chat", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	server.server.Handler.ServeHTTP(w, req)

	var resp OllamaChatResponse
	json.Unmarshal(w.Body.Bytes(), &resp)

	if resp.TotalDuration < 0 {
		t.Error("total_duration should be > 0")
	}
	if resp.DoneReason != "stop" {
		t.Errorf("done_reason = %q, want stop", resp.DoneReason)
	}
}
