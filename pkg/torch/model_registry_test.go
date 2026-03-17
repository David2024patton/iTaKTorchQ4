package torch

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
)

// createTestModelsDir creates a temp directory with fake .gguf files for testing.
func createTestModelsDir(t *testing.T, modelNames ...string) string {
	t.Helper()
	dir := t.TempDir()
	for _, name := range modelNames {
		path := filepath.Join(dir, name+".gguf")
		if err := os.WriteFile(path, []byte("fake-gguf-data"), 0644); err != nil {
			t.Fatalf("write fake model %s: %v", name, err)
		}
	}
	return dir
}

func TestNewModelRegistry(t *testing.T) {
	dir := createTestModelsDir(t, "model-a", "model-b")

	registry, err := NewModelRegistry(dir, 2, EngineOpts{})
	if err != nil {
		t.Fatalf("NewModelRegistry: %v", err)
	}

	stats := registry.Stats()
	if stats.MaxModels != 2 {
		t.Errorf("MaxModels = %d, want 2", stats.MaxModels)
	}
	if stats.LoadedModels != 0 {
		t.Errorf("LoadedModels = %d, want 0", stats.LoadedModels)
	}
	if stats.ModelsDir != dir {
		t.Errorf("ModelsDir = %q, want %q", stats.ModelsDir, dir)
	}
}

func TestNewModelRegistryBadDir(t *testing.T) {
	_, err := NewModelRegistry(filepath.Join(t.TempDir(), "does_not_exist_subdir", "nope"), 1, EngineOpts{})
	if err == nil {
		t.Fatal("expected error for nonexistent directory")
	}
}

func TestModelRegistryListAvailable(t *testing.T) {
	dir := createTestModelsDir(t, "alpha", "beta", "gamma")

	registry, err := NewModelRegistry(dir, 3, EngineOpts{})
	if err != nil {
		t.Fatalf("NewModelRegistry: %v", err)
	}

	available := registry.ListAvailable()
	if len(available) != 3 {
		t.Fatalf("ListAvailable count = %d, want 3", len(available))
	}

	names := make(map[string]bool)
	for _, m := range available {
		names[m.ID] = true
		if m.Object != "model" {
			t.Errorf("model %s Object = %q, want %q", m.ID, m.Object, "model")
		}
		if m.OwnedBy != "itaktorch" {
			t.Errorf("model %s OwnedBy = %q, want %q", m.ID, m.OwnedBy, "itaktorch")
		}
	}

	for _, expected := range []string{"alpha", "beta", "gamma"} {
		if !names[expected] {
			t.Errorf("model %q missing from ListAvailable", expected)
		}
	}
}

func TestModelRegistryResolveModel(t *testing.T) {
	dir := createTestModelsDir(t, "qwen3-0.6b-q4_k_m")

	registry, err := NewModelRegistry(dir, 1, EngineOpts{})
	if err != nil {
		t.Fatalf("NewModelRegistry: %v", err)
	}

	// Test exact name resolution.
	path, err := registry.resolveModel("qwen3-0.6b-q4_k_m")
	if err != nil {
		t.Fatalf("resolveModel exact: %v", err)
	}
	if !filepath.IsAbs(path) {
		t.Errorf("expected absolute path, got %q", path)
	}

	// Test not found.
	_, err = registry.resolveModel("nonexistent-model")
	if err == nil {
		t.Error("expected error for nonexistent model")
	}
}

func TestModelRegistryStats(t *testing.T) {
	dir := createTestModelsDir(t, "alpha")

	registry, err := NewModelRegistry(dir, 2, EngineOpts{})
	if err != nil {
		t.Fatalf("NewModelRegistry: %v", err)
	}

	stats := registry.Stats()
	if stats.TotalLoads != 0 {
		t.Errorf("TotalLoads = %d, want 0", stats.TotalLoads)
	}
	if stats.CacheHits != 0 {
		t.Errorf("CacheHits = %d, want 0", stats.CacheHits)
	}
}

func TestModelRegistryClose(t *testing.T) {
	dir := createTestModelsDir(t, "model-x")

	registry, err := NewModelRegistry(dir, 1, EngineOpts{})
	if err != nil {
		t.Fatalf("NewModelRegistry: %v", err)
	}

	// Manually inject a mock engine to test Close.
	mock := NewMockEngine("model-x")
	registry.engines["model-x"] = mock
	registry.lru = []string{"model-x"}

	if err := registry.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	if len(registry.engines) != 0 {
		t.Errorf("engines should be empty after Close, got %d", len(registry.engines))
	}
}

// TestServerModelsWithRegistry tests that /v1/models shows all available models
// when a registry is configured.
func TestServerModelsWithRegistry(t *testing.T) {
	dir := createTestModelsDir(t, "model-a", "model-b")

	registry, err := NewModelRegistry(dir, 2, EngineOpts{})
	if err != nil {
		t.Fatalf("NewModelRegistry: %v", err)
	}

	engine := NewMockEngine("placeholder")
	server := NewServer(engine, 0, WithRegistry(registry))

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

	if len(resp.Data) != 2 {
		t.Fatalf("models count = %d, want 2", len(resp.Data))
	}
}

// TestServerHealthWithRegistry tests that /health includes registry stats
// when a registry is configured.
func TestServerHealthWithRegistry(t *testing.T) {
	dir := createTestModelsDir(t, "model-a")

	registry, err := NewModelRegistry(dir, 2, EngineOpts{})
	if err != nil {
		t.Fatalf("NewModelRegistry: %v", err)
	}

	engine := NewMockEngine("placeholder")
	server := NewServer(engine, 0, WithRegistry(registry))

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

	// Should have registry section.
	reg, ok := resp["registry"]
	if !ok {
		t.Fatal("missing 'registry' key in health response")
	}

	regMap, ok := reg.(map[string]interface{})
	if !ok {
		t.Fatal("registry is not a map")
	}

	if regMap["max_models"].(float64) != 2 {
		t.Errorf("registry.max_models = %v, want 2", regMap["max_models"])
	}
}

// TestServerChatCompletionsWithRegistry tests that chat completions work in
// multi-model mode using mock engines.
func TestServerChatCompletionsWithRegistry(t *testing.T) {
	dir := createTestModelsDir(t, "mock-model")

	registry, err := NewModelRegistry(dir, 2, EngineOpts{})
	if err != nil {
		t.Fatalf("NewModelRegistry: %v", err)
	}

	// Pre-inject a mock engine so we don't need real GGUF loading.
	mock := NewMockEngine("mock-model")
	registry.engines["mock-model"] = mock
	registry.lru = []string{"mock-model"}

	engine := NewMockEngine("placeholder")
	server := NewServer(engine, 0, WithRegistry(registry))

	chatReq := ChatRequest{
		Model: "mock-model",
		Messages: []ChatMessage{
			{Role: "user", Content: "Hello from multi-model test"},
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

	// Verify model name in response comes from the resolved engine.
	if resp.Model != "mock-model" {
		t.Errorf("model = %q, want %q", resp.Model, "mock-model")
	}

	if len(resp.Choices) != 1 {
		t.Fatalf("choices count = %d, want 1", len(resp.Choices))
	}

	if resp.Choices[0].Message.Content == "" {
		t.Error("expected non-empty content")
	}
}

// TestServerChatCompletionsSingleModelBackwardCompat ensures single-model mode
// (no registry) still works exactly as before.
func TestServerChatCompletionsSingleModelBackwardCompat(t *testing.T) {
	engine := NewMockEngine("single-model")
	server := NewServer(engine, 0)

	chatReq := ChatRequest{
		Model: "single-model",
		Messages: []ChatMessage{
			{Role: "user", Content: "backward compat test"},
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
		t.Fatalf("unmarshal: %v", err)
	}

	if resp.Model != "single-model" {
		t.Errorf("model = %q, want %q", resp.Model, "single-model")
	}
}
