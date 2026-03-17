package torch

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// TestSmoke_AllEndpoints boots a full-featured mock server and hits every
// major endpoint to verify it returns 200 with valid JSON. This is the
// "does the server actually start and respond" canary test.
func TestSmoke_AllEndpoints(t *testing.T) {
	engine := NewMockEngine("smoke-test-model")
	server := NewServer(engine, 0,
		WithOllamaCompat(),
		WithResponseCache(64),
	)

	endpoints := []struct {
		method string
		path   string
		body   string
	}{
		{http.MethodGet, "/health", ""},
		{http.MethodGet, "/v1/models", ""},
		{http.MethodGet, "/api/version", ""},
		{http.MethodGet, "/api/tags", ""},
		{http.MethodGet, "/v1/cache/stats", ""},
		{http.MethodGet, "/v1/scheduler/stats", ""},
		{http.MethodGet, "/metrics", ""},
		{http.MethodPost, "/v1/chat/completions", `{"model":"test","messages":[{"role":"user","content":"hi"}],"max_tokens":10}`},
		{http.MethodPost, "/api/generate", `{"model":"test","prompt":"hi","stream":false}`},
		{http.MethodPost, "/api/chat", `{"model":"test","messages":[{"role":"user","content":"hi"}],"stream":false}`},
		{http.MethodPost, "/api/show", `{"name":"test"}`},
	}

	for _, ep := range endpoints {
		t.Run(ep.method+" "+ep.path, func(t *testing.T) {
			var req *http.Request
			if ep.body != "" {
				req = httptest.NewRequest(ep.method, ep.path, strings.NewReader(ep.body))
				req.Header.Set("Content-Type", "application/json")
			} else {
				req = httptest.NewRequest(ep.method, ep.path, nil)
			}

			w := httptest.NewRecorder()
			server.server.Handler.ServeHTTP(w, req)

			if w.Code != http.StatusOK {
				t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
			}

			// Verify JSON is parseable (except /metrics which is Prometheus text).
			if ep.path != "/metrics" {
				ct := w.Header().Get("Content-Type")
				if !strings.Contains(ct, "json") && !strings.Contains(ct, "ndjson") {
					t.Errorf("content-type = %q, want json", ct)
				}
				var parsed interface{}
				if err := json.Unmarshal(w.Body.Bytes(), &parsed); err != nil {
					t.Errorf("response is not valid JSON: %v", err)
				}
			}
		})
	}
}

// TestSmoke_HealthFields verifies the /health response has critical fields.
func TestSmoke_HealthFields(t *testing.T) {
	engine := NewMockEngine("health-test")
	server := NewServer(engine, 0)

	req := httptest.NewRequest(http.MethodGet, "/health", nil)
	w := httptest.NewRecorder()
	server.server.Handler.ServeHTTP(w, req)

	var resp map[string]interface{}
	json.Unmarshal(w.Body.Bytes(), &resp)

	requiredKeys := []string{"status", "model", "performance", "resources", "scheduler"}
	for _, key := range requiredKeys {
		if _, ok := resp[key]; !ok {
			t.Errorf("/health missing key %q", key)
		}
	}

	if resp["status"] != "ok" {
		t.Errorf("status = %q, want %q", resp["status"], "ok")
	}
}

// TestSmoke_ChatCompletion_Shape verifies the chat completion response matches OpenAI schema.
func TestSmoke_ChatCompletion_Shape(t *testing.T) {
	engine := NewMockEngine("shape-test")
	server := NewServer(engine, 0)

	body := `{"model":"test","messages":[{"role":"user","content":"test"}],"max_tokens":5}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	server.server.Handler.ServeHTTP(w, req)

	var resp ChatResponse
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatalf("failed to parse response: %v", err)
	}

	if resp.Object != "chat.completion" {
		t.Errorf("object = %q, want %q", resp.Object, "chat.completion")
	}
	if len(resp.Choices) == 0 {
		t.Fatal("expected at least 1 choice")
	}
	if resp.Choices[0].Message.Role != "assistant" {
		t.Errorf("role = %q, want %q", resp.Choices[0].Message.Role, "assistant")
	}
	if !strings.HasPrefix(resp.ID, "itaktorch-") {
		t.Errorf("id = %q, should start with itaktorch-", resp.ID)
	}
}

// TestSmoke_404 verifies unknown paths return 404.
func TestSmoke_404(t *testing.T) {
	engine := NewMockEngine("404-test")
	server := NewServer(engine, 0)

	req := httptest.NewRequest(http.MethodGet, "/nonexistent", nil)
	w := httptest.NewRecorder()
	server.server.Handler.ServeHTTP(w, req)

	if w.Code != http.StatusNotFound {
		t.Errorf("expected 404, got %d", w.Code)
	}
}

// TestSmoke_MethodNotAllowed verifies wrong methods get rejected.
func TestSmoke_MethodNotAllowed(t *testing.T) {
	engine := NewMockEngine("method-test")
	server := NewServer(engine, 0)

	req := httptest.NewRequest(http.MethodGet, "/v1/chat/completions", nil)
	w := httptest.NewRecorder()
	server.server.Handler.ServeHTTP(w, req)

	if w.Code != http.StatusMethodNotAllowed {
		t.Errorf("expected 405, got %d", w.Code)
	}
}

// TestSmoke_EmptyMessages verifies empty messages array is rejected.
func TestSmoke_EmptyMessages(t *testing.T) {
	engine := NewMockEngine("empty-test")
	server := NewServer(engine, 0)

	body := `{"model":"test","messages":[]}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	server.server.Handler.ServeHTTP(w, req)

	if w.Code != http.StatusBadRequest {
		t.Errorf("expected 400 for empty messages, got %d", w.Code)
	}
}
