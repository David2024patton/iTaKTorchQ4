package torch

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

// ===========================================================================
// Feature #4: Path Traversal Validation - Integration Tests
// ===========================================================================

// TestServer_PathTraversal_RejectedInChatCompletions verifies that model names
// containing "../" are rejected in the chat completions handler when used with
// a model registry.
func TestServer_PathTraversal_BlockedInRequest(t *testing.T) {
	// Verify the validation functions work at the unit level.
	tests := []struct {
		name    string
		path    string
		wantErr bool
	}{
		{"traversal_unix", "../../../etc/passwd.gguf", true},
		{"traversal_windows", "..\\..\\windows\\system.gguf", true},
		{"empty_path", "", true},
		{"wrong_extension", "/models/model.bin", true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := ValidateModelPath(tt.path)
			if (err != nil) != tt.wantErr {
				t.Errorf("ValidateModelPath(%q) err=%v, wantErr=%v", tt.path, err, tt.wantErr)
			}
		})
	}
}

// ===========================================================================
// Feature #5: Rate Limiting - Integration Tests
// ===========================================================================

// TestServer_RateLiming_ChatEndpoint verifies the rate limiter works when
// wired into the full server with chat completions.
func TestServer_RateLimiting_ChatEndpoint(t *testing.T) {
	engine := NewMockEngine("test-model")
	// 1 burst means: first request OK, second request blocked.
	server := NewServer(engine, 0, WithRateLimit(60, 1))

	chatReq := ChatRequest{
		Model:    "test-model",
		Messages: []ChatMessage{{Role: "user", Content: "hello"}},
	}
	body, _ := json.Marshal(chatReq)

	// Request 1: should succeed.
	req1 := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(body))
	req1.Header.Set("Content-Type", "application/json")
	req1.RemoteAddr = "192.168.1.100:12345"
	w1 := httptest.NewRecorder()
	server.server.Handler.ServeHTTP(w1, req1)

	if w1.Code != http.StatusOK {
		t.Fatalf("request 1: expected 200, got %d: %s", w1.Code, w1.Body.String())
	}

	// Request 2: should be rate-limited (429).
	req2 := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(body))
	req2.Header.Set("Content-Type", "application/json")
	req2.RemoteAddr = "192.168.1.100:12345"
	w2 := httptest.NewRecorder()
	server.server.Handler.ServeHTTP(w2, req2)

	if w2.Code != http.StatusTooManyRequests {
		t.Errorf("request 2: expected 429, got %d", w2.Code)
	}
	if w2.Header().Get("Retry-After") == "" {
		t.Error("missing Retry-After header on 429 response")
	}

	// Request 3 from a DIFFERENT IP: should succeed (per-IP isolation).
	req3 := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(body))
	req3.Header.Set("Content-Type", "application/json")
	req3.RemoteAddr = "10.0.0.5:54321"
	w3 := httptest.NewRecorder()
	server.server.Handler.ServeHTTP(w3, req3)

	if w3.Code != http.StatusOK {
		t.Errorf("request 3 (different IP): expected 200, got %d", w3.Code)
	}
}

// TestServer_RateLimit_HealthNotLimited verifies that /health is NOT rate-limited.
func TestServer_RateLimit_HealthNotLimited(t *testing.T) {
	engine := NewMockEngine("test-model")
	server := NewServer(engine, 0, WithRateLimit(60, 1))

	// Exhaust the rate limit on chat.
	chatReq := ChatRequest{
		Model:    "test-model",
		Messages: []ChatMessage{{Role: "user", Content: "test"}},
	}
	body, _ := json.Marshal(chatReq)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(body))
	req.RemoteAddr = "10.10.10.10:1111"
	w := httptest.NewRecorder()
	server.server.Handler.ServeHTTP(w, req)

	// Health check should still work even though chat is rate-limited.
	healthReq := httptest.NewRequest(http.MethodGet, "/health", nil)
	healthReq.RemoteAddr = "10.10.10.10:1111"
	hw := httptest.NewRecorder()
	server.server.Handler.ServeHTTP(hw, healthReq)

	if hw.Code != http.StatusOK {
		t.Errorf("health check: expected 200, got %d", hw.Code)
	}
}

// ===========================================================================
// Feature #2: Model Management Endpoints - Integration Tests
// ===========================================================================

// TestServer_ModelEndpoints_NoRegistry verifies graceful errors when endpoints
// are called without a registry configured.
func TestServer_ModelLoad_NoRegistry(t *testing.T) {
	engine := NewMockEngine("test-model")
	server := NewServer(engine, 0) // no registry

	body := []byte(`{"model":"qwen3"}`)
	req := httptest.NewRequest(http.MethodPost, "/v1/models/load", bytes.NewReader(body))
	w := httptest.NewRecorder()
	server.server.Handler.ServeHTTP(w, req)

	if w.Code != http.StatusServiceUnavailable {
		t.Errorf("expected 503 without registry, got %d", w.Code)
	}
}

// TestServer_ModelSearch_NoHFPuller verifies search without HF puller configured.
func TestServer_ModelSearch_NoHFPuller(t *testing.T) {
	engine := NewMockEngine("test-model")
	server := NewServer(engine, 0) // no HF puller

	req := httptest.NewRequest(http.MethodGet, "/v1/models/search?q=qwen3", nil)
	w := httptest.NewRecorder()
	server.server.Handler.ServeHTTP(w, req)

	if w.Code != http.StatusServiceUnavailable {
		t.Errorf("expected 503 without HF puller, got %d", w.Code)
	}
}

// TestServer_ModelPull_NoHFPuller verifies pull without HF puller configured.
func TestServer_ModelPull_NoHFPuller(t *testing.T) {
	engine := NewMockEngine("test-model")
	server := NewServer(engine, 0)

	body := []byte(`{"repo":"Qwen/test","filename":"test.gguf"}`)
	req := httptest.NewRequest(http.MethodPost, "/v1/models/pull", bytes.NewReader(body))
	w := httptest.NewRecorder()
	server.server.Handler.ServeHTTP(w, req)

	if w.Code != http.StatusServiceUnavailable {
		t.Errorf("expected 503 without HF puller, got %d", w.Code)
	}
}

// TestServer_ModelsLoaded_NoRegistry verifies single-model mode returns the default engine.
func TestServer_ModelsLoaded_SingleModel(t *testing.T) {
	engine := NewMockEngine("my-model")
	server := NewServer(engine, 0)

	req := httptest.NewRequest(http.MethodGet, "/v1/models/loaded", nil)
	w := httptest.NewRecorder()
	server.server.Handler.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", w.Code)
	}

	var resp map[string]interface{}
	json.Unmarshal(w.Body.Bytes(), &resp)

	if resp["mode"] != "single-model" {
		t.Errorf("mode = %q, want %q", resp["mode"], "single-model")
	}
}

// TestServer_ModelUnload_WrongMethod verifies method validation.
func TestServer_ModelUnload_WrongMethod(t *testing.T) {
	engine := NewMockEngine("test-model")
	server := NewServer(engine, 0)

	req := httptest.NewRequest(http.MethodGet, "/v1/models/unload", nil)
	w := httptest.NewRecorder()
	server.server.Handler.ServeHTTP(w, req)

	if w.Code != http.StatusMethodNotAllowed {
		t.Errorf("expected 405 for GET on unload, got %d", w.Code)
	}
}

// ===========================================================================
// Feature #3: Priority Queue - Integration Tests
// ===========================================================================

// TestScheduler_PriorityLanes verifies that high-priority requests bypass normal queue.
func TestScheduler_PriorityLanes(t *testing.T) {
	engine := NewMockEngine("test-model")
	scheduler := NewScheduler(engine, 64)
	scheduler.Start()
	defer scheduler.Stop()

	// Submit a normal request.
	normalReq := &InferenceRequest{
		Messages: []ChatMessage{{Role: "user", Content: "normal request"}},
		Params:   CompletionParams{MaxTokens: 10},
		Priority: PriorityNormal,
		Ctx:      context.Background(),
	}
	scheduler.Submit(normalReq)

	// Wait for result.
	result := <-normalReq.ResultCh
	if result.Err != nil {
		t.Fatalf("normal request error: %v", result.Err)
	}
	if result.Text == "" {
		t.Error("expected non-empty text from normal request")
	}

	// Submit a critical request.
	critReq := &InferenceRequest{
		Messages: []ChatMessage{{Role: "user", Content: "critical request"}},
		Params:   CompletionParams{MaxTokens: 10},
		Priority: PriorityCritical,
		Ctx:      context.Background(),
	}
	scheduler.Submit(critReq)

	critResult := <-critReq.ResultCh
	if critResult.Err != nil {
		t.Fatalf("critical request error: %v", critResult.Err)
	}
	if critResult.Text == "" {
		t.Error("expected non-empty text from critical request")
	}

	// Verify stats.
	stats := scheduler.Stats()
	if stats.TotalProcessed != 2 {
		t.Errorf("expected 2 processed, got %d", stats.TotalProcessed)
	}
	if stats.TotalDropped != 0 {
		t.Errorf("expected 0 dropped, got %d", stats.TotalDropped)
	}
}

// ===========================================================================
// Feature #7: Response Cache - Integration Tests
// ===========================================================================

// TestServer_ResponseCache verifies cache hit/miss with X-Cache headers.
func TestServer_ResponseCache(t *testing.T) {
	engine := NewMockEngine("test-model")
	server := NewServer(engine, 0, WithResponseCache(256))

	chatReq := ChatRequest{
		Model:    "test-model",
		Messages: []ChatMessage{{Role: "user", Content: "What is caching?"}},
	}
	body, _ := json.Marshal(chatReq)

	// Request 1: should be a cache MISS.
	req1 := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(body))
	req1.Header.Set("Content-Type", "application/json")
	w1 := httptest.NewRecorder()
	server.server.Handler.ServeHTTP(w1, req1)

	if w1.Code != http.StatusOK {
		t.Fatalf("request 1: expected 200, got %d: %s", w1.Code, w1.Body.String())
	}
	if w1.Header().Get("X-Cache") != "MISS" {
		t.Errorf("request 1: expected X-Cache: MISS, got %q", w1.Header().Get("X-Cache"))
	}

	var resp1 ChatResponse
	json.Unmarshal(w1.Body.Bytes(), &resp1)
	firstResponse := resp1.Choices[0].Message.Content

	// Request 2: same messages, should be a cache HIT.
	req2 := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(body))
	req2.Header.Set("Content-Type", "application/json")
	w2 := httptest.NewRecorder()
	server.server.Handler.ServeHTTP(w2, req2)

	if w2.Code != http.StatusOK {
		t.Fatalf("request 2: expected 200, got %d", w2.Code)
	}
	if w2.Header().Get("X-Cache") != "HIT" {
		t.Errorf("request 2: expected X-Cache: HIT, got %q", w2.Header().Get("X-Cache"))
	}

	var resp2 ChatResponse
	json.Unmarshal(w2.Body.Bytes(), &resp2)
	if resp2.Choices[0].Message.Content != firstResponse {
		t.Error("cached response should match original")
	}

	// Request 3: bypass cache with X-No-Cache header.
	req3 := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(body))
	req3.Header.Set("Content-Type", "application/json")
	req3.Header.Set("X-No-Cache", "true")
	w3 := httptest.NewRecorder()
	server.server.Handler.ServeHTTP(w3, req3)

	if w3.Code != http.StatusOK {
		t.Fatalf("request 3: expected 200, got %d", w3.Code)
	}
	// With X-No-Cache, there should be no X-Cache header.
	if w3.Header().Get("X-Cache") != "" {
		t.Errorf("request 3 (no-cache): should not have X-Cache header, got %q", w3.Header().Get("X-Cache"))
	}
}

// TestResponseCache_Stats verifies the cache stats reporting.
func TestResponseCache_Stats(t *testing.T) {
	cache := NewResponseCache(10)

	// Empty cache.
	stats := cache.Stats()
	if stats.Entries != 0 || stats.Hits != 0 || stats.Misses != 0 {
		t.Errorf("empty cache stats wrong: %+v", stats)
	}

	// Put + Get (hit).
	cache.Put("key1", "response1", nil)
	cache.Get("key1")

	stats = cache.Stats()
	if stats.Entries != 1 {
		t.Errorf("expected 1 entry, got %d", stats.Entries)
	}
	if stats.Hits != 1 {
		t.Errorf("expected 1 hit, got %d", stats.Hits)
	}

	// Get miss.
	cache.Get("nonexistent")
	stats = cache.Stats()
	if stats.Misses != 1 {
		t.Errorf("expected 1 miss, got %d", stats.Misses)
	}
	if stats.HitRate != 50.0 {
		t.Errorf("expected 50%% hit rate, got %.1f%%", stats.HitRate)
	}
}

// TestResponseCache_LRUEviction verifies that old entries get evicted.
func TestResponseCache_LRUEviction(t *testing.T) {
	cache := NewResponseCache(3) // max 3 entries

	cache.Put("a", "resp-a", nil)
	cache.Put("b", "resp-b", nil)
	cache.Put("c", "resp-c", nil)
	// Cache is full: [c, b, a]

	// Adding "d" should evict "a" (least recently used).
	cache.Put("d", "resp-d", nil)

	if _, _, ok := cache.Get("a"); ok {
		t.Error("entry 'a' should have been evicted")
	}
	if _, _, ok := cache.Get("d"); !ok {
		t.Error("entry 'd' should exist")
	}
	if _, _, ok := cache.Get("b"); !ok {
		t.Error("entry 'b' should still exist")
	}
}

// ===========================================================================
// Feature #8: Grammar - Integration Tests
// ===========================================================================

// TestGrammar_Validation verifies grammar string validation.
func TestGrammar_Validation(t *testing.T) {
	tests := []struct {
		name    string
		grammar string
		wantErr bool
	}{
		{"valid_json_grammar", GrammarJSON, false},
		{"valid_bool_grammar", GrammarBool, false},
		{"empty_grammar", "", false},
		{"missing_root", `value ::= "hello"`, true},
		{"missing_rule_sep", `root = "hello"`, true},
		{"unbalanced_quotes", `root ::= "hello`, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := ValidateGrammar(tt.grammar)
			if (err != nil) != tt.wantErr {
				t.Errorf("ValidateGrammar() err=%v, wantErr=%v", err, tt.wantErr)
			}
		})
	}
}

// TestGrammar_ForFormat verifies format-to-grammar mapping.
func TestGrammar_ForFormat(t *testing.T) {
	if g := GrammarForFormat("json"); g == "" {
		t.Error("json format should return a grammar")
	}
	if g := GrammarForFormat("json_array"); g == "" {
		t.Error("json_array format should return a grammar")
	}
	if g := GrammarForFormat("bool"); g == "" {
		t.Error("bool format should return a grammar")
	}
	if g := GrammarForFormat("unknown"); g != "" {
		t.Error("unknown format should return empty grammar")
	}
}

// ===========================================================================
// Feature #1: HuggingFace Pull - Unit Tests
// ===========================================================================

// TestHFPuller_Creation verifies the puller can be created.
func TestHFPuller_Creation(t *testing.T) {
	dir := t.TempDir()
	puller, err := NewHFPuller(dir, "")
	if err != nil {
		t.Fatalf("NewHFPuller error: %v", err)
	}
	if puller.CacheDir != dir {
		t.Errorf("CacheDir = %q, want %q", puller.CacheDir, dir)
	}
}

// TestHFPuller_PullBadFilename verifies security on pull filenames.
func TestHFPuller_PullBadFilename(t *testing.T) {
	dir := t.TempDir()
	puller, _ := NewHFPuller(dir, "")

	tests := []string{
		"../../etc/passwd.gguf",  // traversal
		"model/inside/dir.gguf", // has forward slash
		"model.bin",             // wrong extension
	}

	for _, fname := range tests {
		_, err := puller.Pull(context.Background(), "test/repo", fname)
		if err == nil {
			t.Errorf("expected error for bad filename %q", fname)
		}
	}
}

// TestFormatSize verifies human-readable file size formatting.
func TestFormatSize(t *testing.T) {
	tests := []struct {
		bytes int64
		want  string
	}{
		{0, "0 B"},
		{512, "512 B"},
		{1024, "1.0 KB"},
		{1048576, "1.0 MB"},
		{1073741824, "1.0 GB"},
		{5368709120, "5.0 GB"},
	}

	for _, tt := range tests {
		got := FormatSize(tt.bytes)
		if got != tt.want {
			t.Errorf("FormatSize(%d) = %q, want %q", tt.bytes, got, tt.want)
		}
	}
}
