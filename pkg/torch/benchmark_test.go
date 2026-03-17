package torch

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// BenchmarkMockInference measures the full HTTP handler path through
// the scheduler with a mock engine. This is the "overhead" benchmark:
// everything except actual model inference.
func BenchmarkMockInference(b *testing.B) {
	engine := NewMockEngine("bench-model")
	server := NewServer(engine, 0)

	body := `{"model":"bench","messages":[{"role":"user","content":"benchmark test"}],"max_tokens":10}`

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
		req.Header.Set("Content-Type", "application/json")
		w := httptest.NewRecorder()
		server.server.Handler.ServeHTTP(w, req)

		if w.Code != http.StatusOK {
			b.Fatalf("got %d", w.Code)
		}
	}
}

// BenchmarkCacheHit measures response cache lookup speed.
// First request populates the cache, subsequent requests should be near-instant.
func BenchmarkCacheHit(b *testing.B) {
	engine := NewMockEngine("cache-bench")
	server := NewServer(engine, 0, WithResponseCache(256))

	body := `{"model":"cache-bench","messages":[{"role":"user","content":"cached prompt"}],"max_tokens":5}`

	// Warm up cache with first request.
	warmReq := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	warmReq.Header.Set("Content-Type", "application/json")
	warmW := httptest.NewRecorder()
	server.server.Handler.ServeHTTP(warmW, warmReq)

	if warmW.Header().Get("X-Cache") != "MISS" {
		b.Log("Warning: first request was not a cache miss")
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
		req.Header.Set("Content-Type", "application/json")
		w := httptest.NewRecorder()
		server.server.Handler.ServeHTTP(w, req)

		if w.Code != http.StatusOK {
			b.Fatalf("got %d", w.Code)
		}
	}
}

// BenchmarkSchemaToGBNF measures JSON schema to GBNF grammar conversion speed.
func BenchmarkSchemaToGBNF(b *testing.B) {
	schema := map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"name":    map[string]interface{}{"type": "string"},
			"age":     map[string]interface{}{"type": "integer"},
			"email":   map[string]interface{}{"type": "string"},
			"active":  map[string]interface{}{"type": "boolean"},
			"tags":    map[string]interface{}{"type": "array", "items": map[string]interface{}{"type": "string"}},
			"address": map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"street": map[string]interface{}{"type": "string"},
					"city":   map[string]interface{}{"type": "string"},
					"zip":    map[string]interface{}{"type": "string"},
				},
			},
		},
		"required": []interface{}{"name", "age", "email"},
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = SchemaToGBNF(schema)
	}
}

// BenchmarkOllamaConversion measures Ollama option-to-params conversion speed.
func BenchmarkOllamaConversion(b *testing.B) {
	opts := OllamaOptions{
		NumPredict:    256,
		Temperature:   0.7,
		TopK:          40,
		TopP:          0.9,
		MinP:          0.05,
		Seed:          42,
		RepeatPenalty: 1.1,
		Stop:          []string{"\n", "END", "</s>"},
		NumCtx:        4096,
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = ollamaOptsToParams(opts)
	}
}

// BenchmarkHealthEndpoint measures /health response time.
func BenchmarkHealthEndpoint(b *testing.B) {
	engine := NewMockEngine("health-bench")
	server := NewServer(engine, 0)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		req := httptest.NewRequest(http.MethodGet, "/health", nil)
		w := httptest.NewRecorder()
		server.server.Handler.ServeHTTP(w, req)
	}
}

// BenchmarkOllamaGenerate measures Ollama /api/generate non-streaming path.
func BenchmarkOllamaGenerate(b *testing.B) {
	engine := NewMockEngine("ollama-bench")
	server := NewServer(engine, 0, WithOllamaCompat())

	body := `{"model":"ollama-bench","prompt":"benchmark","stream":false}`

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		req := httptest.NewRequest(http.MethodPost, "/api/generate", strings.NewReader(body))
		req.Header.Set("Content-Type", "application/json")
		w := httptest.NewRecorder()
		server.server.Handler.ServeHTTP(w, req)

		if w.Code != http.StatusOK {
			b.Fatalf("got %d", w.Code)
		}
	}
}
