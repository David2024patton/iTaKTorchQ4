// handlers_embeddings.go implements the OpenAI-compatible /v1/embeddings endpoint.
//
// This allows Torch to be used as an embedding server for RAG pipelines,
// semantic search, and similarity matching. The endpoint follows the OpenAI
// embeddings API format so it's a drop-in replacement.
package torch

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	"github.com/David2024patton/iTaKTorch/pkg/torch/llama"
)

// ---------- Request / Response Types ----------

// EmbeddingRequest is the incoming request body for /v1/embeddings.
type EmbeddingRequest struct {
	Input          interface{} `json:"input"` // string or []string
	Model          string      `json:"model"`
	EncodingFormat string      `json:"encoding_format,omitempty"` // "float" (default) or "base64"
}

// EmbeddingResponse is the response body for /v1/embeddings.
type EmbeddingResponse struct {
	Object string          `json:"object"` // "list"
	Data   []EmbeddingData `json:"data"`
	Model  string          `json:"model"`
	Usage  EmbeddingUsage  `json:"usage"`
}

// EmbeddingData holds a single embedding vector.
type EmbeddingData struct {
	Object    string    `json:"object"` // "embedding"
	Embedding []float32 `json:"embedding"`
	Index     int       `json:"index"`
}

// EmbeddingUsage tracks token consumption for an embedding request.
type EmbeddingUsage struct {
	PromptTokens int `json:"prompt_tokens"`
	TotalTokens  int `json:"total_tokens"`
}

// ---------- Handler ----------

// handleEmbeddings handles POST /v1/embeddings.
// Generates embedding vectors for the given input text(s).
func (s *Server) handleEmbeddings(w http.ResponseWriter, r *http.Request) {
	start := time.Now()
	s.debugf("[REQ] %s %s from %s", r.Method, r.URL.Path, r.RemoteAddr)

	if r.Method != http.MethodPost {
		s.writeError(w, http.StatusMethodNotAllowed, "method not allowed, use POST")
		return
	}

	var req EmbeddingRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		s.writeError(w, http.StatusBadRequest, fmt.Sprintf("invalid JSON: %v", err))
		return
	}
	defer r.Body.Close()

	// Normalize input to []string.
	var inputs []string
	switch v := req.Input.(type) {
	case string:
		inputs = []string{v}
	case []interface{}:
		for _, item := range v {
			if s, ok := item.(string); ok {
				inputs = append(inputs, s)
			}
		}
	default:
		s.writeError(w, http.StatusBadRequest, "input must be a string or array of strings")
		return
	}

	if len(inputs) == 0 {
		s.writeError(w, http.StatusBadRequest, "input is empty")
		return
	}

	// Get the engine. For embeddings, we need access to the TorchEngine internals.
	te, ok := s.engine.(*TorchEngine)
	if !ok {
		s.writeError(w, http.StatusInternalServerError, "embeddings require TorchEngine (not mock)")
		return
	}

	te.mu.Lock()
	defer te.mu.Unlock()

	// Enable embeddings mode on the context.
	llama.SetEmbeddings(te.ctx, true)
	defer llama.SetEmbeddings(te.ctx, false)

	nEmbd := int(llama.ModelNEmbd(te.model))
	var allData []EmbeddingData
	totalTokens := 0

	for i, input := range inputs {
		// Tokenize.
		tokens := llama.Tokenize(te.vocab, input, true, false)
		totalTokens += len(tokens)

		// Clear KV cache for clean embedding.
		if mem, err := llama.GetMemory(te.ctx); err == nil {
			llama.MemoryClear(mem, true)
		}

		// Process tokens through the model.
		batch := llama.BatchGetOne(tokens)
		if _, err := llama.Decode(te.ctx, batch); err != nil {
			s.writeError(w, http.StatusInternalServerError, fmt.Sprintf("embedding decode failed: %v", err))
			return
		}

		// Extract embeddings (pooled output for the sequence).
		embeddings, err := llama.GetEmbeddingsSeq(te.ctx, 0, int32(nEmbd))
		if err != nil || embeddings == nil {
			// Fallback: get embeddings for the last token.
			embeddings, err = llama.GetEmbeddingsIth(te.ctx, int32(len(tokens)-1), int32(nEmbd))
			if err != nil {
				s.writeError(w, http.StatusInternalServerError, fmt.Sprintf("embedding extraction failed: %v", err))
				return
			}
		}

		// Normalize the embedding vector (L2 normalization).
		normalized := normalizeL2(embeddings)

		allData = append(allData, EmbeddingData{
			Object:    "embedding",
			Embedding: normalized,
			Index:     i,
		})
	}

	resp := EmbeddingResponse{
		Object: "list",
		Data:   allData,
		Model:  te.ModelName(),
		Usage: EmbeddingUsage{
			PromptTokens: totalTokens,
			TotalTokens:  totalTokens,
		},
	}

	elapsed := time.Since(start)
	s.debugf("[RES] 200 OK embeddings=%d tokens=%d in %s", len(allData), totalTokens, elapsed)

	w.Header().Set("Content-Type", "application/json")
	s.writeJSON(w, resp)
}

// normalizeL2 applies L2 normalization to an embedding vector.
func normalizeL2(v []float32) []float32 {
	var sum float64
	for _, x := range v {
		sum += float64(x) * float64(x)
	}
	if sum == 0 {
		return v
	}
	norm := float32(1.0 / sqrt64(sum))
	result := make([]float32, len(v))
	for i, x := range v {
		result[i] = x * norm
	}
	return result
}

// sqrt64 computes square root without importing math (avoiding potential conflicts).
func sqrt64(x float64) float64 {
	if x <= 0 {
		return 0
	}
	// Newton's method.
	z := x / 2
	for i := 0; i < 100; i++ {
		z = z - (z*z-x)/(2*z)
	}
	return z
}

// EmbeddingEngine extends Engine with embedding capability.
type EmbeddingEngine interface {
	Engine
	Embed(ctx context.Context, input string) ([]float32, error)
}
