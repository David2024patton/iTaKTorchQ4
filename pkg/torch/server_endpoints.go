// server_endpoints.go adds tokenize and model info API endpoints.
//
// Endpoints:
//   POST /v1/tokenize    - Count tokens for text
//   GET  /v1/models/info - Get detailed model metadata
package torch

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"runtime"
	"strings"
	"time"
)

// RegisterAdditionalRoutes mounts tokenize and model info endpoints.
func (s *Server) RegisterAdditionalRoutes(mux *http.ServeMux) {
	mux.HandleFunc("/v1/tokenize", s.handleTokenize)
	mux.HandleFunc("/v1/models/info", s.handleModelsInfo)
	mux.HandleFunc("/v1/audio/speech", s.handleSpeech) // Unified TTS endpoint
}

// TokenizeRequest is the request body for /v1/tokenize.
type TokenizeRequest struct {
	Text  string `json:"text"`
	Model string `json:"model,omitempty"`
}

// TokenizeResponse is the response from /v1/tokenize.
type TokenizeResponse struct {
	Tokens     []int  `json:"tokens"`
	TokenCount int    `json:"token_count"`
	Model      string `json:"model"`
}

func (s *Server) handleTokenize(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		s.writeError(w, http.StatusMethodNotAllowed, "use POST")
		return
	}

	body, err := io.ReadAll(r.Body)
	if err != nil {
		s.writeError(w, http.StatusBadRequest, "failed to read body")
		return
	}
	defer r.Body.Close()

	var req TokenizeRequest
	if err := json.Unmarshal(body, &req); err != nil {
		s.writeError(w, http.StatusBadRequest, fmt.Sprintf("invalid JSON: %v", err))
		return
	}

	if req.Text == "" {
		s.writeError(w, http.StatusBadRequest, "text is required")
		return
	}

	// Simple byte-pair approximation (4 chars per token on average).
	// Real tokenization happens via the engine's tokenizer.
	approxTokens := len(req.Text) / 4
	if approxTokens == 0 && len(req.Text) > 0 {
		approxTokens = 1
	}

	resp := TokenizeResponse{
		TokenCount: approxTokens,
		Model:      s.engine.ModelName(),
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

// ModelDetails matches the JSON structure for GET /v1/models/info.
type ModelDetails struct {
	ID        string    `json:"id"`
	Size      int64     `json:"size_bytes"`
	Path      string    `json:"path"`
	Created   time.Time `json:"created"`
	Backend   string    `json:"backend"`
	Stats     EngineStats `json:"stats"`
}

func (s *Server) handleModelsInfo(w http.ResponseWriter, r *http.Request) {
	name := s.engine.ModelName()
	if name == "" {
		s.writeError(w, http.StatusNotFound, "no model loaded")
		return
	}

	stats := s.engine.GetStats()
	
	backend := "CPU"
	if runtime.GOOS == "darwin" {
		backend = "Metal"
	} else if strings.Contains(strings.ToLower(os.Getenv("ITAK_TORCH_LIB")), "cuda") {
		backend = "CUDA"
	} else if strings.Contains(strings.ToLower(os.Getenv("ITAK_TORCH_LIB")), "vulkan") {
		backend = "Vulkan"
	}

	details := ModelDetails{
		ID:      name,
		Backend: backend,
		Stats:   stats,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(details)
}
