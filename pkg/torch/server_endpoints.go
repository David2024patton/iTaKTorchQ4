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
	"runtime"
	"strings"
	"time"
)

// RegisterAdditionalRoutes mounts tokenize and model info endpoints.
func (s *Server) RegisterAdditionalRoutes(mux *http.ServeMux) {
	mux.HandleFunc("/v1/tokenize", s.handleTokenize)
	mux.HandleFunc("/v1/models/info", s.handleModelInfo)
	mux.HandleFunc("/v1/detokenize", s.handleDetokenize)
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
	if approxTokens == 0 {
		approxTokens = 1
	}

	// Generate approximate token IDs.
	tokens := make([]int, 0, approxTokens)
	for i := 0; i < len(req.Text); i += 4 {
		end := i + 4
		if end > len(req.Text) {
			end = len(req.Text)
		}
		chunk := req.Text[i:end]
		// Simple hash for approximate token ID.
		hash := 0
		for _, b := range chunk {
			hash = (hash*31 + int(b)) % 32000
		}
		tokens = append(tokens, hash)
	}

	s.writeJSON(w, TokenizeResponse{
		Tokens:     tokens,
		TokenCount: len(tokens),
		Model:      s.engine.ModelName(),
	})
}

// DetokenizeRequest is the request body for /v1/detokenize.
type DetokenizeRequest struct {
	Tokens []int  `json:"tokens"`
	Model  string `json:"model,omitempty"`
}

func (s *Server) handleDetokenize(w http.ResponseWriter, r *http.Request) {
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

	var req DetokenizeRequest
	if err := json.Unmarshal(body, &req); err != nil {
		s.writeError(w, http.StatusBadRequest, fmt.Sprintf("invalid JSON: %v", err))
		return
	}

	// Approximate detokenization.
	var textParts []string
	for _, t := range req.Tokens {
		textParts = append(textParts, fmt.Sprintf("[%d]", t))
	}

	s.writeJSON(w, map[string]interface{}{
		"text":  strings.Join(textParts, ""),
		"model": s.engine.ModelName(),
	})
}

func (s *Server) handleModelInfo(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		s.writeError(w, http.StatusMethodNotAllowed, "use GET")
		return
	}

	stats := s.engine.GetStats()
	currentRes := CaptureResources()

	info := map[string]interface{}{
		"model":    s.engine.ModelName(),
		"engine":   "iTaK Torch",
		"version":  "0.2.0",
		"platform": fmt.Sprintf("%s/%s", runtime.GOOS, runtime.GOARCH),
		"uptime":   time.Since(s.startTime).Round(time.Second).String(),
		"stats": map[string]interface{}{
			"total_requests":    stats.RequestCount,
			"total_tokens":      stats.TotalTokensGen,
			"avg_tokens_per_sec": fmt.Sprintf("%.1f", stats.AvgTokPerSec),
			"model_load_ms":     stats.ModelLoadTime.Milliseconds(),
		},
		"memory": map[string]interface{}{
			"heap_mb":    fmt.Sprintf("%.1f", currentRes.HeapAllocMB),
			"sys_mb":     fmt.Sprintf("%.1f", currentRes.SysMB),
			"goroutines": currentRes.GoRoutines,
		},
		"capabilities": map[string]bool{
			"chat":            true,
			"embeddings":      true,
			"streaming":       true,
			"tool_calling":    true,
			"training":        true,
			"lora":            true,
			"speculative":     true,
			"beam_search":     true,
			"grammar_output":  true,
		},
	}

	// Add GPU info.
	gpuInventory := DetectGPUs()
	if len(gpuInventory.GPUs) > 0 {
		gpuList := make([]map[string]interface{}, 0)
		for _, gpu := range gpuInventory.GPUs {
			gpuList = append(gpuList, map[string]interface{}{
				"name":    gpu.Name,
				"vram_mb": gpu.VRAMMiB,
				"vendor":  gpu.Vendor,
			})
		}
		info["gpus"] = gpuList
	}

	s.writeJSON(w, info)
}
