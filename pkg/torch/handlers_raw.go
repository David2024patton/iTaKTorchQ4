package torch

import (
	"encoding/json"
	"fmt"
	"net/http"
)

// RawTokenRequest represents a request to generate text from pre-computed token IDs.
type RawTokenRequest struct {
	Model       string           `json:"model"`
	Tokens      []int32          `json:"tokens"`
	MaxTokens   int              `json:"max_tokens,omitempty"`
	Temperature float64          `json:"temperature,omitempty"`
	TopP        float64          `json:"top_p,omitempty"`
	Stop        []string         `json:"stop,omitempty"`
}

// handleGenerateRaw processes requests containing an exact array of token IDs, bypassing the tokenizer.
// This is used by orchestrator agents to inject pre-computed context (like codebases) instantly.
func (s *Server) handleGenerateRaw(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req RawTokenRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("failed to parse JSON: %v", err), http.StatusBadRequest)
		return
	}

	if len(req.Tokens) == 0 {
		http.Error(w, "tokens array cannot be empty", http.StatusBadRequest)
		return
	}


	params := CompletionParams{
		MaxTokens:   req.MaxTokens,
		Temperature: req.Temperature,
		TopP:        req.TopP,
		Stop:        req.Stop,
	}

	// Route the generation through the server's scheduler to ensure thread safety
	// alongside normal chat/generate requests.
	reqCtx := r.Context()
	job := &InferenceRequest{
		Messages:    nil, // Messages are nil; we rely on InputTokens.
		InputTokens: req.Tokens,
		Params:      params,
		Ctx:         reqCtx,
		ResultCh:    make(chan InferenceResult, 1),
	}

	s.scheduler.Submit(job)

	// Wait for generation to complete (synchronous for now, streaming could be added later).
	select {
	case result := <-job.ResultCh:
		if result.Err != nil {
			http.Error(w, fmt.Sprintf("Inference failed: %v", result.Err), http.StatusInternalServerError)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{
			"model":    req.Model,
			"response": result.Text,
		})
	case <-r.Context().Done():
		// Client disconnected.
	}
}
