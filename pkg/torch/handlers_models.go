// handlers_models.go implements the model management HTTP handlers for iTaK Torch.
//
// These handlers power Features #1 (HuggingFace Model Pull), #2 (Runtime Model Swap),
// and Ollama registry integration.
//
// Endpoints:
//   POST /v1/models/pull            - download a GGUF model from HuggingFace
//   GET  /v1/models/search          - search HuggingFace for GGUF models
//   POST /v1/models/load            - load a model into memory
//   POST /v1/models/unload          - remove a model from memory
//   GET  /v1/models/loaded          - list currently loaded models
//   POST /v1/models/pull/ollama     - download a model from the Ollama registry
//   GET  /v1/models/search/ollama   - search the Ollama library
package torch

import (
	"encoding/json"
	"fmt"
	"net/http"

	"github.com/David2024patton/iTaKCore/pkg/event"
)

// handleModelPull downloads a GGUF model from HuggingFace Hub.
//
// Request:
//
//	POST /v1/models/pull
//	{"repo": "Qwen/Qwen3-0.6B-GGUF", "filename": "qwen3-0.6b-q4_k_m.gguf"}
//
// Response:
//
//	{"local_path": "~/.torch/models/qwen3-0.6b-q4_k_m.gguf", "size": 419430400, "sha256": "abc..."}
func (s *Server) handleModelPull(w http.ResponseWriter, r *http.Request) {
	// Only accept POST requests.
	if r.Method != http.MethodPost {
		http.Error(w, `{"error":"method not allowed, use POST"}`, http.StatusMethodNotAllowed)
		return
	}

	// Check if HF puller is configured.
	if s.hfPuller == nil {
		http.Error(w, `{"error":"HuggingFace pull not configured. Use WithHFPuller() server option."}`, http.StatusServiceUnavailable)
		return
	}

	// Parse request body.
	var req struct {
		Repo     string `json:"repo"`     // e.g. "Qwen/Qwen3-0.6B-GGUF"
		Filename string `json:"filename"` // e.g. "qwen3-0.6b-q4_k_m.gguf"
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf(`{"error":"invalid JSON: %s"}`, err), http.StatusBadRequest)
		return
	}

	if req.Repo == "" || req.Filename == "" {
		http.Error(w, `{"error":"both 'repo' and 'filename' are required"}`, http.StatusBadRequest)
		return
	}

	s.debugLogger.Info("model_pull", "Pulling %s/%s", req.Repo, req.Filename)

	// Download the model (blocks until complete).
	result, err := s.hfPuller.Pull(r.Context(), req.Repo, req.Filename)
	if err != nil {
		s.debugLogger.Error("model_pull", "Pull failed: %v", err)
		http.Error(w, fmt.Sprintf(`{"error":"pull failed: %s"}`, err), http.StatusInternalServerError)
		return
	}

	// Emit event on successful pull.
	s.emitEvent(event.TypeModelLoaded, map[string]interface{}{
		"action":   "pulled",
		"repo":     req.Repo,
		"filename": req.Filename,
		"path":     result.LocalPath,
		"size":     result.Size,
		"sha256":   result.SHA256,
		"resumed":  result.Resumed,
	})

	s.debugLogger.Info("model_pull", "Downloaded %s (%s)", req.Filename, FormatSize(result.Size))

	// Respond with download result.
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"local_path": result.LocalPath,
		"size":       result.Size,
		"size_human": FormatSize(result.Size),
		"sha256":     result.SHA256,
		"resumed":    result.Resumed,
	})
}

// handleModelSearch searches HuggingFace for GGUF models.
//
// Request:
//
//	GET /v1/models/search?q=qwen3
//
// Response:
//
//	[{"modelId": "Qwen/Qwen3-0.6B-GGUF", "downloads": 12345, ...}]
func (s *Server) handleModelSearch(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, `{"error":"method not allowed, use GET"}`, http.StatusMethodNotAllowed)
		return
	}

	if s.hfPuller == nil {
		http.Error(w, `{"error":"HuggingFace search not configured. Use WithHFPuller() server option."}`, http.StatusServiceUnavailable)
		return
	}

	query := r.URL.Query().Get("q")
	if query == "" {
		http.Error(w, `{"error":"'q' query parameter is required"}`, http.StatusBadRequest)
		return
	}

	results, err := s.hfPuller.Search(query)
	if err != nil {
		http.Error(w, fmt.Sprintf(`{"error":"search failed: %s"}`, err), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	s.writeJSON(w, results)
}

// handleModelLoad loads a model into memory via the ModelRegistry.
//
// Request:
//
//	POST /v1/models/load
//	{"model": "qwen3-0.6b"}
//
// Response:
//
//	{"status": "loaded", "model": "qwen3-0.6b"}
func (s *Server) handleModelLoad(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, `{"error":"method not allowed, use POST"}`, http.StatusMethodNotAllowed)
		return
	}

	if s.registry == nil {
		http.Error(w, `{"error":"model registry not configured. Use WithRegistry() server option."}`, http.StatusServiceUnavailable)
		return
	}

	var req struct {
		Model string `json:"model"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf(`{"error":"invalid JSON: %s"}`, err), http.StatusBadRequest)
		return
	}

	if req.Model == "" {
		http.Error(w, `{"error":"'model' field is required"}`, http.StatusBadRequest)
		return
	}

	s.debugLogger.Info("model_load", "Loading model: %s", req.Model)

	// Load the model via the registry (this may evict the LRU model if at capacity).
	_, err := s.registry.GetOrLoad(req.Model)
	if err != nil {
		s.debugLogger.Error("model_load", "Load failed: %v", err)
		http.Error(w, fmt.Sprintf(`{"error":"load failed: %s"}`, err), http.StatusInternalServerError)
		return
	}

	// Emit model.loaded event.
	s.emitEvent(event.TypeModelLoaded, map[string]interface{}{
		"action": "loaded",
		"model":  req.Model,
	})

	s.debugLogger.Info("model_load", "Model loaded: %s", req.Model)

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"status": "loaded",
		"model":  req.Model,
	})
}

// handleModelUnload removes a model from memory.
//
// Request:
//
//	POST /v1/models/unload
//	{"model": "qwen3-0.6b"}
//
// Response:
//
//	{"status": "unloaded", "model": "qwen3-0.6b"}
func (s *Server) handleModelUnload(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, `{"error":"method not allowed, use POST"}`, http.StatusMethodNotAllowed)
		return
	}

	if s.registry == nil {
		http.Error(w, `{"error":"model registry not configured"}`, http.StatusServiceUnavailable)
		return
	}

	var req struct {
		Model string `json:"model"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf(`{"error":"invalid JSON: %s"}`, err), http.StatusBadRequest)
		return
	}

	if req.Model == "" {
		http.Error(w, `{"error":"'model' field is required"}`, http.StatusBadRequest)
		return
	}

	s.debugLogger.Info("model_unload", "Unloading model: %s", req.Model)

	if err := s.registry.Unload(req.Model); err != nil {
		http.Error(w, fmt.Sprintf(`{"error":"unload failed: %s"}`, err), http.StatusInternalServerError)
		return
	}

	// Emit model.unloaded event.
	s.emitEvent(event.TypeModelUnloaded, map[string]interface{}{
		"action": "unloaded",
		"model":  req.Model,
	})

	s.debugLogger.Info("model_unload", "Model unloaded: %s", req.Model)

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"status":  "unloaded",
		"model":   req.Model,
	})
}

// handleModelsLoaded returns all currently loaded models with registry stats.
//
// Request:
//
//	GET /v1/models/loaded
//
// Response:
//
//	{"loaded": ["qwen3-0.6b", "phi3-mini"], "stats": {...}}
func (s *Server) handleModelsLoaded(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, `{"error":"method not allowed, use GET"}`, http.StatusMethodNotAllowed)
		return
	}

	if s.registry == nil {
		// No registry - return just the default engine.
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{
			"loaded": []string{s.engine.ModelName()},
			"mode":   "single-model",
		})
		return
	}

	stats := s.registry.Stats()
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"loaded": stats.LoadedNames,
		"stats":  stats,
		"mode":   "registry",
	})
}

// ---------- Ollama Registry Handlers ----------

// handleOllamaModelPull downloads a model from the Ollama registry.
//
// Request:
//
//	POST /v1/models/pull/ollama
//	{"model": "qwen3:0.6b"}
//
// Response:
//
//	{"local_path": "...", "size": 419430400, "digest": "sha256:abc..."}
func (s *Server) handleOllamaModelPull(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, `{"error":"method not allowed, use POST"}`, http.StatusMethodNotAllowed)
		return
	}

	if s.ollamaPuller == nil {
		http.Error(w, `{"error":"Ollama pull not configured. Use WithOllamaPuller() server option."}`, http.StatusServiceUnavailable)
		return
	}

	var req struct {
		Model string `json:"model"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf(`{"error":"invalid JSON: %s"}`, err), http.StatusBadRequest)
		return
	}

	if req.Model == "" {
		http.Error(w, `{"error":"'model' field is required (e.g. 'qwen3:0.6b')"}`, http.StatusBadRequest)
		return
	}

	model, tag := ParseOllamaModelRef(req.Model)
	s.debugLogger.Info("ollama_pull", "Pulling %s:%s from Ollama registry", model, tag)

	result, err := s.ollamaPuller.Pull(r.Context(), req.Model)
	if err != nil {
		s.debugLogger.Error("ollama_pull", "Pull failed: %v", err)
		http.Error(w, fmt.Sprintf(`{"error":"pull failed: %s"}`, err), http.StatusInternalServerError)
		return
	}

	s.emitEvent(event.TypeModelLoaded, map[string]interface{}{
		"action": "pulled",
		"source": "ollama",
		"model":  model,
		"tag":    tag,
		"path":   result.LocalPath,
		"size":   result.Size,
	})

	s.debugLogger.Info("ollama_pull", "Downloaded %s:%s (%s)", model, tag, FormatSize(result.Size))

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"local_path": result.LocalPath,
		"size":       result.Size,
		"size_human": FormatSize(result.Size),
		"digest":     result.Digest,
		"model":      result.Model,
		"tag":        result.Tag,
	})
}

// handleOllamaModelSearch searches the Ollama library for models.
//
// Request:
//
//	GET /v1/models/search/ollama?q=qwen3
//
// Response:
//
//	[{"name": "qwen3", "category": "general", "tags": [{"name": "0.6b", "size_human": "400 MB", "fits_system": true}]}]
//
// Query parameters:
//   - q: search query, matches model names (optional if type is provided)
//   - type: category filter: general, coding, vision, embedding, thinking, moe (optional)
//   - filter: "system" to auto-detect memory and mark which models fit (optional)
func (s *Server) handleOllamaModelSearch(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, `{"error":"method not allowed, use GET"}`, http.StatusMethodNotAllowed)
		return
	}

	if s.ollamaPuller == nil {
		http.Error(w, `{"error":"Ollama search not configured. Use WithOllamaPuller() server option."}`, http.StatusServiceUnavailable)
		return
	}

	query := r.URL.Query().Get("q")
	modelType := r.URL.Query().Get("type")

	// At least one filter is required.
	if query == "" && modelType == "" {
		http.Error(w, `{"error":"'q' or 'type' query parameter is required"}`, http.StatusBadRequest)
		return
	}

	// When filter=system, detect available memory and pass to Search().
	var maxMemory int64
	if r.URL.Query().Get("filter") == "system" {
		maxMemory = DetectSystemMemory()
		s.debugLogger.Info("ollama_search", "System filter enabled: budget=%s", FormatSize(maxMemory))
	}

	results, err := s.ollamaPuller.Search(query, maxMemory, modelType)
	if err != nil {
		http.Error(w, fmt.Sprintf(`{"error":"search failed: %s"}`, err), http.StatusInternalServerError)
		return
	}

	// Build response with system info and available categories.
	resp := map[string]any{
		"models": results,
		"total":  len(results),
		"categories": []string{
			ModelCategoryGeneral, ModelCategoryCoding, ModelCategoryVision,
			ModelCategoryEmbedding, ModelCategoryThinking, ModelCategoryMOE,
		},
	}
	if maxMemory > 0 {
		resp["system_memory"] = FormatSize(maxMemory)
		resp["system_memory_bytes"] = maxMemory
	}
	if modelType != "" {
		resp["filtered_type"] = modelType
	}

	w.Header().Set("Content-Type", "application/json")
	s.writeJSON(w, resp)
}
