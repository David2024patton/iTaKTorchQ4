// handlers_ollama.go implements Ollama-compatible API endpoints.
//
// WHY: Ollama is the most popular local LLM runner. Many tools (Open WebUI,
// Continue, etc.) speak the Ollama API natively. By implementing these endpoints,
// Torch becomes a drop-in replacement that's faster.
//
// Supported endpoints:
//   POST /api/generate       - text generation (Ollama format)
//   POST /api/chat           - chat completion (Ollama format)
//   GET  /api/tags           - list loaded models
//   POST /api/show           - show model info
//   GET  /api/version        - server version
//
// These translate Ollama's request/response format to our internal engine calls.
package torch

import (
	"bufio"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"
)

// --- Ollama Request/Response Types ---

// OllamaGenerateRequest matches Ollama's POST /api/generate body.
type OllamaGenerateRequest struct {
	Model     string        `json:"model"`
	Prompt    string        `json:"prompt"`
	System    string        `json:"system,omitempty"`
	Template  string        `json:"template,omitempty"`
	Context   []int         `json:"context,omitempty"` // conversation context (token IDs)
	Stream    *bool         `json:"stream,omitempty"`  // defaults to true in Ollama
	Raw       bool          `json:"raw,omitempty"`
	KeepAlive string        `json:"keep_alive,omitempty"`
	Options   OllamaOptions `json:"options,omitempty"`
}

// OllamaChatRequest matches Ollama's POST /api/chat body.
type OllamaChatRequest struct {
	Model     string          `json:"model"`
	Messages  []OllamaChatMsg `json:"messages"`
	Stream    *bool           `json:"stream,omitempty"`
	KeepAlive string          `json:"keep_alive,omitempty"`
	Options   OllamaOptions   `json:"options,omitempty"`
}

// OllamaChatMsg matches Ollama's message format.
type OllamaChatMsg struct {
	Role    string   `json:"role"`
	Content string   `json:"content"`
	Images  []string `json:"images,omitempty"` // base64 images for multimodal
}

// OllamaOptions matches Ollama's generation options.
type OllamaOptions struct {
	NumPredict    int      `json:"num_predict,omitempty"`
	Temperature   float64  `json:"temperature,omitempty"`
	TopK          int      `json:"top_k,omitempty"`
	TopP          float64  `json:"top_p,omitempty"`
	MinP          float64  `json:"min_p,omitempty"`
	Seed          int      `json:"seed,omitempty"`
	RepeatPenalty float64  `json:"repeat_penalty,omitempty"`
	Stop          []string `json:"stop,omitempty"`
	NumCtx        int      `json:"num_ctx,omitempty"`
}

// OllamaGenerateResponse matches Ollama's streaming response chunks.
type OllamaGenerateResponse struct {
	Model              string    `json:"model"`
	CreatedAt          time.Time `json:"created_at"`
	Response           string    `json:"response"`
	Done               bool      `json:"done"`
	DoneReason         string    `json:"done_reason,omitempty"`
	Context            []int     `json:"context,omitempty"`
	TotalDuration      int64     `json:"total_duration,omitempty"`
	LoadDuration       int64     `json:"load_duration,omitempty"`
	PromptEvalCount    int       `json:"prompt_eval_count,omitempty"`
	PromptEvalDuration int64     `json:"prompt_eval_duration,omitempty"`
	EvalCount          int       `json:"eval_count,omitempty"`
	EvalDuration       int64     `json:"eval_duration,omitempty"`
}

// OllamaChatResponse matches Ollama's chat response format.
type OllamaChatResponse struct {
	Model              string        `json:"model"`
	CreatedAt          time.Time     `json:"created_at"`
	Message            OllamaChatMsg `json:"message"`
	Done               bool          `json:"done"`
	DoneReason         string        `json:"done_reason,omitempty"`
	TotalDuration      int64         `json:"total_duration,omitempty"`
	LoadDuration       int64         `json:"load_duration,omitempty"`
	PromptEvalCount    int           `json:"prompt_eval_count,omitempty"`
	PromptEvalDuration int64         `json:"prompt_eval_duration,omitempty"`
	EvalCount          int           `json:"eval_count,omitempty"`
	EvalDuration       int64         `json:"eval_duration,omitempty"`
}

// OllamaModelInfo matches Ollama's model info response.
// Both "name" and "model" are required - Open WebUI uses "model" for selection.
type OllamaModelInfo struct {
	Name       string            `json:"name"`
	Model      string            `json:"model"`
	ModifiedAt time.Time         `json:"modified_at"`
	Size       int64             `json:"size"`
	Digest     string            `json:"digest"`
	Details    OllamaModelDetail `json:"details"`
}

// OllamaModelDetail holds model detail metadata.
type OllamaModelDetail struct {
	ParentModel       string   `json:"parent_model"`
	Format            string   `json:"format"`
	Family            string   `json:"family"`
	Families          []string `json:"families"`
	ParameterSize     string   `json:"parameter_size"`
	QuantizationLevel string   `json:"quantization_level"`
}

// --- Handlers ---

// handleOllamaGenerate handles POST /api/generate (Ollama-compatible text generation).
func (s *Server) handleOllamaGenerate(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, `{"error":"method not allowed"}`, http.StatusMethodNotAllowed)
		return
	}

	var req OllamaGenerateRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		s.writeError(w, http.StatusBadRequest, fmt.Sprintf("invalid JSON: %v", err))
		return
	}

	// Convert Ollama request to our internal format.
	messages := make([]ChatMessage, 0, 2)
	if req.System != "" {
		messages = append(messages, ChatMessage{Role: "system", Content: req.System})
	}
	messages = append(messages, ChatMessage{Role: "user", Content: req.Prompt})

	params := ollamaOptsToParams(req.Options)

	// Determine streaming (Ollama defaults to true).
	streaming := true
	if req.Stream != nil {
		streaming = *req.Stream
	}

	start := time.Now()

	if streaming {
		s.handleOllamaStreamGenerate(w, r, messages, params, req.Model, start)
		return
	}

	// Non-streaming: run inference and return full response.
	inferReq := &InferenceRequest{
		Messages: messages,
		Params:   params,
		Ctx:      r.Context(),
	}

	s.scheduler.Submit(inferReq)

	var result InferenceResult
	select {
	case result = <-inferReq.ResultCh:
	case <-r.Context().Done():
		return
	}

	if result.Err != nil {
		s.writeError(w, http.StatusInternalServerError, result.Err.Error())
		return
	}

	elapsed := time.Since(start)

	resp := OllamaGenerateResponse{
		Model:         s.ollamaModelName(req.Model),
		CreatedAt:     time.Now(),
		Response:      result.Text,
		Done:          true,
		DoneReason:    "stop",
		TotalDuration: elapsed.Nanoseconds(),
	}

	if result.Metrics != nil {
		resp.PromptEvalCount = result.Metrics.PromptTokens
		resp.EvalCount = result.Metrics.CompletionTokens
		resp.EvalDuration = result.Metrics.GenDuration.Nanoseconds()
		resp.PromptEvalDuration = result.Metrics.PromptDuration.Nanoseconds()
	}

	w.Header().Set("Content-Type", "application/json")
	s.writeJSON(w, resp)
}

// handleOllamaChat handles POST /api/chat (Ollama-compatible chat completion).
func (s *Server) handleOllamaChat(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, `{"error":"method not allowed"}`, http.StatusMethodNotAllowed)
		return
	}

	var req OllamaChatRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		s.writeError(w, http.StatusBadRequest, fmt.Sprintf("invalid JSON: %v", err))
		return
	}

	// Convert Ollama messages to our internal format.
	messages := make([]ChatMessage, 0, len(req.Messages))
	for _, m := range req.Messages {
		messages = append(messages, ChatMessage{Role: m.Role, Content: m.Content})
	}

	params := ollamaOptsToParams(req.Options)

	streaming := true
	if req.Stream != nil {
		streaming = *req.Stream
	}

	start := time.Now()

	if streaming {
		s.handleOllamaStreamChat(w, r, messages, params, req.Model, start)
		return
	}

	// Non-streaming.
	inferReq := &InferenceRequest{
		Messages: messages,
		Params:   params,
		Ctx:      r.Context(),
	}

	s.scheduler.Submit(inferReq)

	var result InferenceResult
	select {
	case result = <-inferReq.ResultCh:
	case <-r.Context().Done():
		return
	}

	if result.Err != nil {
		s.writeError(w, http.StatusInternalServerError, result.Err.Error())
		return
	}

	elapsed := time.Since(start)

	resp := OllamaChatResponse{
		Model:     s.ollamaModelName(req.Model),
		CreatedAt: time.Now(),
		Message: OllamaChatMsg{
			Role:    "assistant",
			Content: result.Text,
		},
		Done:          true,
		DoneReason:    "stop",
		TotalDuration: elapsed.Nanoseconds(),
	}

	if result.Metrics != nil {
		resp.PromptEvalCount = result.Metrics.PromptTokens
		resp.EvalCount = result.Metrics.CompletionTokens
		resp.EvalDuration = result.Metrics.GenDuration.Nanoseconds()
		resp.PromptEvalDuration = result.Metrics.PromptDuration.Nanoseconds()
	}

	w.Header().Set("Content-Type", "application/json")
	s.writeJSON(w, resp)
}

// handleOllamaTags handles GET /api/tags (list available models).
// Lists ALL available models from the models directory, not just loaded ones.
// This matches Ollama's behavior where /api/tags shows all pulled models.
func (s *Server) handleOllamaTags(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, `{"error":"method not allowed"}`, http.StatusMethodNotAllowed)
		return
	}

	models := make([]OllamaModelInfo, 0)

	// Multi-model mode: list ALL available .gguf files from the models directory.
	// Open WebUI expects to see all models on startup, not just loaded ones.
	if s.registry != nil {
		for _, mi := range s.registry.ListAvailable() {
			ollamaName := mi.ID
			models = append(models, OllamaModelInfo{
				Name:       ollamaName,
				Model:      ollamaName,
				ModifiedAt: time.Now(),
			Size:       mi.SizeBytes,
				Details: OllamaModelDetail{
					Format:   "gguf",
					Family:   inferFamily(ollamaName),
					Families: []string{inferFamily(ollamaName)},
				},
			})
		}
	} else if s.engine != nil {
		// Single-model mode: list the loaded model.
		name := s.engine.ModelName()
		models = append(models, OllamaModelInfo{
			Name:       name,
			Model:      name,
			ModifiedAt: time.Now(),
			Details: OllamaModelDetail{
				Format:   "gguf",
				Family:   inferFamily(name),
				Families: []string{inferFamily(name)},
			},
		})
	}

	w.Header().Set("Content-Type", "application/json")
	s.writeJSON(w, map[string]interface{}{"models": models})
}

// handleOllamaVersion handles GET /api/version.
func (s *Server) handleOllamaVersion(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	s.writeJSON(w, map[string]string{"version": "0.17.0-torch"})
}

// handleOllamaShow handles POST /api/show (model information).
func (s *Server) handleOllamaShow(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, `{"error":"method not allowed"}`, http.StatusMethodNotAllowed)
		return
	}

	var req struct {
		Name string `json:"name"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		s.writeError(w, http.StatusBadRequest, "invalid JSON")
		return
	}

	if s.engine == nil {
		s.writeError(w, http.StatusNotFound, "no model loaded")
		return
	}

	modelName := s.engine.ModelName()
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"modelfile":  fmt.Sprintf("FROM %s", modelName),
		"parameters": "temperature 0.8",
		"template":   "chatml",
		"details": OllamaModelDetail{
			Format: "gguf",
			Family: "llama",
		},
	})
}

// --- Streaming Helpers ---

// handleOllamaStreamGenerate streams Ollama-format generate responses.
func (s *Server) handleOllamaStreamGenerate(w http.ResponseWriter, r *http.Request,
	messages []ChatMessage, params CompletionParams, model string, start time.Time) {

	flusher, ok := w.(http.Flusher)
	if !ok {
		s.writeError(w, http.StatusInternalServerError, "streaming not supported")
		return
	}

	w.Header().Set("Content-Type", "application/x-ndjson")
	w.Header().Set("Transfer-Encoding", "chunked")
	w.WriteHeader(http.StatusOK)

	bw := bufio.NewWriter(w)

	streamCh := make(chan string, 32)
	inferReq := &InferenceRequest{
		Messages: messages,
		Params:   params,
		Ctx:      r.Context(),
		StreamCh: streamCh,
	}

	s.scheduler.Submit(inferReq)

	totalTokens := 0
	for {
		select {
		case chunk, ok := <-streamCh:
			if !ok {
				// Stream ended, send final response.
				elapsed := time.Since(start)
				final := OllamaGenerateResponse{
					Model:         s.ollamaModelName(model),
					CreatedAt:     time.Now(),
					Response:      "",
					Done:          true,
					DoneReason:    "stop",
					TotalDuration: elapsed.Nanoseconds(),
					EvalCount:     totalTokens,
				}
				data, _ := json.Marshal(final)
				bw.Write(data)
				bw.WriteString("\n")
				bw.Flush()
				flusher.Flush()
				return
			}

			totalTokens++
			resp := OllamaGenerateResponse{
				Model:     s.ollamaModelName(model),
				CreatedAt: time.Now(),
				Response:  chunk,
				Done:      false,
			}
			data, _ := json.Marshal(resp)
			bw.Write(data)
			bw.WriteString("\n")
			bw.Flush()
			flusher.Flush()

		case <-r.Context().Done():
			return
		}
	}
}

// handleOllamaStreamChat streams Ollama-format chat responses.
func (s *Server) handleOllamaStreamChat(w http.ResponseWriter, r *http.Request,
	messages []ChatMessage, params CompletionParams, model string, start time.Time) {

	flusher, ok := w.(http.Flusher)
	if !ok {
		s.writeError(w, http.StatusInternalServerError, "streaming not supported")
		return
	}

	w.Header().Set("Content-Type", "application/x-ndjson")
	w.Header().Set("Transfer-Encoding", "chunked")
	w.WriteHeader(http.StatusOK)

	bw := bufio.NewWriter(w)

	streamCh := make(chan string, 32)
	inferReq := &InferenceRequest{
		Messages: messages,
		Params:   params,
		Ctx:      r.Context(),
		StreamCh: streamCh,
	}

	s.scheduler.Submit(inferReq)

	totalTokens := 0
	for {
		select {
		case chunk, ok := <-streamCh:
			if !ok {
				elapsed := time.Since(start)
				final := OllamaChatResponse{
					Model:         s.ollamaModelName(model),
					CreatedAt:     time.Now(),
					Message:       OllamaChatMsg{Role: "assistant", Content: ""},
					Done:          true,
					DoneReason:    "stop",
					TotalDuration: elapsed.Nanoseconds(),
					EvalCount:     totalTokens,
				}
				data, _ := json.Marshal(final)
				bw.Write(data)
				bw.WriteString("\n")
				bw.Flush()
				flusher.Flush()
				return
			}

			totalTokens++
			resp := OllamaChatResponse{
				Model:     s.ollamaModelName(model),
				CreatedAt: time.Now(),
				Message:   OllamaChatMsg{Role: "assistant", Content: chunk},
				Done:      false,
			}
			data, _ := json.Marshal(resp)
			bw.Write(data)
			bw.WriteString("\n")
			bw.Flush()
			flusher.Flush()

		case <-r.Context().Done():
			return
		}
	}
}

// --- Helpers ---

// ollamaOptsToParams converts Ollama options to our CompletionParams.
func ollamaOptsToParams(opts OllamaOptions) CompletionParams {
	params := CompletionParams{
		MaxTokens:   512,
		Temperature: 0.8,
	}
	if opts.NumPredict > 0 {
		params.MaxTokens = opts.NumPredict
	}
	if opts.Temperature > 0 {
		params.Temperature = opts.Temperature
	}
	if opts.TopP > 0 {
		params.TopP = opts.TopP
	}
	if len(opts.Stop) > 0 {
		params.Stop = opts.Stop
	}
	return params
}

// ollamaModelName returns the model name, preferring the request model if set.
func (s *Server) ollamaModelName(requested string) string {
	if requested != "" {
		return requested
	}
	if s.engine != nil {
		return s.engine.ModelName()
	}
	return "unknown"
}

// inferFamily guesses the Ollama-style family name from a GGUF model filename.
// Examples: "qwen2.5-0.5b-instruct-q4_k_m" -> "qwen2", "qwen3-8b" -> "qwen3".
func inferFamily(name string) string {
	n := strings.ToLower(name)
	switch {
	case strings.Contains(n, "qwen3-vl"), strings.Contains(n, "qwen3vl"):
		return "qwen3vl"
	case strings.Contains(n, "qwen3"):
		return "qwen3"
	case strings.Contains(n, "qwen2"):
		return "qwen2"
	case strings.Contains(n, "gemma3"):
		return "gemma3"
	case strings.Contains(n, "gemma"):
		return "gemma"
	case strings.Contains(n, "mistral"), strings.Contains(n, "ministral"):
		return "mistral3"
	case strings.Contains(n, "deepseek"):
		return "deepseek"
	case strings.Contains(n, "phi"):
		return "phi"
	case strings.Contains(n, "granite"):
		return "granite"
	case strings.Contains(n, "nemotron"):
		return "nemotron"
	case strings.Contains(n, "openclaw"), strings.Contains(n, "gpt"):
		return "gptoss"
	default:
		return "llama"
	}
}

// RegisterOllamaRoutes registers all Ollama-compatible routes on the mux.
func (s *Server) RegisterOllamaRoutes(mux *http.ServeMux) {
	mux.HandleFunc("/api/generate", s.handleOllamaGenerate)
	mux.HandleFunc("/api/chat", s.handleOllamaChat)
	mux.HandleFunc("/api/tags", s.handleOllamaTags)
	mux.HandleFunc("/api/version", s.handleOllamaVersion)
	mux.HandleFunc("/api/show", s.handleOllamaShow)
}
