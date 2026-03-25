package torch

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"os"
	"strings"
	"sync"
	"time"

	coreDebug "github.com/David2024patton/iTaKCore/pkg/debug"
	"github.com/David2024patton/iTaKCore/pkg/event"
	"github.com/David2024patton/iTaKCore/pkg/health"
	"github.com/David2024patton/iTaKCore/pkg/registry"
)

// Server is the OpenAI-compatible HTTP server for iTaKTorch.
// It wraps an Engine and serves chat completions on localhost.
type Server struct {
	engine    Engine
	registry  *ModelRegistry // optional: multi-model serving
	scheduler *Scheduler
	port      int
	server    *http.Server
	startTime time.Time
	mu        sync.RWMutex
	logger    *log.Logger

	// Core integration (optional - nil when running standalone).
	eventBus        *event.Bus        // cross-module event bus
	registryClient  *registry.Client  // service registry for heartbeats
	healthChecker   *TorchHealthChecker
	heartbeatCancel  context.CancelFunc

	// Debug system (always available, gated by ITAK_DEBUG=1 env var).
	// The debug logger is used for all structured logging throughout Torch.
	// Its ring buffer feeds the /debug/logs endpoint for live readouts.
	debugLogger *coreDebug.Logger

	// Rate limiter (optional - nil means no rate limiting).
	// When set, /v1/chat/completions is rate-limited per client IP.
	rateLimiter *RateLimiter

	// HuggingFace puller (optional - nil disables pull/search endpoints).
	// When set, enables downloading models from HuggingFace Hub via API.
	hfPuller *HFPuller

	// Response cache (optional - nil means no caching).
	// When set, identical prompts return cached results instantly.
	responseCache *ResponseCache

	// Ollama puller (optional - nil disables Ollama pull/search endpoints).
	// When set, enables downloading models from the Ollama registry.
	ollamaPuller *OllamaPuller

	// LoRA adapter manager (optional - nil disables adapter endpoints).
	loraManager *LoRAManager

	// Prometheus metrics collector.
	metrics *PrometheusMetrics

	// Ollama API compatibility flag.
	// When true, /api/generate, /api/chat, /api/tags, /api/show, /api/version are mounted.
	ollamaCompat bool

	// Upstream Ollama URL for model aggregation.
	// When set, /api/tags merges local Torch models with models from this Ollama server.
	ollamaUpstream string

	// Training manager (optional - nil disables training endpoints).
	trainingMgr *TrainingManager
}

// ServerOption configures a Server.
type ServerOption func(*Server)

// WithLogger sets a debug logger for request/response tracing.
func WithLogger(l *log.Logger) ServerOption {
	return func(s *Server) { s.logger = l }
}

// WithRegistry enables multi-model serving via a ModelRegistry.
// When set, the server resolves the "model" field in requests to dynamically
// load engines from the models directory.
func WithRegistry(r *ModelRegistry) ServerOption {
	return func(s *Server) { s.registry = r }
}

// WithRateLimit enables per-IP rate limiting on the inference endpoint.
// requestsPerMinute controls the average allowed rate (e.g., 60 = 1 req/sec).
// burstSize controls how many rapid requests are allowed before throttling.
func WithRateLimit(requestsPerMinute, burstSize int) ServerOption {
	return func(s *Server) {
		s.rateLimiter = NewRateLimiter(requestsPerMinute, burstSize)
	}
}

// WithHFPuller enables HuggingFace model pull/search endpoints.
// cacheDir is where models are stored (empty = ~/.torch/models/).
// token is the optional HF API token for gated models.
func WithHFPuller(cacheDir, token string) ServerOption {
	return func(s *Server) {
		puller, err := NewHFPuller(cacheDir, token)
		if err != nil {
			fmt.Printf("[iTaK Torch] Warning: HF puller init failed: %v\n", err)
			return
		}
		s.hfPuller = puller
	}
}

// WithResponseCache enables prompt-level response caching.
// maxEntries controls how many unique prompts can be cached (default: 256).
func WithResponseCache(maxEntries int) ServerOption {
	return func(s *Server) {
		s.responseCache = NewResponseCache(maxEntries)
	}
}

// WithOllamaPuller enables Ollama registry pull/search endpoints.
// cacheDir is where models are stored (empty = ~/.torch/models/).
func WithOllamaPuller(cacheDir string) ServerOption {
	return func(s *Server) {
		puller, err := NewOllamaPuller(cacheDir)
		if err != nil {
			fmt.Printf("[iTaK Torch] Warning: Ollama puller init failed: %v\n", err)
			return
		}
		s.ollamaPuller = puller
	}
}

// WithOllamaCompat enables Ollama-compatible API routes.
// Registers /api/generate, /api/chat, /api/tags, /api/show, /api/version.
// This makes Torch a drop-in replacement for Ollama with existing tools.
func WithOllamaCompat() ServerOption {
	return func(s *Server) { s.ollamaCompat = true }
}

// WithOllamaUpstream sets an upstream Ollama URL for model aggregation.
// When set, /api/tags merges local Torch models with models from this Ollama server.
// Example: "http://localhost:11434" or "http://host.docker.internal:11434".
func WithOllamaUpstream(url string) ServerOption {
	return func(s *Server) { s.ollamaUpstream = url }
}

// WithEventBus connects the server to Core's event bus for cross-module events.
// When set, the server emits module.started, module.stopped, model.loaded,
// inference.done, and inference.error events.
func WithEventBus(bus *event.Bus) ServerOption {
	return func(s *Server) { s.eventBus = bus }
}

// WithRegistryClient connects the server to Core's service registry.
// When set, the server launches a heartbeat goroutine on Start() that
// periodically registers this Torch instance with the registry.
func WithRegistryClient(rc *registry.Client) ServerOption {
	return func(s *Server) { s.registryClient = rc }
}

// NewServer creates a iTaKTorch server bound to the given port.
func NewServer(engine Engine, port int, opts ...ServerOption) *Server {
	// Create scheduler: use continuous batching if the engine supports it.
	var scheduler *Scheduler
	if te, ok := engine.(*TorchEngine); ok && te.opts.MaxSlots > 1 {
		scheduler = NewBatchScheduler(te, 64, te.opts.MaxSlots)
	} else {
		scheduler = NewScheduler(engine, 64)
	}

	s := &Server{
		engine:        engine,
		scheduler:     scheduler,
		port:          port,
		debugLogger:   coreDebug.NewLogger("torch", coreDebug.LevelInfo),
		metrics:       NewPrometheusMetrics(),
		loraManager:   NewLoRAManager(),
		responseCache: NewResponseCache(256, 5*time.Minute),
	}
	for _, opt := range opts {
		opt(s)
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/", s.handleRoot)
	mux.HandleFunc("/v1/models", s.handleModels)
	mux.HandleFunc("/health", s.handleHealth)

	// Wrap the chat completions endpoint with rate limiting if configured.
	// Other endpoints (models, health) are NOT rate-limited.
	completionsHandler := http.HandlerFunc(s.handleChatCompletions)
	if s.rateLimiter != nil {
		mux.Handle("/v1/chat/completions", RateLimitMiddleware(completionsHandler, s.rateLimiter))
		fmt.Printf("[iTaK Torch] Rate limiting: %d req/min, burst %d\n",
			int(s.rateLimiter.tokensPerSecond*60), s.rateLimiter.burstSize)
	} else {
		mux.Handle("/v1/chat/completions", completionsHandler)
	}

	// Advanced inference endpoints.
	mux.HandleFunc("/v1/generate_raw", s.handleGenerateRaw)

	// Model management endpoints (Feature #1: HF Pull, Feature #2: Runtime Swap).
	mux.HandleFunc("/v1/models/pull", s.handleModelPull)
	mux.HandleFunc("/v1/models/search", s.handleModelSearch)
	mux.HandleFunc("/v1/models/load", s.handleModelLoad)
	mux.HandleFunc("/v1/models/unload", s.handleModelUnload)
	mux.HandleFunc("/v1/models/loaded", s.handleModelsLoaded)

	// Ollama registry endpoints (pull and search from Ollama's model library).
	mux.HandleFunc("/v1/models/pull/ollama", s.handleOllamaModelPull)
	mux.HandleFunc("/v1/models/search/ollama", s.handleOllamaModelSearch)

	// Operations endpoints (cache management, scheduler metrics).
	mux.HandleFunc("/v1/cache/clear", s.handleCacheClear)
	mux.HandleFunc("/v1/cache/stats", s.handleCacheStats)
	mux.HandleFunc("/v1/scheduler/stats", s.handleSchedulerStats)
	mux.HandleFunc("/v1/state/save", s.handleStateSave)
	mux.HandleFunc("/v1/state/load", s.handleStateLoad)

	// Embeddings endpoint (OpenAI-compatible, for RAG/semantic search).
	mux.HandleFunc("/v1/embeddings", s.handleEmbeddings)

	// Swarm endpoint (parallel batch inference with auto-strategy detection).
	mux.HandleFunc("/v1/swarm", s.handleSwarm)
	mux.HandleFunc("/v1/capabilities", s.handleCapabilities)

	// Distributed cluster (multi-node inference across LAN).
	mux.HandleFunc("/v1/cluster/join", s.handleClusterJoin)
	mux.HandleFunc("/v1/cluster/peers", s.handleClusterPeers)

	// LoRA adapter management endpoints.
	mux.HandleFunc("/v1/adapters", s.handleAdapterList)
	mux.HandleFunc("/v1/adapters/load", s.handleAdapterLoad)
	mux.HandleFunc("/v1/adapters/unload", s.handleAdapterUnload)

	// Prometheus-compatible metrics endpoint.
	mux.HandleFunc("/metrics", s.handleMetrics)

	// Training and fine-tuning endpoints.
	s.RegisterTrainingRoutes(mux)

	// Tokenize, detokenize, and model info endpoints.
	s.RegisterAdditionalRoutes(mux)

	// Ollama-compatible API routes (drop-in Ollama replacement).
	if s.ollamaCompat {
		s.RegisterOllamaRoutes(mux)
	}

	// Mount debug endpoints if ITAK_DEBUG=1 is set.
	// These provide live readouts at /debug/snapshot, /debug/logs, etc.
	if coreDebug.Enabled() {
		debugMux := coreDebug.NewMux(coreDebug.MuxConfig{
			Logger:    s.debugLogger,
			StartTime: time.Now(),
		})
		mux.Handle("/debug/", debugMux)
		s.debugLogger.Info("server", "Debug endpoints mounted at /debug/*")
	}

	s.server = &http.Server{
		Addr:         fmt.Sprintf("0.0.0.0:%d", port),
		Handler:      mux,
		ReadTimeout:  30 * time.Second,
		WriteTimeout: 120 * time.Second,
	}

	// Start scheduler immediately so requests can be processed
	// both via server.Start() and direct httptest.ServeHTTP().
	s.scheduler.Start()

	return s
}

// debugf logs a formatted debug message if a logger is set.
func (s *Server) debugf(format string, args ...interface{}) {
	if s.logger != nil {
		s.logger.Printf(format, args...)
	}
}

// writeError writes a JSON error response.
func (s *Server) writeError(w http.ResponseWriter, code int, message string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	fmt.Fprintf(w, `{"error": %q}`, message)
}

// writeJSON writes a generic JSON response using the bytes.Buffer pool to avoid allocations.
func (s *Server) writeJSON(w http.ResponseWriter, v interface{}) {
	buf := GetByteBuf()
	defer PutByteBuf(buf)

	if err := json.NewEncoder(buf).Encode(v); err != nil {
		s.writeError(w, http.StatusInternalServerError, "failed to marshal response")
		return
	}

	w.Header().Set("Content-Type", "application/json")
	w.Write(buf.Bytes())
}

// Start starts the HTTP server. Blocks until the server is stopped.
func (s *Server) Start() error {
	s.startTime = time.Now()
	fmt.Printf("[iTaK Torch] Server starting on http://localhost:%d\n", s.port)
	fmt.Printf("[iTaK Torch] Model: %s\n", s.engine.ModelName())
	fmt.Printf("[iTaK Torch] Endpoints:\n")
	fmt.Printf("  POST /v1/chat/completions\n")
	fmt.Printf("  GET  /v1/models\n")
	fmt.Printf("  GET  /health\n")

	// Core integration: emit module.started event.
	s.emitEvent(event.TypeModuleStarted, map[string]interface{}{
		"port":  s.port,
		"model": s.engine.ModelName(),
	})

	// Core integration: launch heartbeat goroutine.
	if s.registryClient != nil {
		s.healthChecker = NewTorchHealthChecker(s, "0.1.0")
		ctx, cancel := context.WithCancel(context.Background())
		s.heartbeatCancel = cancel

		backend := "cpu"
		metadata := map[string]string{"backend": backend}
		if te, ok := s.engine.(*TorchEngine); ok && te.opts.Backend != "" {
			metadata["backend"] = te.opts.Backend
		}

		go health.StartHeartbeat(ctx, health.HeartbeatConfig{
			Interval: 10 * time.Second,
			RegistryClient: s.registryClient,
			Endpoint: registry.ServiceEndpoint{
				Module:   "torch",
				Host:     "127.0.0.1",
				Port:     s.port,
				Protocol: registry.ProtocolREST,
				Healthy:  true,
				Metadata: metadata,
			},
			Checker: s.healthChecker,
		})
		fmt.Printf("[iTaK Torch] Heartbeat registered with service registry\n")
	}

	// Start cluster auto-discovery (UDP beacon + listener + health cleanup).
	// Detects LAN IP so other nodes can reach this server.
	if lanIP := detectLANIP(); lanIP != "" {
		selfAddr := fmt.Sprintf("%s:%d", lanIP, s.port)
		hostname, _ := os.Hostname()
		if hostname == "" {
			hostname = "torch-node"
		}
		s.StartClusterDiscovery(hostname, selfAddr)
	}

	err := s.server.ListenAndServe()
	if err == http.ErrServerClosed {
		return nil
	}
	return err
}

// detectLANIP returns this machine's primary LAN IP address.
// Used for cluster discovery beacons so peers know how to reach us.
func detectLANIP() string {
	addrs, err := net.InterfaceAddrs()
	if err != nil {
		return ""
	}
	for _, addr := range addrs {
		if ipNet, ok := addr.(*net.IPNet); ok && !ipNet.IP.IsLoopback() {
			if ip4 := ipNet.IP.To4(); ip4 != nil {
				return ip4.String()
			}
		}
	}
	return ""
}

// Stop gracefully shuts down the server.
func (s *Server) Stop() error {
	// Core integration: cancel heartbeat goroutine (auto-deregisters).
	if s.heartbeatCancel != nil {
		s.heartbeatCancel()
	}

	s.emitEvent(event.TypeModuleStopped, nil)

	s.scheduler.Stop()
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	return s.server.Shutdown(ctx)
}

// emitEvent publishes an event to the Core event bus if one is connected.
// No-op when running standalone without Core.
func (s *Server) emitEvent(typ event.Type, data interface{}) {
	if s.eventBus != nil {
		s.eventBus.Emit(event.New(typ, "torch", data))
	}
}

// Port returns the port the server is bound to.
func (s *Server) Port() int {
	return s.port
}

// buildChatResponse constructs an OpenAI-compatible ChatResponse.
// Used by both the normal path and the cache-hit path.
func buildChatResponse(model, text string, metrics *InferenceMetrics) ChatResponse {
	promptTokens := 0
	completionTokens := 0
	if metrics != nil {
		promptTokens = metrics.PromptTokens
		completionTokens = metrics.CompletionTokens
	}
	return ChatResponse{
		ID:      fmt.Sprintf("itaktorch-%d", time.Now().UnixNano()),
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   model,
		Choices: []ChatChoice{
			{
				Index: 0,
				Message: ChatMessage{
					Role:    "assistant",
					Content: text,
				},
				FinishReason: "stop",
			},
		},
		Usage: ChatUsage{
			PromptTokens:     promptTokens,
			CompletionTokens: completionTokens,
			TotalTokens:      promptTokens + completionTokens,
		},
	}
}

// handleChatCompletions handles POST /v1/chat/completions.
func (s *Server) handleChatCompletions(w http.ResponseWriter, r *http.Request) {
	start := time.Now()
	s.debugf("[REQ] %s %s from %s", r.Method, r.URL.Path, r.RemoteAddr)

	if r.Method != http.MethodPost {
		s.debugf("[ERR] method not allowed: %s", r.Method)
		s.writeError(w, http.StatusMethodNotAllowed, "method not allowed, use POST")
		return
	}

	// Read and parse request body.
	body, err := io.ReadAll(r.Body)
	if err != nil {
		s.debugf("[ERR] read body: %v", err)
		s.writeError(w, http.StatusBadRequest, "failed to read request body")
		return
	}
	defer r.Body.Close()

	var req ChatRequest
	if err := json.Unmarshal(body, &req); err != nil {
		s.debugf("[ERR] parse JSON: %v", err)
		s.writeError(w, http.StatusBadRequest, fmt.Sprintf("invalid JSON: %v", err))
		return
	}

	if len(req.Messages) == 0 {
		s.debugf("[ERR] empty messages")
		s.writeError(w, http.StatusBadRequest, "messages array is empty")
		return
	}

	s.debugf("[INF] model=%s msgs=%d max_tokens=%d", req.Model, len(req.Messages), req.MaxTokens)

	// Resolve engine: use registry if available, otherwise fall back to single engine.
	engine := s.engine
	if s.registry != nil && req.Model != "" {
		// Multi-model mode: resolve from registry.
		resolved, err := s.registry.GetOrLoad(req.Model)
		if err != nil {
			s.debugf("[ERR] model resolution: %v", err)
			s.writeError(w, http.StatusNotFound, fmt.Sprintf("model not found: %v", err))
			return
		}
		engine = resolved
	}

	// Build completion params from request.
	params := CompletionParams{
		MaxTokens: req.MaxTokens,
		Stop:      req.Stop,
	}
	if req.Temperature != nil {
		params.Temperature = *req.Temperature
	} else {
		params.Temperature = 0.7
	}
	if req.TopP != nil {
		params.TopP = *req.TopP
	} else {
		params.TopP = 0.9
	}
	if params.MaxTokens == 0 {
		params.MaxTokens = 512
	}

	// --- Response Cache Check (Feature #7) ---
	// If caching is enabled and client didn't send X-No-Cache, check the cache first.
	var cacheKey string
	useCache := s.responseCache != nil && r.Header.Get("X-No-Cache") != "true"

	if useCache {
		cacheKey = CacheKey(engine.ModelName(), req.Messages, params)
		if cachedText, cachedMetrics, ok := s.responseCache.Get(cacheKey); ok {
			// Cache HIT: return immediately without inference.
			w.Header().Set("X-Cache", "HIT")
			s.debugf("[CACHE] HIT for key %s (model=%s)", cacheKey[:8], engine.ModelName())

			resp := buildChatResponse(engine.ModelName(), cachedText, cachedMetrics)
			w.Header().Set("Content-Type", "application/json")
			s.writeJSON(w, resp)
			return
		}
	}

	// Submit to scheduler queue (Phase 4B: concurrent request handling).
	// In multi-model mode, create an ad-hoc scheduler for the resolved engine.
	scheduler := s.scheduler
	if s.registry != nil && engine != s.engine {
		scheduler = NewScheduler(engine, 64)
		scheduler.Start()
		defer scheduler.Stop()
	}

	inferReq := &InferenceRequest{
		Messages: req.Messages,
		Params:   params,
		Ctx:      r.Context(),
	}

	// SSE streaming path (streaming responses are NOT cached).
	if req.Stream {
		if useCache {
			w.Header().Set("X-Cache", "BYPASS-STREAM")
		}
		s.handleStreamingResponse(w, r, inferReq)
		return
	}

	scheduler.Submit(inferReq)

	// Wait for result from the scheduler.
	var inferResult InferenceResult
	select {
	case inferResult = <-inferReq.ResultCh:
	case <-r.Context().Done():
		s.debugf("[ERR] client disconnected while waiting in queue")
		return
	}

	result := inferResult.Text
	elapsed := time.Since(start)

	if inferResult.Err != nil {
		s.debugf("[ERR] inference failed after %s: %v", elapsed, inferResult.Err)
		s.emitEvent(event.TypeInferenceError, map[string]interface{}{
			"model": engine.ModelName(),
			"error": inferResult.Err.Error(),
			"elapsed_ms": elapsed.Milliseconds(),
		})
		s.writeError(w, http.StatusInternalServerError, fmt.Sprintf("inference error: %v", inferResult.Err))
		return
	}

	// Use actual token counts from engine metrics.
	runStats := engine.GetStats()

	// Build response using the shared helper.
	resp := buildChatResponse(engine.ModelName(), result, runStats.LastMetrics)

	// Store in cache (Feature #7) - only cache successful, non-empty responses.
	if useCache && cacheKey != "" && result != "" {
		s.responseCache.Put(cacheKey, result, runStats.LastMetrics)
	}

	// Log with performance data.
	tokSec := 0.0
	if runStats.LastMetrics != nil {
		tokSec = runStats.LastMetrics.TokensPerSecond
	}
	s.debugf("[RES] 200 OK in %s | %d tok | %.1f tok/s", elapsed, resp.Usage.TotalTokens, tokSec)

	// Core integration: emit inference.done event.
	s.emitEvent(event.TypeInferenceDone, map[string]interface{}{
		"model":      engine.ModelName(),
		"tokens":     resp.Usage.TotalTokens,
		"tok_per_sec": tokSec,
		"elapsed_ms": elapsed.Milliseconds(),
	})

	// Set cache status header.
	if useCache {
		w.Header().Set("X-Cache", "MISS")
	}

	w.Header().Set("Content-Type", "application/json")
	s.writeJSON(w, resp)
}

// handleStreamingResponse sends SSE (Server-Sent Events) for streaming chat completions.
// Follows OpenAI's streaming format: chat.completion.chunk objects with delta content.
func (s *Server) handleStreamingResponse(w http.ResponseWriter, r *http.Request, inferReq *InferenceRequest) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		s.writeError(w, http.StatusInternalServerError, "streaming not supported")
		return
	}

	// Create stream channel for token deltas.
	inferReq.StreamCh = make(chan string, 16)
	s.scheduler.Submit(inferReq)

	// Set SSE headers.
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("X-Accel-Buffering", "no")
	w.WriteHeader(http.StatusOK)

	id := fmt.Sprintf("itaktorch-%d", time.Now().UnixNano())
	model := s.engine.ModelName()
	created := time.Now().Unix()

	// Send role chunk first (OpenAI convention).
	roleChunk := fmt.Sprintf(`{"id":"%s","object":"chat.completion.chunk","created":%d,"model":"%s","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}`, id, created, model)
	fmt.Fprintf(w, "data: %s\n\n", roleChunk)
	flusher.Flush()

	// Stream token deltas with buffered flushing.
	// Instead of flushing every single token (high HTTP overhead), buffer for
	// up to 5ms and flush accumulated tokens as one chunk. At 134 tok/s,
	// each token takes ~7.5ms, so we typically flush every 1-2 tokens.
	// Under faster generation, we batch more tokens per flush.
	// JSONEscaper is a package-level singleton (mem_pool.go) to avoid per-call allocation.
	flushTimer := time.NewTimer(5 * time.Millisecond)
	flushTimer.Stop()
	var tokenBuf strings.Builder

	streamLoop:
	for {
		select {
		case delta, ok := <-inferReq.StreamCh:
			if !ok {
				// Channel closed: flush remaining buffer and exit.
				if tokenBuf.Len() > 0 {
					escaped := JSONEscaper.Replace(tokenBuf.String())
					chunk := fmt.Sprintf(`{"id":"%s","object":"chat.completion.chunk","created":%d,"model":"%s","choices":[{"index":0,"delta":{"content":"%s"},"finish_reason":null}]}`, id, created, model, escaped)
					fmt.Fprintf(w, "data: %s\n\n", chunk)
					flusher.Flush()
				}
				break streamLoop
			}
			tokenBuf.WriteString(delta)
			// Reset the flush timer for each token received.
			if !flushTimer.Stop() {
				select {
				case <-flushTimer.C:
				default:
				}
			}
			flushTimer.Reset(5 * time.Millisecond)

		case <-flushTimer.C:
			// Timer fired: flush buffered tokens.
			if tokenBuf.Len() > 0 {
				escaped := JSONEscaper.Replace(tokenBuf.String())
				chunk := fmt.Sprintf(`{"id":"%s","object":"chat.completion.chunk","created":%d,"model":"%s","choices":[{"index":0,"delta":{"content":"%s"},"finish_reason":null}]}`, id, created, model, escaped)
				fmt.Fprintf(w, "data: %s\n\n", chunk)
				flusher.Flush()
				tokenBuf.Reset()
			}
		}
	}

	// Wait for final result to get error status.
	select {
	case result := <-inferReq.ResultCh:
		if result.Err != nil {
			s.debugf("[ERR] streaming inference failed: %v", result.Err)
		}
	case <-r.Context().Done():
		return
	}

	// Send finish chunk.
	finishChunk := fmt.Sprintf(`{"id":"%s","object":"chat.completion.chunk","created":%d,"model":"%s","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`, id, created, model)
	fmt.Fprintf(w, "data: %s\n\n", finishChunk)
	fmt.Fprintf(w, "data: [DONE]\n\n")
	flusher.Flush()
}

// handleModels handles GET /v1/models.
func (s *Server) handleModels(w http.ResponseWriter, r *http.Request) {
	s.debugf("[REQ] %s %s from %s", r.Method, r.URL.Path, r.RemoteAddr)

	if r.Method != http.MethodGet {
		s.writeError(w, http.StatusMethodNotAllowed, "method not allowed, use GET")
		return
	}

	var data []ModelInfo
	if s.registry != nil {
		// Multi-model mode: list all available models from disk.
		data = s.registry.ListAvailable()
	} else {
		// Single-model mode: just the loaded model.
		data = []ModelInfo{
			{
				ID:      s.engine.ModelName(),
				Object:  "model",
				OwnedBy: "itaktorch",
			},
		}
	}

	resp := ModelsResponse{
		Object: "list",
		Data:   data,
	}

	s.debugf("[RES] 200 OK models=%d", len(resp.Data))

	w.Header().Set("Content-Type", "application/json")
	s.writeJSON(w, resp)
}

// handleHealth handles GET /health.
func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	s.debugf("[REQ] %s %s from %s", r.Method, r.URL.Path, r.RemoteAddr)
	uptime := time.Since(s.startTime).Round(time.Second).String()

	stats := s.engine.GetStats()
	currentRes := CaptureResources()

	// Extended health response with performance data.
	resp := map[string]interface{}{
		"status": "ok",
		"model":  s.engine.ModelName(),
		"uptime": uptime,
		"port":   s.port,
		"performance": map[string]interface{}{
			"model_load_time_ms": stats.ModelLoadTime.Milliseconds(),
			"request_count":      stats.RequestCount,
			"total_tokens_gen":   stats.TotalTokensGen,
			"avg_tokens_per_sec": fmt.Sprintf("%.1f", stats.AvgTokPerSec),
		},
		"resources": map[string]interface{}{
			"current": map[string]interface{}{
				"heap_mb":    fmt.Sprintf("%.1f", currentRes.HeapAllocMB),
				"sys_mb":     fmt.Sprintf("%.1f", currentRes.SysMB),
				"goroutines": currentRes.GoRoutines,
				"gc_cycles":  currentRes.NumGC,
			},
			"pre_model_load": map[string]interface{}{
				"heap_mb": fmt.Sprintf("%.1f", stats.PreLoadRes.HeapAllocMB),
				"sys_mb":  fmt.Sprintf("%.1f", stats.PreLoadRes.SysMB),
			},
			"post_model_load": map[string]interface{}{
				"heap_mb": fmt.Sprintf("%.1f", stats.PostLoadRes.HeapAllocMB),
				"sys_mb":  fmt.Sprintf("%.1f", stats.PostLoadRes.SysMB),
			},
		},
		"scheduler": map[string]interface{}{
			"queue_depth":       s.scheduler.QueueDepth(),
			"total_processed":   s.scheduler.Stats().TotalProcessed,
			"total_dropped":     s.scheduler.Stats().TotalDropped,
			"avg_wait_ms":       fmt.Sprintf("%.1f", s.scheduler.Stats().AvgWaitMs),
			"avg_processing_ms": fmt.Sprintf("%.1f", s.scheduler.Stats().AvgProcessingMs),
		},
	}

	// Add registry stats if multi-model mode is enabled.
	if s.registry != nil {
		rStats := s.registry.Stats()
		resp["registry"] = map[string]interface{}{
			"loaded_models": rStats.LoadedModels,
			"max_models":    rStats.MaxModels,
			"models_dir":    rStats.ModelsDir,
			"total_loads":   rStats.TotalLoads,
			"total_evicts":  rStats.TotalEvicts,
			"cache_hits":    rStats.CacheHits,
			"cache_misses":  rStats.CacheMisses,
			"loaded_names":  rStats.LoadedNames,
		}
	}

	// Add last request metrics if available.
	if stats.LastMetrics != nil {
		resp["last_request"] = map[string]interface{}{
			"prompt_tokens":     stats.LastMetrics.PromptTokens,
			"completion_tokens": stats.LastMetrics.CompletionTokens,
			"tokens_per_second": fmt.Sprintf("%.1f", stats.LastMetrics.TokensPerSecond),
			"prompt_ms":         stats.LastMetrics.PromptDuration.Milliseconds(),
			"gen_ms":            stats.LastMetrics.GenDuration.Milliseconds(),
			"total_ms":          stats.LastMetrics.TotalDuration.Milliseconds(),
		}
	}

	// Add GPU memory monitoring.
	gpuInventory := DetectGPUs()
	if len(gpuInventory.GPUs) > 0 {
		gpuInfo := make([]map[string]interface{}, 0)
		for _, gpu := range gpuInventory.GPUs {
			info := map[string]interface{}{
				"name":   gpu.Name,
				"vendor": gpu.Vendor,
			}
			if gpu.VRAMMiB > 0 {
				info["total_vram_mb"] = gpu.VRAMMiB
			}
			gpuInfo = append(gpuInfo, info)
		}
		resp["gpu"] = gpuInfo
	}

	w.Header().Set("Content-Type", "application/json")
	s.writeJSON(w, resp)
}

func errorTypeFromStatus(status int) string {
	switch {
	case status == 400:
		return "invalid_request_error"
	case status == 404:
		return "not_found_error"
	case status == 405:
		return "invalid_request_error"
	case status >= 500:
		return "server_error"
	default:
		return "api_error"
	}
}

// handleRoot serves the landing page on GET /.
// Returns a flashy HTML page in browsers, plain text for curl/API clients.
func (s *Server) handleRoot(w http.ResponseWriter, r *http.Request) {
	// Only match exact root path.
	if r.URL.Path != "/" {
		http.NotFound(w, r)
		return
	}

	s.debugf("[REQ] %s %s from %s", r.Method, r.URL.Path, r.RemoteAddr)

	// Plain text for curl/wget/API clients.
	userAgent := r.Header.Get("User-Agent")
	if !strings.Contains(userAgent, "Mozilla") {
		w.Header().Set("Content-Type", "text/plain")
		fmt.Fprintf(w, "iTaKTorch is running.\nUptime: %s\n", time.Since(s.startTime).Round(time.Second))
		return
	}

	// HTML landing page for browsers.
	uptime := time.Since(s.startTime).Round(time.Second).String()
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	fmt.Fprintf(w, landingPageHTML, uptime, s.port)
}

const landingPageHTML = `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>iTaKTorch</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    background: #0a0a0f;
    color: #e0e0e0;
    min-height: 100vh;
    display: flex; align-items: center; justify-content: center;
    overflow: hidden;
  }
  .bg-glow {
    position: fixed; top: 50%%; left: 50%%; transform: translate(-50%%, -50%%);
    width: 600px; height: 600px; border-radius: 50%%;
    background: radial-gradient(circle, rgba(255,100,0,0.08) 0%%, transparent 70%%);
    animation: breathe 4s ease-in-out infinite;
  }
  @keyframes breathe { 0%%,100%% { transform: translate(-50%%,-50%%) scale(1); opacity:0.6; } 50%% { transform: translate(-50%%,-50%%) scale(1.15); opacity:1; } }
  .card {
    position: relative; z-index: 1;
    text-align: center; padding: 48px 56px;
    background: rgba(18,18,28,0.85); border: 1px solid rgba(255,100,0,0.15);
    border-radius: 20px;
    backdrop-filter: blur(20px);
    box-shadow: 0 0 80px rgba(255,80,0,0.06);
  }
  .logo {
    font-size: 52px; font-weight: 800; letter-spacing: -2px;
    background: linear-gradient(135deg, #ff6a00, #ff3d00, #ff6a00);
    background-size: 200%% 200%%;
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    animation: shimmer 3s ease-in-out infinite;
  }
  @keyframes shimmer { 0%%,100%% { background-position: 0%% 50%%; } 50%% { background-position: 100%% 50%%; } }
  .torch-icon {
    display: inline-block; font-size: 40px; margin-bottom: 8px;
    animation: flicker 2s ease-in-out infinite;
  }
  @keyframes flicker { 0%%,100%% { opacity:1; transform:scale(1); } 30%% { opacity:0.85; transform:scale(0.97); } 60%% { opacity:1; transform:scale(1.02); } }
  .status {
    display: inline-flex; align-items: center; gap: 8px;
    margin: 20px 0 24px; padding: 8px 20px;
    background: rgba(0,200,80,0.08); border: 1px solid rgba(0,200,80,0.25);
    border-radius: 50px; font-size: 14px; color: #4ade80;
  }
  .pulse {
    width: 8px; height: 8px; border-radius: 50%%;
    background: #4ade80; position: relative;
  }
  .pulse::after {
    content: ''; position: absolute; inset: -4px;
    border-radius: 50%%; background: rgba(74,222,128,0.3);
    animation: ping 2s cubic-bezier(0,0,0.2,1) infinite;
  }
  @keyframes ping { 75%%,100%% { transform: scale(2.5); opacity: 0; } }
  .info { margin: 16px 0 0; }
  .info-row {
    display: flex; justify-content: space-between; padding: 8px 0;
    border-bottom: 1px solid rgba(255,255,255,0.05); font-size: 14px;
  }
  .info-row:last-child { border-bottom: none; }
  .label { color: #888; } .value { color: #fff; font-family: 'Cascadia Code', monospace; }
  .endpoints { margin-top: 24px; text-align: left; }
  .endpoints h3 { font-size: 12px; text-transform: uppercase; letter-spacing: 2px; color: #666; margin-bottom: 12px; }
  .ep {
    display: flex; align-items: center; gap: 10px;
    padding: 6px 0; font-size: 13px; font-family: 'Cascadia Code', monospace;
  }
  .method {
    padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 700;
  }
  .method-post { background: rgba(59,130,246,0.15); color: #60a5fa; }
  .method-get { background: rgba(74,222,128,0.15); color: #4ade80; }
  .ecosystem { margin-top: 24px; text-align: left; }
  .ecosystem h3 { font-size: 12px; text-transform: uppercase; letter-spacing: 2px; color: #666; margin-bottom: 12px; }
  .eco-links { display: flex; flex-wrap: wrap; gap: 8px; }
  .eco-link {
    padding: 4px 12px; border-radius: 6px; font-size: 12px;
    background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08);
    color: #aaa; text-decoration: none;
    transition: all 0.2s;
  }
  .eco-link:hover { background: rgba(255,100,0,0.1); border-color: rgba(255,100,0,0.3); color: #ff6a00; }
  .footer { margin-top: 28px; font-size: 11px; color: #444; }
</style>
</head>
<body>
<div class="bg-glow"></div>
<div class="card">
  <div class="torch-icon">&#128293;</div>
  <div class="logo">iTaKTorch</div>
  <div class="status"><div class="pulse"></div> Running</div>
  <div class="info">
    <div class="info-row"><span class="label">Uptime</span><span class="value">%s</span></div>
    <div class="info-row"><span class="label">Port</span><span class="value">%d</span></div>
  </div>
  <div class="endpoints">
    <h3>Endpoints</h3>
    <div class="ep"><span class="method method-post">POST</span> /v1/chat/completions</div>
    <div class="ep"><span class="method method-get">GET</span> /v1/models</div>
    <div class="ep"><span class="method method-get">GET</span> /health</div>
  </div>
  <div class="ecosystem">
    <h3>Ecosystem</h3>
    <div class="eco-links">
      <a class="eco-link" href="https://github.com/David2024patton/iTaKAgent" target="_blank">iTaK Agent</a>
      <a class="eco-link" href="https://github.com/David2024patton/iTaKTorch" target="_blank">iTaKTorch</a>
      <a class="eco-link" href="https://github.com/David2024patton/iTaKBrowser" target="_blank">iTaK Browser</a>
      <a class="eco-link" href="https://github.com/David2024patton/iTaKDashboard" target="_blank">iTaK Dashboard</a>
      <a class="eco-link" href="https://github.com/David2024patton/iTaKMedia" target="_blank">iTaK Media</a>
      <a class="eco-link" href="https://github.com/David2024patton/iTaKForge" target="_blank">iTaK Forge</a>
      <a class="eco-link" href="https://github.com/David2024patton/iTaKGateway" target="_blank">iTaK Gateway</a>
    </div>
  </div>
  <div class="footer">iTaK Torch | Go-Native Inference Engine</div>
</div>
</body>
</html>`

// BuildPrompt converts chat messages into a single prompt string
// for models that don't support chat format natively.
func BuildPrompt(messages []ChatMessage) string {
	var sb strings.Builder
	for _, m := range messages {
		switch m.Role {
		case "system":
			sb.WriteString(fmt.Sprintf("System: %s\n\n", m.Content))
		case "user":
			sb.WriteString(fmt.Sprintf("User: %s\n\n", m.Content))
		case "assistant":
			sb.WriteString(fmt.Sprintf("Assistant: %s\n\n", m.Content))
		default:
			sb.WriteString(fmt.Sprintf("%s: %s\n\n", m.Role, m.Content))
		}
	}
	sb.WriteString("Assistant: ")
	return sb.String()
}
