package torch

import (
	"context"
	"fmt"
	"strings"

	"github.com/David2024patton/iTaKCore/pkg/types"
)

// TorchAdapter wraps Torch's Engine and ModelRegistry behind Core's
// InferenceEngine interface. This is the bridge that lets the Agent
// call Torch via the same interface it uses for Gateway or Ollama.
//
// Usage:
//
//	engine, _ := torch.NewTorchEngine(modelPath, opts)
//	adapter := torch.NewTorchAdapter(engine, registry, "vulkan")
//	// adapter implements contract.InferenceEngine
type TorchAdapter struct {
	engine   Engine
	registry *ModelRegistry
	backend  string // "vulkan", "cuda", "cpu"
}

// NewTorchAdapter creates an adapter that implements contract.InferenceEngine.
// If registry is nil, only the default engine is used (single-model mode).
func NewTorchAdapter(engine Engine, registry *ModelRegistry, backend string) *TorchAdapter {
	if backend == "" {
		backend = "cpu"
	}
	return &TorchAdapter{
		engine:   engine,
		registry: registry,
		backend:  backend,
	}
}

// EngineID returns a stable identifier for this engine instance.
func (a *TorchAdapter) EngineID() string {
	id := "torch-" + a.backend
	LogTrace("[CoreAdapter] EngineID() = %s", id)
	return id
}

// Infer sends a synchronous chat completion request.
// Converts Core types to Torch types, delegates to the engine, converts back.
func (a *TorchAdapter) Infer(ctx context.Context, req types.InferenceRequest) (types.InferenceResponse, error) {
	defer TimeTrace("CoreAdapter.Infer")()
	LogTrace("[CoreAdapter] Infer: model=%s msgs=%d max_tokens=%d temp=%.2f",
		req.Model, len(req.Messages), req.MaxTokens, req.Temperature)

	// Resolve engine for the requested model.
	engine, err := a.resolveEngine(string(req.Model))
	if err != nil {
		LogTrace("[CoreAdapter] Infer: engine resolve failed: %v", err)
		return types.InferenceResponse{}, err
	}

	// Convert Core messages to Torch messages.
	torchMsgs := coreMsgsToTorch(req.Messages)
	params := coreParamsToTorch(req)
	LogTrace("[CoreAdapter] Infer: converted %d messages, dispatching to engine %q",
		len(torchMsgs), engine.ModelName())

	// Run inference.
	result, err := engine.Complete(ctx, torchMsgs, params)
	if err != nil {
		LogTrace("[CoreAdapter] Infer: engine error: %v", err)
		return types.InferenceResponse{}, fmt.Errorf("torch infer: %w", err)
	}

	// Build response from engine stats.
	stats := engine.GetStats()
	resp := types.InferenceResponse{
		Model: req.Model,
		Message: types.ChatMessage{
			Role:    types.RoleAssistant,
			Content: result,
		},
		Done: true,
	}

	// Fill usage from engine metrics.
	if stats.LastMetrics != nil {
		resp.Usage = types.TokenUsage{
			PromptTokens:     stats.LastMetrics.PromptTokens,
			CompletionTokens: stats.LastMetrics.CompletionTokens,
			TotalTokens:      stats.LastMetrics.PromptTokens + stats.LastMetrics.CompletionTokens,
		}
		resp.Latency = types.LatencyStats{
			PromptEvalMs: float64(stats.LastMetrics.PromptDuration.Milliseconds()),
			GenerationMs: float64(stats.LastMetrics.GenDuration.Milliseconds()),
			TotalMs:      float64(stats.LastMetrics.TotalDuration.Milliseconds()),
			TokensPerSec: stats.LastMetrics.TokensPerSecond,
		}
	}

	LogTrace("[CoreAdapter] Infer: done, usage=%d prompt + %d completion tokens",
		resp.Usage.PromptTokens, resp.Usage.CompletionTokens)
	return resp, nil
}

// InferStream sends a streaming chat completion request. Returns a channel
// that yields partial responses as tokens are generated.
func (a *TorchAdapter) InferStream(ctx context.Context, req types.InferenceRequest) (<-chan types.InferenceResponse, error) {
	LogTrace("[CoreAdapter] InferStream: model=%s msgs=%d", req.Model, len(req.Messages))

	engine, err := a.resolveEngine(string(req.Model))
	if err != nil {
		LogTrace("[CoreAdapter] InferStream: engine resolve failed: %v", err)
		return nil, err
	}

	torchMsgs := coreMsgsToTorch(req.Messages)
	params := coreParamsToTorch(req)

	// Create the Torch-internal streaming request.
	streamCh := make(chan string, 16)
	resultCh := make(chan InferenceResult, 1)
	inferReq := &InferenceRequest{
		Messages: torchMsgs,
		Params:   params,
		Ctx:      ctx,
		StreamCh: streamCh,
		ResultCh: resultCh,
	}

	// Submit to a one-off scheduler for this engine.
	sched := NewScheduler(engine, 64)
	sched.Start()
	sched.Submit(inferReq)

	// Relay stream tokens as Core InferenceResponses.
	out := make(chan types.InferenceResponse, 16)
	go func() {
		defer close(out)
		defer sched.Stop()

		for delta := range streamCh {
			select {
			case out <- types.InferenceResponse{
				Model: req.Model,
				Message: types.ChatMessage{
					Role:    types.RoleAssistant,
					Content: delta,
				},
				Done: false,
			}:
			case <-ctx.Done():
				return
			}
		}

		// Wait for the final result to get usage stats.
		select {
		case result := <-resultCh:
			if result.Err != nil {
				return
			}
			stats := engine.GetStats()
			final := types.InferenceResponse{
				Model: req.Model,
				Message: types.ChatMessage{
					Role:    types.RoleAssistant,
					Content: result.Text,
				},
				Done: true,
			}
			if stats.LastMetrics != nil {
				final.Usage = types.TokenUsage{
					PromptTokens:     stats.LastMetrics.PromptTokens,
					CompletionTokens: stats.LastMetrics.CompletionTokens,
					TotalTokens:      stats.LastMetrics.PromptTokens + stats.LastMetrics.CompletionTokens,
				}
				final.Latency = types.LatencyStats{
					TotalMs:      float64(stats.LastMetrics.TotalDuration.Milliseconds()),
					TokensPerSec: stats.LastMetrics.TokensPerSecond,
				}
			}
			select {
			case out <- final:
			case <-ctx.Done():
			}
		case <-ctx.Done():
		}
	}()

	return out, nil
}

// ListModels returns all models available on this engine (from disk).
func (a *TorchAdapter) ListModels(ctx context.Context) ([]types.ModelInfo, error) {
	LogTrace("[CoreAdapter] ListModels called")
	if a.registry != nil {
		torchModels := a.registry.ListAvailable()
		coreModels := make([]types.ModelInfo, len(torchModels))
		for i, m := range torchModels {
			coreModels[i] = torchModelToCore(m)
		}
		LogTrace("[CoreAdapter] ListModels: %d models from registry", len(coreModels))
		return coreModels, nil
	}

	// Single-model mode: just the loaded engine.
	return []types.ModelInfo{
		{
			ID:      types.ModelID(a.engine.ModelName()),
			Backend: types.ModelBackend(a.backend),
			Loaded:  true,
		},
	}, nil
}

// LoadModel loads a model into memory via the ModelRegistry.
func (a *TorchAdapter) LoadModel(ctx context.Context, id types.ModelID, params types.ModelParams) error {
	defer TimeTrace(fmt.Sprintf("CoreAdapter.LoadModel(%s)", id))()
	LogTrace("[CoreAdapter] LoadModel: id=%s backend=%s", id, params.Backend)

	if a.registry == nil {
		// Single-model mode: check if the requested model is already loaded.
		if a.engine.ModelName() == string(id) {
			LogTrace("[CoreAdapter] LoadModel: already loaded (single-model mode)")
			return nil
		}
		return fmt.Errorf("torch: single-model mode, cannot load %q (loaded: %q)", id, a.engine.ModelName())
	}

	_, err := a.registry.GetOrLoad(string(id))
	if err != nil {
		LogTrace("[CoreAdapter] LoadModel: failed: %v", err)
	} else {
		LogTrace("[CoreAdapter] LoadModel: success")
	}
	return err
}

// UnloadModel removes a model from memory.
func (a *TorchAdapter) UnloadModel(ctx context.Context, id types.ModelID) error {
	LogTrace("[CoreAdapter] UnloadModel: id=%s", id)
	if a.registry == nil {
		return fmt.Errorf("torch: single-model mode, cannot unload")
	}
	err := a.registry.Unload(string(id))
	if err != nil {
		LogTrace("[CoreAdapter] UnloadModel: failed: %v", err)
	} else {
		LogTrace("[CoreAdapter] UnloadModel: success")
	}
	return err
}

// resolveEngine picks the right Torch Engine for a model name.
func (a *TorchAdapter) resolveEngine(model string) (Engine, error) {
	if a.registry != nil && model != "" {
		LogTrace("[CoreAdapter] resolveEngine: looking up %q in registry", model)
		return a.registry.GetOrLoad(model)
	}
	LogTrace("[CoreAdapter] resolveEngine: using default engine %q", a.engine.ModelName())
	return a.engine, nil
}

// ─── Type Converters ───────────────────────────────────────────────
//
// These functions translate between Core's types (used by Agent, Dashboard)
// and Torch's internal types. Why do we need this?
//
// Core defines standard types that ALL modules use (types.ChatMessage).
// Torch has its OWN ChatMessage type optimized for its inference engine.
// The adapter sits at the boundary and converts between them.
//
// Core <-> Torch type mapping:
//   types.ChatMessage      <-> torch.ChatMessage
//   types.InferenceRequest  -> torch.CompletionParams
//   torch.ModelInfo         -> types.ModelInfo
// ────────────────────────────────────────────────────────────────────

// coreMsgToTorch converts a single Core ChatMessage to Torch's ChatMessage.
func coreMsgToTorch(m types.ChatMessage) ChatMessage {
	return ChatMessage{
		Role:    string(m.Role),
		Content: m.Content,
	}
}

// coreMsgsToTorch converts a slice of Core ChatMessages to Torch format.
func coreMsgsToTorch(msgs []types.ChatMessage) []ChatMessage {
	result := make([]ChatMessage, len(msgs))
	for i, m := range msgs {
		result[i] = coreMsgToTorch(m)
	}
	return result
}

// torchMsgToCore converts a Torch ChatMessage to Core format.
func torchMsgToCore(m ChatMessage) types.ChatMessage {
	return types.ChatMessage{
		Role:    types.Role(m.Role),
		Content: m.Content,
	}
}

// coreParamsToTorch extracts the generation parameters from an InferenceRequest
// and converts them to Torch's CompletionParams format.
// Only Temperature and MaxTokens are currently mapped. Other params (top_p,
// frequency_penalty, etc.) can be added here as Torch gains support for them.
func coreParamsToTorch(req types.InferenceRequest) CompletionParams {
	return CompletionParams{
		MaxTokens:   req.MaxTokens,
		Temperature: float64(req.Temperature),
	}
}

// torchModelToCore converts Torch ModelInfo to Core types.ModelInfo.
// Because Torch loads GGUF files directly from disk, we don't have metadata
// embedded in the model file. Instead, we parse the GGUF FILENAME to detect:
//
//   - Parameter count (e.g., "8b" in "qwen3-8b-instruct-q4_k_m.gguf" = 8 billion)
//   - Quantization format (e.g., "q4_k_m" = 4-bit K-quant medium, smallest useful size)
//   - Model family (e.g., "qwen", "llama", "gemma" for vendor identification)
//
// This is a best-effort heuristic. Custom-named model files may not be detected.
func torchModelToCore(m ModelInfo) types.ModelInfo {
	info := types.ModelInfo{
		ID:     types.ModelID(m.ID),
		Loaded: false, // ListAvailable shows disk models, not necessarily loaded ones
	}

	// --- Detect parameter count from filename ---
	// GGUF convention: model names include "8b", "70b", etc. to indicate
	// the number of parameters (in billions).
	lower := strings.ToLower(m.ID)
	switch {
	case strings.Contains(lower, "70b"):
		info.Params = 70_000_000_000 // 70 billion params (needs ~40GB VRAM at Q4)
	case strings.Contains(lower, "32b"):
		info.Params = 32_000_000_000 // 32 billion params (needs ~20GB VRAM at Q4)
	case strings.Contains(lower, "14b"):
		info.Params = 14_000_000_000 // 14 billion params (needs ~10GB VRAM at Q4)
	case strings.Contains(lower, "8b"):
		info.Params = 8_000_000_000  // 8 billion params (sweet spot for consumer GPUs)
	case strings.Contains(lower, "3b"):
		info.Params = 3_000_000_000  // 3 billion params (fast, lower quality)
	case strings.Contains(lower, "1.5b"):
		info.Params = 1_500_000_000  // 1.5 billion params (very fast, basic tasks)
	case strings.Contains(lower, "0.6b"), strings.Contains(lower, "0.5b"):
		info.Params = 600_000_000    // 0.5-0.6B params (ultra-fast, limited quality)
	}

	// --- Detect quantization format from filename ---
	// Quantization reduces model size by using fewer bits per weight.
	// Common formats (from smallest/fastest to largest/most accurate):
	//   Q2_K  = 2-bit (very lossy, ~25% original size)
	//   Q3_K  = 3-bit (lossy but usable)
	//   Q4_K_M = 4-bit K-quant medium (best balance of size vs quality)
	//   Q5_K  = 5-bit (high quality, larger)
	//   Q6_K  = 6-bit (near-lossless)
	//   Q8_0  = 8-bit (minimal quality loss)
	//   FP16  = 16-bit float (full precision, largest)
	for _, q := range []string{"q4_k_m", "q4_k_s", "q5_k_m", "q5_k_s", "q6_k", "q8_0", "fp16", "f16", "q2_k", "q3_k_s", "q3_k_m", "q3_k_l", "iq4_xs"} {
		if strings.Contains(lower, q) {
			info.Quantized = strings.ToUpper(q)
			break
		}
	}

	// --- Detect model family from filename ---
	// Family helps the Agent know which model architecture is running,
	// which affects tokenizer choice, context window size, and capabilities.
	families := []string{"qwen", "llama", "gemma", "mistral", "phi", "deepseek", "yi", "command"}
	for _, fam := range families {
		if strings.Contains(lower, fam) {
			info.Family = fam
			break
		}
	}

	return info
}
