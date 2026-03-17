// feature_hub.go is the central integration point for all Torch engine features.
//
// WHAT: Instead of each feature being standalone, FeatureHub wires them together
// into a coherent system. It initializes all subsystems on startup and provides
// a single access point for the server and engine to use them.
//
// WIRED FEATURES:
//   Batch 3: Logit processors, token budget, watermark, structured logging,
//            priority queue, retry, dynamic quantization
//   Batch 4: Guardrails, request tracing, model warmup, A/B routing,
//            model versioning, semantic cache, tensor memoization, vision input
package torch

import (
	"context"
	"fmt"
	"time"

	"github.com/David2024patton/iTaKTorch/pkg/torch/native"
)

// FeatureHub centralizes all Torch engine features.
type FeatureHub struct {
	// Sampling and decoding.
	LogitChain *native.LogitChain

	// Security and metering.
	TokenBudget *TokenBudgetManager
	Guardrails  *native.Guardrails

	// Production infrastructure.
	StructuredLog *StructuredLogger
	PriorityQueue *RequestScheduler
	RetryExec     *RetryExecutor
	Shutdown      *ShutdownManager

	// Intelligent caching.
	SemanticCache *native.SemanticCache
	TensorMemo    *native.TensorMemo

	// Operations.
	Tracer         *TraceCollector
	ABRouter       *ABRouter
	VersionManager *ModelVersionManager
	VisionProc     *native.ImageProcessor

	// Internal state.
	config  FeatureConfig
	started bool
}

// FeatureConfig controls which features are enabled.
type FeatureConfig struct {
	// Sampling.
	EnableLogitChain  bool
	Temperature       float32
	TopK              int
	TopP              float32
	MinP              float32
	RepetitionPenalty float32

	// Safety.
	EnableGuardrails bool
	GuardrailConfig  native.GuardrailConfig

	// Token tracking.
	EnableTokenBudget bool
	DefaultTokenLimit int64

	// Logging.
	EnableStructuredLog bool
	LogLevel            string

	// Caching.
	EnableSemanticCache bool
	SemanticThreshold   float32
	EnableTensorMemo    bool

	// Operations.
	EnableTracing  bool
	MaxTraces      int
	EnableABRouter bool
	ABRoutes       []ABRoute
	DrainTimeout   time.Duration

	// Vision.
	EnableVision bool
	VisionConfig native.ImageConfig

	// Warmup.
	EnableWarmup  bool
	WarmupConfig  WarmupConfig
}

// DefaultFeatureConfig returns recommended settings for all features.
func DefaultFeatureConfig() FeatureConfig {
	return FeatureConfig{
		EnableLogitChain:    true,
		Temperature:         0.7,
		TopK:                40,
		TopP:                0.9,
		MinP:                0.05,
		RepetitionPenalty:   1.1,
		EnableGuardrails:    true,
		GuardrailConfig:     native.DefaultGuardrailConfig(),
		EnableTokenBudget:   false,
		DefaultTokenLimit:   100000,
		EnableStructuredLog: true,
		LogLevel:            "info",
		EnableSemanticCache: false,
		SemanticThreshold:   0.95,
		EnableTensorMemo:    true,
		EnableTracing:       true,
		MaxTraces:           1000,
		EnableABRouter:      false,
		DrainTimeout:        30 * time.Second,
		EnableVision:        false,
		VisionConfig:        native.DefaultLLaVAConfig(),
		EnableWarmup:        true,
		WarmupConfig:        DefaultWarmupConfig(),
	}
}

// NewFeatureHub creates and initializes all enabled features.
func NewFeatureHub(config FeatureConfig) *FeatureHub {
	hub := &FeatureHub{config: config}

	// Sampling chain.
	if config.EnableLogitChain {
		hub.LogitChain = native.NewLogitChain()
		hub.LogitChain.Add(native.TemperatureProcessor(config.Temperature))
		if config.TopK > 0 {
			hub.LogitChain.Add(native.TopKProcessor(config.TopK))
		}
		if config.TopP > 0 && config.TopP < 1.0 {
			hub.LogitChain.Add(native.TopPProcessor(config.TopP))
		}
		if config.MinP > 0 {
			hub.LogitChain.Add(native.MinPProcessor(config.MinP))
		}
		fmt.Println("[FeatureHub] Logit processor chain initialized")
	}

	// Guardrails.
	if config.EnableGuardrails {
		hub.Guardrails = native.NewGuardrails(config.GuardrailConfig)
		fmt.Println("[FeatureHub] Guardrails enabled")
	}

	// Token budget.
	if config.EnableTokenBudget {
		hub.TokenBudget = NewTokenBudgetManager(DefaultBudgetLimits())
		fmt.Println("[FeatureHub] Token budget tracking enabled")
	}

	// Structured logging.
	if config.EnableStructuredLog {
		hub.StructuredLog = NewStructuredLogger("torch")
		fmt.Println("[FeatureHub] Structured logging enabled")
	}

	// Semantic cache.
	if config.EnableSemanticCache {
		scConfig := native.DefaultSemanticCacheConfig()
		scConfig.SimilarityThresh = config.SemanticThreshold
		hub.SemanticCache = native.NewSemanticCache(scConfig)
		fmt.Println("[FeatureHub] Semantic cache enabled")
	}

	// Tensor memoization.
	if config.EnableTensorMemo {
		hub.TensorMemo = native.NewTensorMemo(native.DefaultTensorMemoConfig())
		fmt.Println("[FeatureHub] Tensor memoization enabled")
	}

	// Tracing.
	if config.EnableTracing {
		hub.Tracer = NewTraceCollector(config.MaxTraces)
		fmt.Println("[FeatureHub] Request tracing enabled")
	}

	// A/B routing.
	if config.EnableABRouter && len(config.ABRoutes) > 0 {
		hub.ABRouter = NewABRouter(ABConfig{
			Routes:    config.ABRoutes,
			Overrides: make(map[string]string),
			Enabled:   true,
		})
		fmt.Println("[FeatureHub] A/B routing enabled")
	}

	// Model versioning.
	hub.VersionManager = NewModelVersionManager()

	// Graceful shutdown.
	hub.Shutdown = NewShutdownManager(config.DrainTimeout)

	// Vision processor.
	if config.EnableVision {
		hub.VisionProc = native.NewImageProcessor(config.VisionConfig)
		fmt.Println("[FeatureHub] Vision input enabled")
	}

	hub.started = true
	fmt.Println("[FeatureHub] All features initialized")
	return hub
}

// RunWarmupIfEnabled runs model warmup if configured.
func (hub *FeatureHub) RunWarmupIfEnabled(ctx context.Context, engine Engine) *WarmupResult {
	if !hub.config.EnableWarmup {
		return nil
	}
	result := RunWarmup(ctx, engine, hub.config.WarmupConfig)
	return &result
}

// CheckInput runs guardrails on user input. Returns nil if safe.
func (hub *FeatureHub) CheckInput(text string) []native.GuardrailViolation {
	if hub.Guardrails == nil {
		return nil
	}
	return hub.Guardrails.CheckInput(text)
}

// CheckOutput runs guardrails on model output and optionally redacts PII.
func (hub *FeatureHub) CheckOutput(text string) (string, []native.GuardrailViolation) {
	if hub.Guardrails == nil {
		return text, nil
	}
	violations := hub.Guardrails.CheckOutput(text)
	for _, v := range violations {
		if v.Action == "redact" {
			text = hub.Guardrails.RedactPII(text)
		}
	}
	return text, violations
}

// StartTrace creates a new request trace.
func (hub *FeatureHub) StartTrace(operation string) *RequestTrace {
	if hub.Tracer == nil {
		return nil
	}
	return NewRequestTrace(operation)
}

// FinishTrace records a completed trace.
func (hub *FeatureHub) FinishTrace(trace *RequestTrace) {
	if hub.Tracer == nil || trace == nil {
		return
	}
	trace.Finish()
	hub.Tracer.Add(trace)
}

// Stats returns metrics for all features.
func (hub *FeatureHub) Stats() map[string]interface{} {
	stats := map[string]interface{}{
		"features_active": hub.started,
	}

	if hub.SemanticCache != nil {
		stats["semantic_cache"] = hub.SemanticCache.Stats()
	}
	if hub.TensorMemo != nil {
		stats["tensor_memo"] = hub.TensorMemo.Stats()
	}
	if hub.ABRouter != nil {
		stats["ab_routing"] = hub.ABRouter.Stats()
	}
	if hub.VersionManager != nil {
		active := hub.VersionManager.ActiveVersion()
		if active != nil {
			stats["active_model_version"] = active.ID
		}
	}
	if hub.Shutdown != nil {
		stats["active_requests"] = hub.Shutdown.ActiveCount()
	}

	return stats
}
