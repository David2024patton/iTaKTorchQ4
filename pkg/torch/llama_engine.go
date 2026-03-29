package torch

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"runtime"
	"strings"
	"sync"
	"time"
	"unsafe"

	"github.com/David2024patton/iTaKTorchQ4/pkg/torch/llama"
	"github.com/David2024patton/iTaKTorchQ4/pkg/torch/tokenizer"
)

// TorchEngine implements the Engine interface using the forked yzma/llama.cpp
// bindings via purego/ffi. No CGo required.
type TorchEngine struct {
	model     llama.Model
	ctx       llama.Context
	vocab     llama.Vocab
	sampler   llama.Sampler
	tokenBuf  []byte // pre-allocated buffer for TokenToPiece (eliminates per-token heap alloc)
	modelName string
	modelPath string
	opts      EngineOpts
	mu        sync.Mutex
	loaded    bool
	Stats     EngineStats

	// Speculative decoding (Phase 3 Stretch)
	draftModel   llama.Model
	draftCtx     llama.Context
	draftVocab   llama.Vocab
	draftSampler llama.Sampler
	hasDraft     bool

	// Go-native tokenizer (Phase 4A) - eliminates FFI overhead.
	goTokenizer    *tokenizer.GoTokenizer
	hasGoTokenizer bool

	// Prefix cache (Phase 4B) - reuses KV state for shared system prompts.
	prefixCache *PrefixCache

	// Chat template (auto-detected from model metadata).
	chatTemplate *ChatTemplate

	// streamCh receives token deltas during streaming inference.
	streamCh chan string
}

// tokenToText converts a token ID to its text representation.
// Phase 4A: uses Go-native vocab lookup (zero FFI) when available.
// Phase 7B: uses unsafe.String to eliminate per-token heap allocation.
//
//go:nosplit
func (e *TorchEngine) tokenToText(token llama.Token) string {
	if e.hasGoTokenizer {
		return e.goTokenizer.DecodeToken(int32(token))
	}
	// FFI fallback with zero-copy string (unsafe.String avoids heap alloc).
	n := llama.TokenToPiece(e.vocab, token, e.tokenBuf, 0, true)
	if n > 0 {
		// unsafe.String: returns a string header pointing directly at tokenBuf memory.
		// Safe here because tokenBuf is pre-allocated and lives for the engine's lifetime.
		return unsafe.String(&e.tokenBuf[0], n)
	}
	return ""
}

// isEOG checks if a token is an end-of-generation token.
// Phase 4A: uses Go-native lookup (zero FFI) when available.
func (e *TorchEngine) isEOG(token llama.Token) bool {
	if e.hasGoTokenizer {
		return e.goTokenizer.IsEOG(int32(token))
	}
	return llama.VocabIsEOG(e.vocab, token)
}

// NewTorchEngine creates an engine that loads a GGUF model via llama.cpp.
// libPath is the directory containing the llama.cpp shared libraries.
// If empty, checks the ITAK_TORCH_LIB environment variable.
func NewTorchEngine(modelPath string, opts EngineOpts) (*TorchEngine, error) {
	// --- Security: validate model path before doing anything ---
	// Blocks directory traversal (../), wrong extensions, and symlink escapes.
	if err := ValidateModelPath(modelPath); err != nil {
		return nil, fmt.Errorf("model path rejected: %w", err)
	}

	// Also validate draft model path if speculative decoding is configured.
	if opts.DraftModelPath != "" {
		if err := ValidateModelPath(opts.DraftModelPath); err != nil {
			return nil, fmt.Errorf("draft model path rejected: %w", err)
		}
	}

	// --- Auto-Configuration: detect hardware and select optimal backend ---
	// Probes GPUs, estimates VRAM needs from model size, picks backend/layers/threads.
	// User overrides (env var, explicit Backend, explicit GPULayers) always win.

	// If ITAK_TORCH_LIB is set, infer the backend from the path name.
	// This prevents the situation where setting ITAK_TORCH_LIB=.../cuda/
	// still auto-selects Vulkan - the lib path IS the user's backend choice.
	envLib := os.Getenv("ITAK_TORCH_LIB")
	if envLib != "" && opts.Backend == "" {
		pathLower := strings.ToLower(envLib)
		switch {
		case strings.Contains(pathLower, "cuda"):
			opts.Backend = "cuda"
			fmt.Println("[iTaK Torch] Backend inferred from ITAK_TORCH_LIB: cuda")
		case strings.Contains(pathLower, "vulkan"):
			opts.Backend = "vulkan"
			fmt.Println("[iTaK Torch] Backend inferred from ITAK_TORCH_LIB: vulkan")
		case strings.Contains(pathLower, "metal"):
			opts.Backend = "metal"
			fmt.Println("[iTaK Torch] Backend inferred from ITAK_TORCH_LIB: metal")
		}
	}

	ac := ApplyAutoConfig(&opts, modelPath)
	fmt.Printf("[iTaK Torch] AutoConfig: %s\n", ac)

	// --- Startup Diagnostics ---
	diag := NewDiagReport(DiagLevelFromEnv())

	// Find the llama.cpp shared libraries.
	libPath := envLib
	if libPath == "" {
		// Use AutoConfig recommendation with graceful fallback.
		libPath = RecommendLibPath(ac)

		// If forced backend was specified but lib dir doesn't exist, try candidates.
		if libPath == "" {
			platformDir := runtime.GOOS + "_" + runtime.GOARCH
			for _, suffix := range []string{"_vulkan", "_cuda", "_metal", ""} {
				candidate := "./lib/" + platformDir + suffix
				if _, err := os.Stat(candidate); err == nil {
					libPath = candidate
					break
				}
			}
		}
	}
	fmt.Printf("[iTaK Torch] Using libs from: %s\n", libPath)

	// Run diagnostics (DLL scan, backend check, GPU verification, perf settings).
	diag.RunDLLDiagnostics(libPath)
	diag.RunBackendDiagnostics(ac, &opts)
	diag.RunGPUVerification()
	diag.RunPerfDiagnostics(&opts)
	diag.Print()

	if libPath == "" {
		return nil, fmt.Errorf("llama.cpp libraries not found. Set ITAK_TORCH_LIB env variable or run 'itaktorch install'")
	}

	// Load the shared libraries.
	if err := llama.Load(libPath); err != nil {
		return nil, fmt.Errorf("load llama.cpp libraries from %s: %w", libPath, err)
	}

	// Fix: Some environments (Docker, IDE shells) set CUDA_VISIBLE_DEVICES=-1
	// which hides all GPUs from the CUDA runtime. If GPU layers are requested
	// (positive count or -1 for auto-all) and the variable blocks GPU access, override it.
	if opts.GPULayers != 0 {
		if cvd := os.Getenv("CUDA_VISIBLE_DEVICES"); cvd == "-1" || cvd == "" {
			os.Setenv("CUDA_VISIBLE_DEVICES", "0")
		}
	}

	// Initialize the backend.
	llama.BackendLoadAll() // Fix: Load backends before Init
	llama.Init()

	// Initialize NUMA if configured.
	if opts.NumaStrategy > 0 {
		llama.NumaInit(llama.NumaStrategy(opts.NumaStrategy))
		fmt.Printf("[iTaK Torch] NUMA strategy: %d\n", opts.NumaStrategy)
	}

	// Apply smart defaults (AutoConfig already set Threads and GPULayers if needed).
	if opts.ContextSize == 0 {
		// Context auto-sizing: read n_ctx_train from model metadata after loading.
		// Will be refined below once we have the model handle.
		opts.ContextSize = 2048
	}

	topo := DetectCPUTopology()

	// Safety net: if Threads still unset after AutoConfig, fall back to physical cores.
	if opts.Threads == 0 {
		opts.Threads = topo.PhysicalCores
	}
	fmt.Printf("[iTaK Torch] Threads: %d (physical=%d, logical=%d, hybrid=%v)\n",
		opts.Threads, topo.PhysicalCores, topo.LogicalCores, topo.IsHybrid)

	if opts.BatchSize == 0 {
		// Optimize Batch Size for Mini PCs (Low core count)
		// Processing 2048 tokens on 4 cores can cause UI freezes/latency spikes.
		if topo.PhysicalCores <= 4 {
			opts.BatchSize = 512
			fmt.Println("[iTaK Torch] Mini PC detected: reduced batch_size to 512 for responsiveness")
		} else {
			opts.BatchSize = 2048
		}
	}

	// GPU layers: -1 means offload everything.
	if opts.GPULayers == -1 {
		opts.GPULayers = 999 // llama.cpp clamps to actual layer count
	}

	// Auto-enable flash attention on GPU mode.
	if opts.GPULayers > 0 && !opts.NoFlashAttention {
		opts.FlashAttention = true
	}

	// Snapshot resources before loading.
	preLoad := CaptureResources()
	loadStart := time.Now()

	// Load the model.
	// ModelDefaultParams() returns mmap=1 from llama.cpp, which we preserve
	// unless the user explicitly disables it (required for WSL2 9P mounts).
	modelParams := llama.ModelDefaultParams()
	modelParams.NGpuLayers = int32(opts.GPULayers)
	if opts.UseMlock {
		modelParams.UseMlock = 1
	}
	if opts.NoMmap {
		modelParams.UseMmap = 0
		fmt.Println("[iTaK Torch] mmap disabled (--no-mmap): using read() for model loading")
	}

	// Verify model integrity via SHA256 sidecar before loading.
	if err := VerifyModelIntegrity(modelPath); err != nil {
		return nil, fmt.Errorf("integrity check: %w", err)
	}

	model, err := llama.ModelLoadFromFile(modelPath, modelParams)
	if err != nil {
		return nil, fmt.Errorf("load model %s: %w", modelPath, err)
	}

	loadDuration := time.Since(loadStart)

	// Context auto-sizing: if user didn't set a specific context size,
	// use the model's training context (n_ctx_train) capped by VRAM.
	if opts.ContextSize == 2048 {
		trainCtx := int(llama.ModelNCtxTrain(model))
		if trainCtx > 2048 {
			// Cap at 8192 for reasonable VRAM usage on consumer GPUs.
			// Models trained on 128k context don't need that much in practice.
			maxCtx := 8192
			if trainCtx < maxCtx {
				maxCtx = trainCtx
			}
			opts.ContextSize = maxCtx
			fmt.Printf("[iTaK Torch] Context auto-sized: %d (model trained on %d)\n", maxCtx, trainCtx)
		}
	}

	// Pre-fault mmap'd model pages into RAM to avoid page fault stalls
	// during first inference. Skipped when mmap is disabled (--no-mmap).
	if !opts.NoMmap {
		if err := PrefaultModelFile(modelPath); err != nil {
			fmt.Printf("[iTaK Torch] Prefault warning: %v\n", err)
		}
	}

	// Create context with performance optimizations.
	ctxParams := llama.ContextDefaultParams()
	ctxParams.NCtx = uint32(opts.ContextSize)
	ctxParams.NThreads = int32(opts.Threads)
	ctxParams.NThreadsBatch = int32(opts.Threads)
	ctxParams.NBatch = uint32(opts.BatchSize)

	// Continuous batching: allow multiple concurrent sequences.
	if opts.MaxSlots > 1 {
		ctxParams.NSeqMax = uint32(opts.MaxSlots)
	}

	// KV cache quantization: default to q8_0 for 50% VRAM savings.
	// q8_0 has negligible quality loss vs f16 but halves KV cache memory,
	// allowing larger context windows or bigger models in the same VRAM.
	kvType := strings.ToLower(opts.KVCacheType)
	if kvType == "" {
		kvType = "q8_0" // Default: q8_0 (Ollama defaults to f16, we default to q8_0)
	}
	switch kvType {
	case "q8_0", "q8":
		ctxParams.TypeK = llama.GGMLTypeQ8_0
		ctxParams.TypeV = llama.GGMLTypeQ8_0
		fmt.Println("[iTaK Torch] KV cache: q8_0 (50% VRAM reduction vs f16)")
	case "q4_0", "q4":
		ctxParams.TypeK = llama.GGMLTypeQ4_0
		ctxParams.TypeV = llama.GGMLTypeQ4_0
		fmt.Println("[iTaK Torch] KV cache: q4_0 (75% VRAM reduction vs f16)")
	case "f16":
		fmt.Println("[iTaK Torch] KV cache: f16 (full precision)")
	default:
		fmt.Printf("[iTaK Torch] Warning: unknown kv-cache-type %q, using q8_0\n", opts.KVCacheType)
		ctxParams.TypeK = llama.GGMLTypeQ8_0
		ctxParams.TypeV = llama.GGMLTypeQ8_0
	}

	// Phase 5: KV cache defragmentation threshold.
	if opts.DefragThreshold > 0 {
		ctxParams.DefragThold = opts.DefragThreshold
	}

	// Enable flash attention for faster inference (auto mode detects support).
	if opts.FlashAttention {
		ctxParams.FlashAttentionType = llama.FlashAttentionTypeEnabled
	}

	ctx, err := llama.InitFromModel(model, ctxParams)
	if err != nil {
		return nil, fmt.Errorf("create context: %w", err)
	}

	// Get vocab.
	vocab := llama.ModelGetVocab(model)

	// Create default sampler chain with standard sampling pipeline.
	samplerParams := llama.DefaultSamplerParams()
	sampler := llama.NewSampler(model, llama.DefaultSamplers, samplerParams)

	// Extract model name from path.
	name := modelPath
	if idx := strings.LastIndex(name, "/"); idx >= 0 {
		name = name[idx+1:]
	}
	if idx := strings.LastIndex(name, "\\"); idx >= 0 {
		name = name[idx+1:]
	}
	name = strings.TrimSuffix(name, ".gguf")

	engine := &TorchEngine{
		model:       model,
		ctx:         ctx,
		vocab:       vocab,
		sampler:     sampler,
		tokenBuf:    make([]byte, 256), // pre-allocated to avoid per-token heap allocation
		modelName:   name,
		modelPath:   modelPath,
		opts:        opts,
		loaded:      true,
		prefixCache: NewPrefixCache(opts.PrefixCacheSize),
	}

	// Auto-detect chat template using multi-level resolution:
	//   1. GGUF metadata (llama_model_chat_template)
	//   2. Local tokenizer_config.json (HuggingFace models on disk)
	//   3. Model name heuristic (filename pattern matching)
	rawTemplate := llama.ModelChatTemplate(model, "")
	engine.chatTemplate = ResolveTemplate(rawTemplate, modelPath, name)
	fmt.Printf("[iTaK Torch] Chat template: %s\n", engine.chatTemplate.Name)

	// Record load metrics.
	postLoad := CaptureResources()
	engine.Stats.ModelLoadTime = loadDuration
	engine.Stats.PreLoadRes = preLoad
	engine.Stats.PostLoadRes = postLoad

	fmt.Printf("[iTaK Torch] Model loaded in %s\n", loadDuration.Round(time.Millisecond))
	fmt.Printf("[iTaK Torch] %s\n", preLoad.String())
	fmt.Printf("[iTaK Torch] %s\n", postLoad.String())

	// Print system capabilities.
	sysInfo := llama.PrintSystemInfo()
	if sysInfo != "" {
		fmt.Printf("[iTaK Torch] System: %s\n", sysInfo)
	}
	fmt.Printf("[iTaK Torch] Optimizations: mmap=%v mlock=%v gpu_offload=%v flash_attn=%v threads=%d batch=%d\n",
		llama.SupportsMmap(), opts.UseMlock, llama.SupportsGpuOffload(),
		opts.FlashAttention, opts.Threads, opts.BatchSize)

	// Warmup: triggers GPU kernel JIT compilation for common tensor shapes,
	// reducing latency on first real request.
	if err := llama.Warmup(ctx, model); err != nil {
		fmt.Printf("[iTaK Torch] Warmup warning: %v\n", err)
	} else {
		fmt.Printf("[iTaK Torch] Model warmup complete\n")
	}

	// --- Phase 4A: Load Go-native tokenizer from GGUF metadata ---
	goTok, tokErr := tokenizer.NewFromGGUF(modelPath)
	if tokErr != nil {
		fmt.Printf("[iTaK Torch] Go tokenizer unavailable (using FFI fallback): %v\n", tokErr)
	} else {
		engine.goTokenizer = goTok
		engine.hasGoTokenizer = true
		fmt.Printf("[iTaK Torch] Go-native tokenizer loaded: %d tokens, %d merges\n",
			goTok.VocabSize, len(goTok.MergeRank))
	}

	// --- Speculative Decoding: Load draft model if configured ---
	if opts.DraftModelPath != "" {
		fmt.Printf("[iTaK Torch] Loading draft model: %s\n", opts.DraftModelPath)

		specTokens := opts.SpeculativeTokens
		if specTokens == 0 {
			specTokens = 5
		}

		draftGPU := opts.DraftGPULayers
		if draftGPU == 0 {
			draftGPU = opts.GPULayers
		}

		draftParams := llama.ModelDefaultParams()
		draftParams.NGpuLayers = int32(draftGPU)

		draftModel, draftErr := llama.ModelLoadFromFile(opts.DraftModelPath, draftParams)
		if draftErr != nil {
			fmt.Printf("[iTaK Torch] Draft model load failed (continuing without speculative decoding): %v\n", draftErr)
		} else {
			draftCtxParams := llama.ContextDefaultParams()
			draftCtxParams.NCtx = uint32(opts.ContextSize)
			draftCtxParams.NThreads = int32(opts.Threads)
			draftCtxParams.NThreadsBatch = int32(opts.Threads)
			draftCtxParams.NBatch = uint32(opts.BatchSize)
			if opts.FlashAttention {
				draftCtxParams.FlashAttentionType = llama.FlashAttentionTypeEnabled
			}

			draftCtx, draftCtxErr := llama.InitFromModel(draftModel, draftCtxParams)
			if draftCtxErr != nil {
				fmt.Printf("[iTaK Torch] Draft context init failed: %v\n", draftCtxErr)
				llama.ModelFree(draftModel)
			} else {
				draftVocab := llama.ModelGetVocab(draftModel)
				draftSamplerParams := llama.DefaultSamplerParams()
				draftSampler := llama.NewSampler(draftModel, llama.DefaultSamplers, draftSamplerParams)

				engine.draftModel = draftModel
				engine.draftCtx = draftCtx
				engine.draftVocab = draftVocab
				engine.draftSampler = draftSampler
				engine.hasDraft = true

				// Warmup draft model.
				if err := llama.Warmup(draftCtx, draftModel); err != nil {
					fmt.Printf("[iTaK Torch] Draft warmup warning: %v\n", err)
				}

				fmt.Printf("[iTaK Torch] Speculative decoding enabled: draft=%s, speculative_tokens=%d\n",
					opts.DraftModelPath, specTokens)
			}
		}
	}

	return engine, nil
}

// Complete runs inference on the given messages and returns the generated text.
func (e *TorchEngine) Complete(ctx context.Context, messages []ChatMessage, params CompletionParams) (string, error) {
	e.mu.Lock()
	defer e.mu.Unlock()

	if !e.loaded {
		return "", fmt.Errorf("engine not loaded")
	}

	// Build prompt from messages using auto-detected chat template.
	var prompt string
	var systemPrefix string
	if e.chatTemplate != nil && e.chatTemplate.Type != TemplatePlain {
		prompt = e.chatTemplate.Apply(messages)
		systemPrefix = e.chatTemplate.RenderPrefix(messages)
	} else {
		prompt = BuildPrompt(messages)
		systemPrefix = BuildPrefixPrompt(messages)
	}

	// Tokenize the prompt.
	// Phase 4A: prefer Go-native tokenizer (no FFI overhead).
	var tokens []llama.Token
	if e.hasGoTokenizer {
		goTokens := e.goTokenizer.Encode(prompt, true)
		tokens = make([]llama.Token, len(goTokens))
		for i, t := range goTokens {
			tokens[i] = llama.Token(t)
		}
	} else {
		tokens = llama.Tokenize(e.vocab, prompt, true, false)
	}

	// Tokenize system prefix to find the token boundary for prefix caching.
	// Only do this if prefix caching is enabled and there's a non-trivial prefix.
	var prefixTokenCount int
	if systemPrefix != "" && e.opts.PrefixCacheSize > 0 {
		if e.hasGoTokenizer {
			prefixTokenCount = len(e.goTokenizer.Encode(systemPrefix, true))
		} else {
			prefixTokenCount = len(llama.Tokenize(e.vocab, systemPrefix, true, false))
		}
	}

	return e.generateFromTokensLocked(ctx, prompt, tokens, params, systemPrefix, prefixTokenCount)
}

// GenerateTokens bypasses the tokenizer and runs inference directly on the given token IDs.
// This supports the Nexa-inspired Exact Token ID Injection feature.
func (e *TorchEngine) GenerateTokens(ctx context.Context, inputTokens []int32, params CompletionParams) (string, error) {
	e.mu.Lock()
	defer e.mu.Unlock()

	if !e.loaded {
		return "", fmt.Errorf("engine not loaded")
	}

	var tokens []llama.Token
	for _, t := range inputTokens {
		tokens = append(tokens, llama.Token(t))
	}

	// No prefix caching for raw token injection (no message structure to extract prefix from).
	return e.generateFromTokensLocked(ctx, "", tokens, params, "", 0)
}

// generateFromTokensLocked implements the core generation loop. The caller must hold e.mu.Lock().
func (e *TorchEngine) generateFromTokensLocked(ctx context.Context, prompt string, tokens []llama.Token, params CompletionParams, systemPrefix string, prefixTokenCount int) (string, error) {
	// Reset sampler state for this request.
	llama.SamplerReset(e.sampler)

	maxTokens := params.MaxTokens
	if maxTokens == 0 {
		maxTokens = 512
	}

	const minPrefixTokens = 32
	promptStart := time.Now()
	prefixCacheHit := false
	var batch llama.Batch

	batchSize := int(e.opts.BatchSize)
	if batchSize <= 0 {
		batchSize = 2048
	}

	usePrefixCache := e.opts.PrefixCacheSize > 0 &&
		systemPrefix != "" &&
		prefixTokenCount >= minPrefixTokens &&
		prefixTokenCount < len(tokens)

	if usePrefixCache {
		if entry, ok := e.prefixCache.Lookup(systemPrefix); ok {
			if _, err := e.prefixCache.Restore(e.ctx, entry); err == nil {
				prefixCacheHit = true
				fmt.Printf("[iTaK Torch] Prefix cache HIT: %d tokens cached\n", prefixTokenCount)
			}
		}
	}

	if prefixCacheHit {
		remaining := tokens[prefixTokenCount:]
		for i := 0; i < len(remaining); i += batchSize {
			end := i + batchSize
			if end > len(remaining) {
				end = len(remaining)
			}
			chunk := remaining[i:end]
			batch = llama.BatchGetOne(chunk)
			if _, err := llama.Decode(e.ctx, batch); err != nil {
				return "", fmt.Errorf("decode remaining chunk: %w", err)
			}
		}
	} else if usePrefixCache {
		prefixTokens := tokens[:prefixTokenCount]
		for i := 0; i < len(prefixTokens); i += batchSize {
			end := i + batchSize
			if end > len(prefixTokens) {
				end = len(prefixTokens)
			}
			chunk := prefixTokens[i:end]
			batch = llama.BatchGetOne(chunk)
			if _, err := llama.Decode(e.ctx, batch); err != nil {
				return "", fmt.Errorf("decode prefix chunk: %w", err)
			}
		}
		e.prefixCache.Save(e.ctx, systemPrefix, prefixTokens)
		remaining := tokens[prefixTokenCount:]
		for i := 0; i < len(remaining); i += batchSize {
			end := i + batchSize
			if end > len(remaining) {
				end = len(remaining)
			}
			chunk := remaining[i:end]
			batch = llama.BatchGetOne(chunk)
			if _, err := llama.Decode(e.ctx, batch); err != nil {
				return "", fmt.Errorf("decode remaining chunk: %w", err)
			}
		}
	} else {
		for i := 0; i < len(tokens); i += batchSize {
			end := i + batchSize
			if end > len(tokens) {
				end = len(tokens)
			}
			chunk := tokens[i:end]
			batch = llama.BatchGetOne(chunk)
			if _, err := llama.Decode(e.ctx, batch); err != nil {
				return "", fmt.Errorf("decode prompt chunk: %w", err)
			}
		}
	}
	promptDuration := time.Since(promptStart)

	genStart := time.Now()
	var result strings.Builder
	completionTokens := 0
	hasStopSequences := len(params.Stop) > 0
	singleToken := make([]llama.Token, 1)

	for i := 0; i < maxTokens; i++ {
		select {
		case <-ctx.Done():
			return result.String(), ctx.Err()
		default:
		}

		if i > 0 {
			llama.Synchronize(e.ctx)
		}

		token := llama.SamplerSample(e.sampler, e.ctx, -1)
		if e.isEOG(token) {
			break
		}

		piece := e.tokenToText(token)
		if len(piece) > 0 {
			result.WriteString(piece)
			completionTokens++
			if e.streamCh != nil {
				select {
				case e.streamCh <- piece:
				case <-ctx.Done():
					return result.String(), ctx.Err()
				}
			}
		}

		if hasStopSequences {
			fullText := result.String()
			for _, stop := range params.Stop {
				if strings.HasSuffix(fullText, stop) {
					result.Reset()
					result.WriteString(fullText[:len(fullText)-len(stop)])
					return result.String(), nil
				}
			}
		}

		singleToken[0] = token
		batch = llama.BatchGetOne(singleToken)
		if _, err := llama.Decode(e.ctx, batch); err != nil {
			return result.String(), fmt.Errorf("decode token: %w", err)
		}
	}

	genDuration := time.Since(genStart)
	tokPerSec := 0.0
	if genDuration.Seconds() > 0 {
		tokPerSec = float64(completionTokens) / genDuration.Seconds()
	}

	metrics := &InferenceMetrics{
		PromptTokens:     len(tokens),
		CompletionTokens: completionTokens,
		TotalTokens:      len(tokens) + completionTokens,
		PromptDuration:   promptDuration,
		GenDuration:      genDuration,
		TotalDuration:    promptDuration + genDuration,
		TokensPerSecond:  tokPerSec,
	}
	e.Stats.RecordRequest(metrics)
	fmt.Printf("%s\n", metrics.String())

	return result.String(), nil
}

// CompleteStream runs inference and sends tokens to a channel.
func (e *TorchEngine) CompleteStream(ctx context.Context, messages []ChatMessage, params CompletionParams, ch chan string) (string, error) {
	e.streamCh = ch
	defer func() {
		e.streamCh = nil
		close(ch)
	}()
	return e.Complete(ctx, messages, params)
}

func (e *TorchEngine) GetStats() EngineStats {
	return e.Stats.Snapshot()
}

func (e *TorchEngine) ModelName() string {
	return e.modelName
}

func (e *TorchEngine) IsLoaded() bool {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.loaded
}

func (e *TorchEngine) Reload() error {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.loaded {
		return nil
	}
	tmp, err := NewTorchEngine(e.modelPath, e.opts)
	if err != nil {
		return err
	}
	e.model = tmp.model
	e.ctx = tmp.ctx
	e.vocab = tmp.vocab
	e.sampler = tmp.sampler
	e.loaded = true
	return nil
}

func (e *TorchEngine) Close() error {
	e.mu.Lock()
	defer e.mu.Unlock()
	if !e.loaded {
		return nil
	}
	if e.hasDraft {
		llama.ModelFree(e.draftModel)
		e.hasDraft = false
	}
	llama.Close()
	e.loaded = false
	return nil
}



// Synthesize converts text to speech audio bytes using Go-native iTaK Torch kernels.
func (e *TorchEngine) Synthesize(ctx context.Context, req SpeechRequest) ([]byte, error) {
	return nil, fmt.Errorf("Go-native GGUF-TTS synthesis is in development. Use 'itak' engine architecture for live synthesis.")
}

func detectIntegratedGPU() (bool, error) {
	if runtime.GOOS == "windows" {
		out, err := execCommand("wmic", "path", "win32_videocontroller", "get", "name")
		if err == nil {
			s := strings.ToLower(string(out))
			return strings.Contains(s, "intel") || strings.Contains(s, "amd") || strings.Contains(s, "radeon"), nil
		}
	}
	return false, nil
}

func hasDiscreteNVIDIA() bool {
	if runtime.GOOS == "windows" {
		out, err := execCommand("nvidia-smi", "-L")
		return err == nil && len(out) > 0
	}
	return false
}

func execCommand(name string, arg ...string) ([]byte, error) {
	cmd := exec.Command(name, arg...)
	return cmd.Output()
}
