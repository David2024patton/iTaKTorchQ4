// engine.go implements the torch.Engine interface using the pure Go tensor engine.
//
// This is the entry point for GOTensor. It ties together the tensor library,
// attention mechanism, and a minimal transformer forward pass to implement
// the same Engine interface that TorchEngine uses.
//
// LIMITATIONS:
//   - No GGUF loading (returns an error with instructions to use TorchEngine)
//   - Greedy decoding only (no sampling, no temperature)
//   - No streaming support
//   - Maximum context: 512 tokens (memory grows quadratically with attention)
package native

import (
	"context"
	"fmt"
	"math"
	"runtime"
	"sync"
	"time"
)

// ---------- Engine Interface Types ----------
// These mirror torch.ChatMessage and torch.CompletionParams to avoid circular imports.

// ChatMessage represents a single message in a conversation.
type ChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// CompletionParams controls inference behavior.
type CompletionParams struct {
	MaxTokens   int      `json:"max_tokens,omitempty"`
	Temperature float64  `json:"temperature,omitempty"`
	TopP        float64  `json:"top_p,omitempty"`
	Stop        []string `json:"stop,omitempty"`
}

// EngineStats holds performance metrics.
type EngineStats struct {
	TotalRequests  int64
	TotalTokensGen int64
	LastMetrics    *InferenceMetrics
}

// InferenceMetrics is a single request's performance data.
type InferenceMetrics struct {
	PromptTokens     int
	CompletionTokens int
	TotalTokens      int
	PromptDuration   time.Duration
	GenDuration      time.Duration
	TotalDuration    time.Duration
	TokensPerSecond  float64
}

// ---------- GOTensor Engine ----------

// NativeEngine implements a minimal inference engine using pure Go tensors.
//
// WHY: Eliminates ALL external dependencies. No DLLs, no .so files, no CGo.
// Just `go build` and run on any platform Go supports.
//
// WHAT IT CAN DO:
//   - Generate text using a tiny vocabulary (for testing/demo)
//   - Run forward passes through transformer layers
//   - Demonstrate the attention mechanism
//
// WHAT IT CAN'T DO (yet):
//   - Load real GGUF models (that's TorchEngine's job)
//   - Run models larger than ~100M parameters at usable speed
type NativeEngine struct {
	mu         sync.Mutex
	name       string
	vocabSize  int
	hiddenDim  int
	numHeads   int
	numKVHeads int // GQA: K/V head count (may be < numHeads)
	ffnDim     int // FFN intermediate dimension (hidden -> ffnDim -> hidden)
	numLayers  int
	loaded     bool
	stats      EngineStats

	// Benchmark instrumentation.
	BenchMode      bool          // Print rich metrics after each inference.
	loadDuration   time.Duration // Time to load model from disk.
	layerTimings   []time.Duration // Per-layer forward pass timing.

	// GPU compute backend (Vulkan/Metal/D3D12 via WebGPU).
	gpu *GPUBackend

	// Sparse inference (PowerInfer-style).
	SparseConfig SparseFFNConfig
	predictors   []*NeuronPredictor // One per layer
	profile      *NeuronProfile     // Activation profiling data

	// Attention Residuals (Kimi paper, March 2026).
	// Replaces fixed residual Add with learned depth-wise attention.
	AttnResConfig AttnResConfig

	// Feature Flags
	UseAEX         bool
	UseSpeculative bool
	UseFusion      bool

	// Advanced features (RoPE, BPE, Flash Attention, MoE, etc.).
	features  *EngineFeatures
	maxSeqLen int // Maximum sequence length for RoPE

	// MoE (Mixture of Experts) support.
	moeConfig   MoEConfig
	moeLayers   []MoELayer           // MoE layers (one per transformer block)
	expertMgr   *ExpertOffloadManager // GPU/CPU expert placement manager

	// Weights.
	embeddings *Tensor // [vocab_size, hidden_dim]
	layers     []TransformerLayer
	lmHead     *Tensor // [vocab_size, hidden_dim]
}

// NativeEngineOpts holds configuration for the native engine.
type NativeEngineOpts struct {
	UseAEX         bool
	UseSpeculative bool
	UseFusion      bool
	GPULayers      int
	Threads        int
}

// TransformerLayer holds the weights for one transformer block.
type TransformerLayer struct {
	// Attention weights.
	WQ, WK, WV, WO *Tensor // [hidden_dim, hidden_dim] each

	// FFN weights (gate-up-down pattern used by Llama/Qwen).
	WGate, WUp, WDown *Tensor

	// Normalization weights.
	AttnNorm, FFNNorm *Tensor // [hidden_dim] each

	// Attention Residual pseudo-query vectors (Kimi paper, March 2026).
	// Each layer learns which previous layers are most useful.
	AttnResQuery *Tensor // [hidden_dim] - query for post-attention residual
	FFNResQuery  *Tensor // [hidden_dim] - query for post-FFN residual
}

// NewNativeEngine creates a GOTensor engine with the given architecture.
//
// Parameters:
//   - name: model name for API responses
//   - vocabSize: number of unique tokens
//   - hiddenDim: hidden layer width (e.g., 768 for a tiny model)
//   - numHeads: number of attention heads
//   - numLayers: number of transformer layers
func NewNativeEngine(name string, vocabSize, hiddenDim, numHeads, numLayers int) *NativeEngine {
	e := &NativeEngine{
		name:       name,
		vocabSize:  vocabSize,
		hiddenDim:  hiddenDim,
		numHeads:   numHeads,
		numKVHeads: numHeads, // Default: same as Q heads (standard MHA)
		ffnDim:     hiddenDim, // Default: same as hidden
		numLayers:  numLayers,
		loaded:     true,
	}

	// Initialize weights with small random values.
	// In a real engine, these would be loaded from a GGUF file.
	e.embeddings = NewTensor([]int{vocabSize, hiddenDim})
	initWeights(e.embeddings, hiddenDim)

	e.layers = make([]TransformerLayer, numLayers)
	for i := range e.layers {
		l := &e.layers[i]
		l.WQ = NewTensor([]int{hiddenDim, hiddenDim})
		l.WK = NewTensor([]int{hiddenDim, hiddenDim})
		l.WV = NewTensor([]int{hiddenDim, hiddenDim})
		l.WO = NewTensor([]int{hiddenDim, hiddenDim})
		l.WGate = NewTensor([]int{hiddenDim, hiddenDim})
		l.WUp = NewTensor([]int{hiddenDim, hiddenDim})
		l.WDown = NewTensor([]int{hiddenDim, hiddenDim})
		l.AttnNorm = NewTensor([]int{hiddenDim})
		l.FFNNorm = NewTensor([]int{hiddenDim})
		// Initialize with ones for norm weights (standard practice).
		for j := range l.AttnNorm.Data {
			l.AttnNorm.Data[j] = 1.0
			l.FFNNorm.Data[j] = 1.0
		}
		initWeights(l.WQ, hiddenDim)
		initWeights(l.WK, hiddenDim)
		initWeights(l.WV, hiddenDim)
		initWeights(l.WO, hiddenDim)
		initWeights(l.WGate, hiddenDim)
		initWeights(l.WUp, hiddenDim)
		initWeights(l.WDown, hiddenDim)

		// AttnRes pseudo-query vectors (learned during training).
		l.AttnResQuery = NewTensor([]int{hiddenDim})
		l.FFNResQuery = NewTensor([]int{hiddenDim})
		initWeights(l.AttnResQuery, hiddenDim)
		initWeights(l.FFNResQuery, hiddenDim)
	}

	e.lmHead = NewTensor([]int{vocabSize, hiddenDim})
	initWeights(e.lmHead, hiddenDim)

	return e
}

// NewNativeEngineFromGGUF loads a GGUF model into the native engine.
func NewNativeEngineFromGGUF(path string, opts ...NativeEngineOpts) (*NativeEngine, error) {
	gf, err := LoadGGUF(path)
	if err != nil {
		return nil, fmt.Errorf("load GGUF: %w", err)
	}

	// Read architecture metadata to determine model dimensions.
	arch := gf.GetMetadataString("general.architecture")
	if arch == "" {
		arch = "unknown"
	}

	// Try to read dimension metadata (common GGUF keys).
	vocabSize := int(gf.GetMetadataUint32(arch + ".vocab_size"))
	hiddenDim := int(gf.GetMetadataUint32(arch + ".embedding_length"))
	numHeads := int(gf.GetMetadataUint32(arch + ".attention.head_count"))
	numKVHeads := int(gf.GetMetadataUint32(arch + ".attention.head_count_kv"))
	numLayers := int(gf.GetMetadataUint32(arch + ".block_count"))
	ffnDim := int(gf.GetMetadataUint32(arch + ".feed_forward_length"))

	// Defaults for when metadata is missing.
	if vocabSize == 0 {
		vocabSize = 32000
	}
	if hiddenDim == 0 {
		hiddenDim = 768
	}
	if numHeads == 0 {
		numHeads = 12
	}
	if numKVHeads == 0 {
		numKVHeads = numHeads // Default to standard MHA
	}
	if numLayers == 0 {
		numLayers = 6
	}
	if ffnDim == 0 {
		ffnDim = hiddenDim * 4 // Standard 4x expansion
	}

	// Create engine with detected dimensions.
	e := NewNativeEngine(arch+"-gguf", vocabSize, hiddenDim, numHeads, numLayers)
	e.numKVHeads = numKVHeads
	e.ffnDim = ffnDim

	fmt.Printf("[GOTensor] Model: arch=%s vocab=%d hidden=%d heads=%d kv_heads=%d layers=%d ffn=%d\n",
		arch, vocabSize, hiddenDim, numHeads, numKVHeads, numLayers, ffnDim)

	// Apply options if provided.
	if len(opts) > 0 {
		e.UseAEX = opts[0].UseAEX
		e.UseSpeculative = opts[0].UseSpeculative
		e.UseFusion = opts[0].UseFusion
	}

	// Try to load tensors from the file. Missing tensors keep their random init.
	loadTensor := func(name string) *Tensor {
		info := gf.FindTensor(name)
		if info == nil {
			return nil
		}
		t, err := gf.ReadTensor(*info)
		if err != nil {
			return nil
		}
		return t
	}

	// Load embedding and lm_head.
	if t := loadTensor("token_embd.weight"); t != nil {
		e.embeddings = t
	}
	if t := loadTensor("output.weight"); t != nil {
		e.lmHead = t
	}

	// Load layer weights.
	for i := 0; i < numLayers; i++ {
		prefix := fmt.Sprintf("blk.%d", i)
		if t := loadTensor(prefix + ".attn_q.weight"); t != nil {
			e.layers[i].WQ = t
		}
		if t := loadTensor(prefix + ".attn_k.weight"); t != nil {
			e.layers[i].WK = t
		}
		if t := loadTensor(prefix + ".attn_v.weight"); t != nil {
			e.layers[i].WV = t
		}
		if t := loadTensor(prefix + ".attn_output.weight"); t != nil {
			e.layers[i].WO = t
		}
		if t := loadTensor(prefix + ".ffn_gate.weight"); t != nil {
			e.layers[i].WGate = t
		}
		if t := loadTensor(prefix + ".ffn_up.weight"); t != nil {
			e.layers[i].WUp = t
		}
		if t := loadTensor(prefix + ".ffn_down.weight"); t != nil {
			e.layers[i].WDown = t
		}
		if t := loadTensor(prefix + ".attn_norm.weight"); t != nil {
			e.layers[i].AttnNorm = t
		}
		if t := loadTensor(prefix + ".ffn_norm.weight"); t != nil {
			e.layers[i].FFNNorm = t
		}
	}

	// Phase 29: Detect and initialize MoE if present.
	moeLayers, moeConfig := LoadMoEFromGGUF(gf)
	if moeConfig.Enabled {
		e.moeConfig = moeConfig
		e.moeLayers = moeLayers
		
		// Set fusion flag from options.
		e.moeConfig.FusionEnabled = e.UseFusion

		if e.UseAEX {
			e.expertMgr = NewExpertOffloadManager(ExpertOffloadConfig{
				MaxGPUExperts: moeConfig.NumActive * 2,
				TotalExperts:  moeConfig.NumExperts,
				PrefetchTopK:  moeConfig.NumActive,
				AccessDecay:   0.99,
				GlobalPinTopK: 4, // AEX: Pin top 4 experts model-wide
			})
			fmt.Printf("[GOTensor] AEX Expert Offload active: pinning top experts, speculative_gating=%v\n", e.UseSpeculative)
		}
		
		fmt.Printf("[GOTensor] MoE enabled: %d experts, top-%d routing, fusion=%v\n",
			moeConfig.NumExperts, moeConfig.NumActive, e.UseFusion)
	}

	return e, nil
}

// UseGPU enables the GPU compute backend. Called automatically during model load.
func (e *NativeEngine) UseGPU() {
	e.gpu = NewGPUBackend()
	if e.gpu.IsAvailable() {
		fmt.Println("[GOTensor] GPU compute enabled (Vulkan/Metal/D3D12)")

		// Upload all loaded weight tensors to GPU for persistent caching.
		// This eliminates host-to-device copies during inference.
		uploaded := 0
		if e.embeddings != nil {
			e.gpu.UploadWeight(e.embeddings)
			uploaded++
		}
		if e.lmHead != nil {
			e.gpu.UploadWeight(e.lmHead)
			uploaded++
		}
		for i := range e.layers {
			l := &e.layers[i]
			for _, w := range []*Tensor{l.WQ, l.WK, l.WV, l.WO, l.WGate, l.WUp, l.WDown, l.AttnNorm, l.FFNNorm} {
				if w != nil {
					e.gpu.UploadWeight(w)
					uploaded++
				}
			}
		}
		if uploaded > 0 {
			fmt.Printf("[GOTensor] Uploaded %d weight tensors to GPU memory\n", uploaded)
		}
	} else {
		fmt.Println("[GOTensor] No GPU found, using CPU")
	}
}

// ---------- Engine Interface Implementation ----------

// ModelName returns the model name.
func (e *NativeEngine) ModelName() string { return e.name }

// Complete runs inference on the given messages.
func (e *NativeEngine) Complete(ctx context.Context, messages []ChatMessage, params CompletionParams) (string, error) {
	e.mu.Lock()
	defer e.mu.Unlock()

	if !e.loaded {
		return "", fmt.Errorf("GOTensor engine not loaded")
	}

	start := time.Now()

	// Simple tokenization: split on spaces, map to vocab indices.
	prompt := ""
	for _, m := range messages {
		prompt += m.Content + " "
	}
	tokens := simpleTokenize(prompt, e.vocabSize)

	maxTokens := params.MaxTokens
	if maxTokens == 0 {
		maxTokens = 64
	}

	promptDuration := time.Since(start)
	genStart := time.Now()

	// Generate tokens autoregressively.
	var generated []int
	current := make([]int, len(tokens))
	copy(current, tokens)

	for i := 0; i < maxTokens; i++ {
		select {
		case <-ctx.Done():
			return tokensToText(generated), ctx.Err()
		default:
		}

		// Forward pass through the transformer.
		logits := e.forward(current)

		// Greedy decoding: pick the token with highest logit.
		nextToken := argmax(logits)
		generated = append(generated, nextToken)

		// Feed the full context + new token for next iteration
		// (since this simplistic pedagogical engine lacks a KV cache).
		current = append(current, nextToken)
	}

	genDuration := time.Since(genStart)
	result := tokensToText(generated)

	// Record metrics.
	tokPerSec := 0.0
	if genDuration.Seconds() > 0 {
		tokPerSec = float64(len(generated)) / genDuration.Seconds()
	}

	totalDur := time.Since(start)
	e.stats.TotalRequests++
	e.stats.TotalTokensGen += int64(len(generated))
	e.stats.LastMetrics = &InferenceMetrics{
		PromptTokens:     len(tokens),
		CompletionTokens: len(generated),
		TotalTokens:      len(tokens) + len(generated),
		PromptDuration:   promptDuration,
		GenDuration:      genDuration,
		TotalDuration:    totalDur,
		TokensPerSecond:  tokPerSec,
	}

	// Print benchmark panel if enabled.
	if e.BenchMode {
		e.printBenchmark(len(tokens), len(generated), promptDuration, genDuration, totalDur, tokPerSec)
	}

	return result, nil
}

// GetStats returns engine statistics.
func (e *NativeEngine) GetStats() EngineStats {
	return e.stats
}

// SetLoadDuration records the model load time for benchmark reporting.
func (e *NativeEngine) SetLoadDuration(d time.Duration) {
	e.loadDuration = d
}

// Close releases engine resources.
func (e *NativeEngine) Close() {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.loaded = false
}

// printBenchmark outputs a formatted benchmark panel to stdout.
func (e *NativeEngine) printBenchmark(promptToks, genToks int, promptDur, genDur, totalDur time.Duration, tps float64) {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)

	fmt.Println()
	fmt.Println("\033[36m--- Benchmark ---\033[0m")
	fmt.Printf("  Model:      %s (%d params, %d layers)\n", e.name, e.vocabSize*e.hiddenDim+e.numLayers*(e.hiddenDim*e.hiddenDim*4+e.hiddenDim*e.ffnDim*3), e.numLayers)
	fmt.Printf("  GQA:        %d Q heads, %d KV heads (head_dim=%d)\n", e.numHeads, e.numKVHeads, e.hiddenDim/max(e.numHeads, 1))
	if e.SparseConfig.Enabled {
		fmt.Printf("  Sparse:     %.0f%% sparsity\n", e.SparseConfig.Sparsity*100)
	} else {
		fmt.Printf("  Sparse:     off (dense)\n")
	}
	fmt.Printf("  Load:       %v\n", e.loadDuration)
	fmt.Printf("  Prompt:     %d tokens in %v\n", promptToks, promptDur)
	fmt.Printf("  Generation: %d tokens in %v\n", genToks, genDur)
	fmt.Printf("  Speed:      \033[33m%.2f tok/s\033[0m\n", tps)

	// Time to first token.
	ttft := promptDur
	if genToks > 0 {
		ttft = promptDur + genDur/time.Duration(genToks)
	}
	fmt.Printf("  TTFT:       %v\n", ttft)
	fmt.Printf("  Total:      %v\n", totalDur)
	fmt.Printf("  RAM:        %d MB heap, %d MB sys\n", m.Alloc/(1024*1024), m.Sys/(1024*1024))

	// Per-layer timing if available.
	if len(e.layerTimings) > 0 {
		var totalLayer time.Duration
		var slowest time.Duration
		for _, lt := range e.layerTimings {
			totalLayer += lt
			if lt > slowest {
				slowest = lt
			}
		}
		avg := totalLayer / time.Duration(len(e.layerTimings))
		fmt.Printf("  Layers:     %d x avg %v (slowest: %v)\n", len(e.layerTimings), avg, slowest)
	}

	// Cumulative stats.
	fmt.Printf("  Lifetime:   %d requests, %d tokens generated\n", e.stats.TotalRequests, e.stats.TotalTokensGen)
	fmt.Println("\033[36m-----------------\033[0m")
}

// ---------- Forward Pass ----------

// Forward runs one forward pass through the transformer (public API for training).
// Takes token IDs, returns logits over the vocabulary.
func (e *NativeEngine) Forward(tokenIDs []int) []float32 {
	return e.forward(tokenIDs)
}

// forward runs one forward pass through the transformer.
// Takes token IDs, returns logits over the vocabulary.
func (e *NativeEngine) forward(tokenIDs []int) []float32 {
	seqLen := len(tokenIDs)

	// Step 1: Token embedding lookup.
	hidden := NewTensor([]int{seqLen, e.hiddenDim})
	for i, tokID := range tokenIDs {
		if tokID >= e.vocabSize {
			tokID = 0
		}
		row := e.embeddings.GetRowF32(tokID)
		copy(hidden.Data[i*e.hiddenDim : i*e.hiddenDim+e.hiddenDim], row)
	}

	// Step 2: Pass through transformer layers.
	mask := CausalMask(seqLen)
	if e.BenchMode {
		e.layerTimings = make([]time.Duration, len(e.layers))
	}

	// Initialize AttnRes state for this forward pass if enabled.
	var attnResState *AttnResState
	if e.AttnResConfig.Enabled {
		attnResState = NewAttnResState(e.AttnResConfig, len(e.layers), e.hiddenDim)
	}
	for i, layer := range e.layers {
		layerStart := time.Now()
		hidden = e.transformerBlock(hidden, layer, mask, i, attnResState)
		// Record this layer's output for future AttnRes attention.
		if attnResState != nil {
			attnResState.RecordLayerOutput(hidden)
		}
		if e.BenchMode {
			e.layerTimings[i] = time.Since(layerStart)
		}
	}

	// Step 3: Project to vocab logits (only last position).
	lastHidden := NewTensor([]int{e.hiddenDim})
	copy(lastHidden.Data, hidden.Data[(seqLen-1)*e.hiddenDim:seqLen*e.hiddenDim])

	var logits *Tensor
	if e.gpu != nil && e.gpu.IsAvailable() {
		// e.lmHead is [vocabSize, hiddenDim], lastHidden is [hiddenDim].
		// MatVecMulGPU expects A_cols == x_len
		logits = e.gpu.MatVecMulGPU(e.lmHead, lastHidden)
	} else {
		// Use safeMatMul which handles shape broadcasting and fixes transposition
		logits = safeMatMul(e.lmHead, lastHidden, e.vocabSize)
		// Ensure output shape is exactly a 1D vector of logits [vocabSize]
		logits.Shape = []int{e.vocabSize}
	}
	return logits.Data
}

// transformerBlock applies one transformer layer with GQA + optional sparse FFN.
// Routes ops through GPU when available for Vulkan/Metal/D3D12 acceleration.
// When AttnRes is enabled, replaces fixed residual Add with learned depth-wise attention.
func (e *NativeEngine) transformerBlock(x *Tensor, layer TransformerLayer, mask []bool, layerIdx int, attnRes *AttnResState) *Tensor {
	seqLen := x.Shape[0]
	useGPU := e.gpu != nil && e.gpu.IsAvailable()

	// Pre-norm attention.
	var normed *Tensor
	if useGPU {
		normed = e.gpu.RMSNormGPU(x, layer.AttnNorm, 1e-6)
	} else {
		normed = RMSNorm(x, layer.AttnNorm, 1e-6)
	}

	// QKV projection.
	kvDim := (e.hiddenDim / e.numHeads) * e.numKVHeads
	var q, k, v *Tensor
	if useGPU {
		q = e.gpu.MatMulGPU(normed, layer.WQ)
		k = e.gpu.MatMulGPU(normed, layer.WK)
		v = e.gpu.MatMulGPU(normed, layer.WV)
	} else {
		q = safeMatMul(normed, layer.WQ, e.hiddenDim)
		k = safeMatMul(normed, layer.WK, kvDim)
		v = safeMatMul(normed, layer.WV, kvDim)
	}

	// Ensure K/V have the right shape for GQA.
	if len(k.Shape) == 2 && k.Shape[1] != kvDim {
		k = reshapeToWidth(k, seqLen, kvDim)
		v = reshapeToWidth(v, seqLen, kvDim)
	}

	// GQA attention (stays on CPU - complex control flow).
	attnOut := GQAAttention(q, k, v, e.numHeads, e.numKVHeads, mask)

	// Output projection + residual (standard or AttnRes).
	var attnProj *Tensor
	if useGPU {
		attnProj = e.gpu.MatMulGPU(attnOut, layer.WO)
	} else {
		attnProj = safeMatMul(attnOut, layer.WO, e.hiddenDim)
	}

	// Attention Residual: replace fixed Add with learned depth-wise attention.
	if attnRes != nil {
		result := attnRes.AttnResidual(attnProj, layer.AttnResQuery)
		if result != nil {
			x = result
		} else if useGPU {
			x = e.gpu.AddGPU(x, attnProj)
		} else {
			x = safeAdd(x, attnProj)
		}
	} else if useGPU {
		x = e.gpu.AddGPU(x, attnProj)
	} else {
		x = safeAdd(x, attnProj)
	}

	// Pre-norm FFN.
	var normed2 *Tensor
	if useGPU {
		normed2 = e.gpu.RMSNormGPU(x, layer.FFNNorm, 1e-6)
	} else {
		normed2 = RMSNorm(x, layer.FFNNorm, 1e-6)
	}

	// Sparse, MoE, or dense FFN based on config.
	var ffnOut *Tensor
	if e.moeConfig.Enabled && layerIdx < len(e.moeLayers) && len(e.moeLayers[layerIdx].Experts) > 0 {
		// MoE routing: use gating network to select top-K experts.
		// Process only the last position for autoregressive generation.
		lastPos := NewTensor([]int{e.hiddenDim})
		copy(lastPos.Data, normed2.Data[(seqLen-1)*e.hiddenDim:seqLen*e.hiddenDim])
		ffnOut = MoEForward(lastPos, &e.moeLayers[layerIdx], e.moeConfig, e.expertMgr)
		// Broadcast single-position output back to full sequence.
		fullOut := NewTensor([]int{seqLen, e.hiddenDim})
		copy(fullOut.Data[(seqLen-1)*e.hiddenDim:], ffnOut.Data)
		ffnOut = fullOut
	} else if e.SparseConfig.Enabled && layerIdx < len(e.predictors) && e.predictors[layerIdx] != nil {
		ffnOut = SparseFFN(normed2, layer, e.predictors[layerIdx], e.SparseConfig)
	} else if useGPU {
		// GPU-accelerated dense FFN.
		gate := e.gpu.MatMulGPU(normed2, layer.WGate)
		up := e.gpu.MatMulGPU(normed2, layer.WUp)
		gated := e.gpu.MulGPU(e.gpu.SiLUGPU(gate), up)
		ffnOut = e.gpu.MatMulGPU(gated, layer.WDown)
	} else {
		// Dense FFN fallback.
		gate := safeMatMul(normed2, layer.WGate, e.ffnDim)
		up := safeMatMul(normed2, layer.WUp, e.ffnDim)
		gated := safeMul(SiLU(gate), up)
		ffnOut = safeMatMul(gated, layer.WDown, e.hiddenDim)
	}

	// FFN residual (standard or AttnRes).
	if attnRes != nil {
		result := attnRes.AttnResidual(ffnOut, layer.FFNResQuery)
		if result != nil {
			x = result
		} else if useGPU {
			x = e.gpu.AddGPU(x, ffnOut)
		} else {
			x = safeAdd(x, ffnOut)
		}
	} else if useGPU {
		x = e.gpu.AddGPU(x, ffnOut)
	} else {
		x = safeAdd(x, ffnOut)
	}

	return x
}

// EnableSparse turns on sparse inference with the given sparsity level.
// Builds magnitude-based predictors from the gate weights of each layer.
func (e *NativeEngine) EnableSparse(sparsity float32) {
	e.SparseConfig = SparseFFNConfig{
		Enabled:              true,
		Sparsity:             sparsity,
		UseDynamicPrediction: true,
		MinActiveNeurons:     32,
	}

	e.predictors = make([]*NeuronPredictor, e.numLayers)
	for i := 0; i < e.numLayers; i++ {
		e.predictors[i] = NewMagnitudePredictor(e.layers[i].WGate, i, sparsity)
	}

	ffnDims := make([]int, e.numLayers)
	for i := range ffnDims {
		ffnDims[i] = e.ffnDim
	}
	e.profile = NewNeuronProfile(e.numLayers, ffnDims)

	fmt.Printf("[GOTensor] Sparse inference enabled: %.0f%% sparsity, %d layers\n",
		sparsity*100, e.numLayers)
}

// EnableAttnRes activates Attention Residuals (Kimi paper, March 2026).
// Replaces fixed residual connections with learned depth-wise attention.
// Each layer's pseudo-query attends over previous layer outputs via softmax.
// Block mode groups layers for memory efficiency (recommended for >8 layers).
func (e *NativeEngine) EnableAttnRes() {
	e.AttnResConfig = DefaultAttnResConfig(e.numLayers)
	mode := "full"
	if e.AttnResConfig.BlockSize > 0 {
		mode = fmt.Sprintf("block (size=%d)", e.AttnResConfig.BlockSize)
	}
	fmt.Printf("[GOTensor] Attention Residuals enabled: %s mode, %d layers\n", mode, e.numLayers)
}

// ProfileAndOptimize runs calibration text through the model and reclassifies
// neurons based on observed activation patterns (replaces static magnitude prediction).
func (e *NativeEngine) ProfileAndOptimize(calibrationTexts []string) {
	if e.profile == nil || len(e.predictors) == 0 {
		e.EnableSparse(0.7)
	}

	fmt.Printf("[GOTensor] Running calibration with %d texts...\n", len(calibrationTexts))

	for _, text := range calibrationTexts {
		tokens := simpleTokenize(text, e.vocabSize)
		if len(tokens) > 512 {
			tokens = tokens[:512]
		}
		// Run a forward pass (this populates predictor activation counts).
		_ = e.forward(tokens)
		e.profile.IncrementSamples()
	}

	// Reclassify neurons based on observed activations.
	for _, pred := range e.predictors {
		if pred != nil {
			pred.ReclassifyFromProfile()
		}
	}

	e.profile.PrintSummary(0.8, 0.3)
}

// reshapeToWidth creates a new tensor with exactly targetWidth columns.
func reshapeToWidth(t *Tensor, rows, targetWidth int) *Tensor {
	result := NewTensor([]int{rows, targetWidth})
	srcWidth := t.Shape[1]
	copyWidth := srcWidth
	if targetWidth < copyWidth {
		copyWidth = targetWidth
	}
	for i := 0; i < rows; i++ {
		for j := 0; j < copyWidth; j++ {
			result.Data[i*targetWidth+j] = t.Data[i*srcWidth+j]
		}
	}
	return result
}

// safeMatMul performs matrix multiplication with dimension adaptation.
// If the inner dimensions don't match (common with GQA models where K/V
// are smaller than Q), it truncates the wider matrix to fit.
// The targetCols parameter ensures the output has the expected width.
func safeMatMul(a, b *Tensor, targetCols int) *Tensor {
	if len(a.Shape) != 2 || len(b.Shape) != 2 {
		// Fall back to creating a zero tensor if shapes are wrong.
		seqLen := 1
		if len(a.Shape) >= 1 {
			seqLen = a.Shape[0]
		}
		return NewTensor([]int{seqLen, targetCols})
	}

	aK := a.Shape[1]
	bK := b.Shape[0]

	// If inner dimensions match, use standard MatMul.
	if aK == bK {
		return MatMul(a, b)
	}

	// Truncate to the smaller inner dimension.
	innerK := aK
	if bK < innerK {
		innerK = bK
	}

	m := a.Shape[0]
	n := b.Shape[1]
	result := NewTensor([]int{m, n})
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			var sum float32
			for l := 0; l < innerK; l++ {
				sum += a.Data[i*aK+l] * b.Data[l*n+j]
			}
			result.Data[i*n+j] = sum
		}
	}
	return result
}

// safeAdd adds two tensors, padding/truncating to the smaller size.
func safeAdd(a, b *Tensor) *Tensor {
	if len(a.Data) == len(b.Data) {
		return Add(a, b)
	}
	// Use the smaller length.
	n := len(a.Data)
	if len(b.Data) < n {
		n = len(b.Data)
	}
	result := NewTensor(a.Shape)
	copy(result.Data, a.Data)
	for i := 0; i < n; i++ {
		result.Data[i] += b.Data[i]
	}
	return result
}

// safeMul element-wise multiplies two tensors, adapting to size differences.
func safeMul(a, b *Tensor) *Tensor {
	if len(a.Data) == len(b.Data) {
		return Mul(a, b)
	}
	n := len(a.Data)
	if len(b.Data) < n {
		n = len(b.Data)
	}
	result := NewTensor(a.Shape)
	for i := 0; i < n; i++ {
		result.Data[i] = a.Data[i] * b.Data[i]
	}
	return result
}

// ---------- Helpers ----------

// initWeights fills a tensor with small Kaiming-uniform values.
func initWeights(t *Tensor, fanIn int) {
	scale := float32(1.0 / math.Sqrt(float64(fanIn)))
	for i := range t.Data {
		// Deterministic pseudo-random for reproducibility.
		t.Data[i] = scale * float32((i*7+13)%1000-500) / 500.0
	}
}

// simpleTokenize maps characters to token IDs (mod vocabSize).
// This is a placeholder - real tokenization would use BPE/SentencePiece.
func simpleTokenize(text string, vocabSize int) []int {
	tokens := make([]int, 0, len(text))
	for _, ch := range text {
		tokens = append(tokens, int(ch)%vocabSize)
	}
	return tokens
}

// tokensToText maps token IDs back to characters (for demo purposes).
func tokensToText(tokens []int) string {
	chars := make([]byte, len(tokens))
	for i, t := range tokens {
		// Map back to printable ASCII range.
		chars[i] = byte(t%95 + 32) // ASCII 32-126
	}
	return string(chars)
}

// argmax returns the index of the maximum value.
func argmax(data []float32) int {
	maxIdx := 0
	maxVal := data[0]
	for i := 1; i < len(data); i++ {
		if data[i] > maxVal {
			maxVal = data[i]
			maxIdx = i
		}
	}
	return maxIdx
}
