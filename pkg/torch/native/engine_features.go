// engine_features.go wires advanced features (RoPE, BPE, Flash Attention,
// sliding window, MoE, prompt cache, KV compression) into the engine.
//
// These functions bridge the standalone feature implementations with the
// NativeEngine's forward pass. Call EnableFeature* methods after model load.
package native

import (
	"fmt"
)

// FeatureFlags tracks which advanced features are active.
type FeatureFlags struct {
	RoPE           bool
	BPETokenizer   bool
	FlashAttention bool
	SlidingWindow  bool
	MoE            bool
	PromptCache    bool
	KVCompression  bool
	GBNF           bool
}

// EngineFeatures holds initialized feature instances attached to the engine.
type EngineFeatures struct {
	Flags        FeatureFlags
	RoPECache    *RoPECache
	Tokenizer    *BPETokenizer
	FlashConfig  FlashAttentionConfig
	SWConfig     SlidingWindowConfig
	MoEConfig    MoEConfig
	MoELayers    []MoELayer
	PromptCache  *PromptCache
	KVCompress   *KVCompressedCache
	GBNFGrammar  *GBNFGrammar
	Constrainer  *GBNFConstrainer
}

// EnableRoPE initializes and attaches RoPE to the engine.
func (e *NativeEngine) EnableRoPE() {
	if e.features == nil {
		e.features = &EngineFeatures{}
	}

	headDim := e.hiddenDim / e.numHeads
	maxSeq := 8192 // Default max sequence length
	if e.maxSeqLen > 0 {
		maxSeq = e.maxSeqLen
	}

	config := DefaultRoPEConfig(headDim, maxSeq)
	e.features.RoPECache = NewRoPECache(config)
	e.features.Flags.RoPE = true
	fmt.Printf("[Engine] RoPE enabled: headDim=%d, maxSeq=%d\n", headDim, maxSeq)
}

// EnableBPETokenizer loads the BPE tokenizer from GGUF metadata.
func (e *NativeEngine) EnableBPETokenizer(gf *GGUFFile) error {
	if e.features == nil {
		e.features = &EngineFeatures{}
	}

	tokenizer := NewBPETokenizer()
	if err := tokenizer.LoadFromGGUF(gf); err != nil {
		return fmt.Errorf("load tokenizer: %w", err)
	}

	e.features.Tokenizer = tokenizer
	e.features.Flags.BPETokenizer = true
	return nil
}

// EnableFlashAttention activates memory-efficient tiled attention.
func (e *NativeEngine) EnableFlashAttention() {
	if e.features == nil {
		e.features = &EngineFeatures{}
	}
	e.features.FlashConfig = DefaultFlashConfig()
	e.features.Flags.FlashAttention = true
	fmt.Println("[Engine] Flash Attention enabled (block 64x64)")
}

// EnableSlidingWindow activates windowed attention with the given window size.
func (e *NativeEngine) EnableSlidingWindow(windowSize int) {
	if e.features == nil {
		e.features = &EngineFeatures{}
	}
	e.features.SWConfig = SlidingWindowConfig{
		WindowSize: windowSize,
		Enabled:    true,
	}
	e.features.Flags.SlidingWindow = true
	fmt.Printf("[Engine] Sliding window attention: window=%d\n", windowSize)
}

// EnableMoE activates Mixture of Experts routing.
func (e *NativeEngine) EnableMoE(numExperts, numActive int) {
	if e.features == nil {
		e.features = &EngineFeatures{}
	}
	e.features.MoEConfig = MoEConfig{
		NumExperts: numExperts,
		NumActive:  numActive,
		Enabled:    true,
	}
	e.features.Flags.MoE = true
	fmt.Printf("[Engine] MoE enabled: %d experts, top-%d\n", numExperts, numActive)
}

// EnablePromptCache activates KV cache prefix sharing.
func (e *NativeEngine) EnablePromptCache(maxEntries int) {
	if e.features == nil {
		e.features = &EngineFeatures{}
	}
	e.features.PromptCache = NewPromptCache(maxEntries)
	e.features.Flags.PromptCache = true
	fmt.Printf("[Engine] Prompt cache: max %d entries\n", maxEntries)
}

// EnableKVCompression activates INT8 KV cache compression.
func (e *NativeEngine) EnableKVCompression() {
	if e.features == nil {
		e.features = &EngineFeatures{}
	}
	config := DefaultKVCompressConfig()
	e.features.KVCompress = NewKVCompressedCache(e.numLayers, config)
	e.features.Flags.KVCompression = true
}

// SetGBNFGrammar constrains output to the given grammar.
func (e *NativeEngine) SetGBNFGrammar(grammarStr string) error {
	if e.features == nil {
		e.features = &EngineFeatures{}
	}

	grammar, err := ParseGBNF(grammarStr)
	if err != nil {
		return fmt.Errorf("parse GBNF: %w", err)
	}

	var tokenizer *BPETokenizer
	if e.features.Flags.BPETokenizer {
		tokenizer = e.features.Tokenizer
	}

	e.features.GBNFGrammar = grammar
	e.features.Constrainer = NewGBNFConstrainer(grammar, tokenizer)
	e.features.Flags.GBNF = true
	fmt.Printf("[Engine] GBNF constrained output: root=%s, %d rules\n",
		grammar.Root, len(grammar.Rules))
	return nil
}

// EnableAllFeatures turns on core features with sensible defaults.
func (e *NativeEngine) EnableAllFeatures() {
	e.EnableRoPE()
	e.EnableFlashAttention()
	e.EnablePromptCache(64)
	e.EnableKVCompression()
	fmt.Println("[Engine] All core features enabled")
}

// FeatureStatus returns which features are active.
func (e *NativeEngine) FeatureStatus() map[string]bool {
	if e.features == nil {
		return map[string]bool{}
	}
	return map[string]bool{
		"rope":            e.features.Flags.RoPE,
		"bpe_tokenizer":   e.features.Flags.BPETokenizer,
		"flash_attention": e.features.Flags.FlashAttention,
		"sliding_window":  e.features.Flags.SlidingWindow,
		"moe":             e.features.Flags.MoE,
		"prompt_cache":    e.features.Flags.PromptCache,
		"kv_compression":  e.features.Flags.KVCompression,
		"gbnf":            e.features.Flags.GBNF,
		"attn_res":        e.AttnResConfig.Enabled,
	}
}
