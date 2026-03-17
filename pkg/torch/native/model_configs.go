// model_configs.go provides pre-set configurations for popular model architectures.
//
// WHAT: Each model family (LLaMA, Mistral, Gemma, Qwen, DeepSeek, etc.) has
// specific architecture parameters: hidden size, number of heads, RoPE base
// frequency, activation function, normalization type, etc.
//
// WHY: Instead of manually specifying every parameter, users can just say
// "load a LLaMA-7B" and the engine knows all the right settings. This also
// enables model-specific optimizations (e.g., GQA head ratio, sliding window
// size for Mistral, MoE config for DeepSeek).
package native

// ModelArch identifies a model architecture family.
type ModelArch string

const (
	ArchLLaMA     ModelArch = "llama"
	ArchMistral   ModelArch = "mistral"
	ArchGemma     ModelArch = "gemma"
	ArchQwen      ModelArch = "qwen2"
	ArchDeepSeek  ModelArch = "deepseek"
	ArchPhi       ModelArch = "phi3"
	ArchStarCoder ModelArch = "starcoder2"
	ArchOLMo      ModelArch = "olmo"
)

// ModelConfig holds all architecture-specific parameters for a model.
type ModelConfig struct {
	Arch          ModelArch
	Name          string
	Params        string // e.g., "7B", "13B", "70B"

	// Transformer dimensions.
	HiddenDim     int
	IntermediateDim int // FFN intermediate size
	NumLayers     int
	NumHeads      int
	NumKVHeads    int // For GQA (Grouped Query Attention). 0 = MHA.
	HeadDim       int
	VocabSize     int

	// RoPE.
	RoPEBase      float64
	RoPEScaling   RoPEScaleMethod
	RoPEScaleFactor float64
	MaxSeqLen     int

	// Architecture details.
	Activation    string // "silu", "gelu", "geglu"
	NormType      string // "rmsnorm", "layernorm"
	NormEps       float64
	TieEmbeddings bool

	// Attention.
	SlidingWindow int // 0 = full attention, >0 = sliding window
	UseFlashAttn  bool

	// MoE (Mixture of Experts).
	IsMoE         bool
	NumExperts    int
	NumActiveExperts int

	// Chat template.
	ChatFormat    ChatFormat
	BOS           int32
	EOS           int32

	// Training defaults.
	DefaultLR       float32
	DefaultBatchSize int
	DefaultWarmup   int
}

// ModelRegistry holds known model configurations.
var ModelRegistry = map[string]ModelConfig{
	// ---- LLaMA Family ----
	"llama-7b": {
		Arch: ArchLLaMA, Name: "LLaMA 7B", Params: "7B",
		HiddenDim: 4096, IntermediateDim: 11008, NumLayers: 32,
		NumHeads: 32, NumKVHeads: 32, HeadDim: 128, VocabSize: 32000,
		RoPEBase: 10000, MaxSeqLen: 4096,
		Activation: "silu", NormType: "rmsnorm", NormEps: 1e-5,
		ChatFormat: LlamaFormat, DefaultLR: 3e-4,
	},
	"llama-13b": {
		Arch: ArchLLaMA, Name: "LLaMA 13B", Params: "13B",
		HiddenDim: 5120, IntermediateDim: 13824, NumLayers: 40,
		NumHeads: 40, NumKVHeads: 40, HeadDim: 128, VocabSize: 32000,
		RoPEBase: 10000, MaxSeqLen: 4096,
		Activation: "silu", NormType: "rmsnorm", NormEps: 1e-5,
		ChatFormat: LlamaFormat, DefaultLR: 2e-4,
	},
	"llama-70b": {
		Arch: ArchLLaMA, Name: "LLaMA 70B", Params: "70B",
		HiddenDim: 8192, IntermediateDim: 28672, NumLayers: 80,
		NumHeads: 64, NumKVHeads: 8, HeadDim: 128, VocabSize: 32000,
		RoPEBase: 10000, MaxSeqLen: 4096,
		Activation: "silu", NormType: "rmsnorm", NormEps: 1e-5,
		ChatFormat: LlamaFormat, DefaultLR: 1.5e-4,
	},
	"llama3-8b": {
		Arch: ArchLLaMA, Name: "LLaMA 3 8B", Params: "8B",
		HiddenDim: 4096, IntermediateDim: 14336, NumLayers: 32,
		NumHeads: 32, NumKVHeads: 8, HeadDim: 128, VocabSize: 128256,
		RoPEBase: 500000, RoPEScaling: RoPEScaleYaRN, MaxSeqLen: 8192,
		Activation: "silu", NormType: "rmsnorm", NormEps: 1e-5,
		ChatFormat: ChatMLFormat, DefaultLR: 2e-4,
	},
	"llama3-70b": {
		Arch: ArchLLaMA, Name: "LLaMA 3 70B", Params: "70B",
		HiddenDim: 8192, IntermediateDim: 28672, NumLayers: 80,
		NumHeads: 64, NumKVHeads: 8, HeadDim: 128, VocabSize: 128256,
		RoPEBase: 500000, RoPEScaling: RoPEScaleYaRN, MaxSeqLen: 8192,
		Activation: "silu", NormType: "rmsnorm", NormEps: 1e-5,
		ChatFormat: ChatMLFormat, DefaultLR: 1e-4,
	},

	// ---- Mistral Family ----
	"mistral-7b": {
		Arch: ArchMistral, Name: "Mistral 7B", Params: "7B",
		HiddenDim: 4096, IntermediateDim: 14336, NumLayers: 32,
		NumHeads: 32, NumKVHeads: 8, HeadDim: 128, VocabSize: 32000,
		RoPEBase: 10000, MaxSeqLen: 32768, SlidingWindow: 4096,
		Activation: "silu", NormType: "rmsnorm", NormEps: 1e-5,
		ChatFormat: MistralFormat, DefaultLR: 2e-4,
	},
	"mixtral-8x7b": {
		Arch: ArchMistral, Name: "Mixtral 8x7B", Params: "47B",
		HiddenDim: 4096, IntermediateDim: 14336, NumLayers: 32,
		NumHeads: 32, NumKVHeads: 8, HeadDim: 128, VocabSize: 32000,
		RoPEBase: 10000, MaxSeqLen: 32768, SlidingWindow: 4096,
		Activation: "silu", NormType: "rmsnorm", NormEps: 1e-5,
		IsMoE: true, NumExperts: 8, NumActiveExperts: 2,
		ChatFormat: MistralFormat, DefaultLR: 1e-4,
	},

	// ---- Gemma Family ----
	"gemma-2b": {
		Arch: ArchGemma, Name: "Gemma 2B", Params: "2B",
		HiddenDim: 2048, IntermediateDim: 16384, NumLayers: 18,
		NumHeads: 8, NumKVHeads: 1, HeadDim: 256, VocabSize: 256000,
		RoPEBase: 10000, MaxSeqLen: 8192,
		Activation: "geglu", NormType: "rmsnorm", NormEps: 1e-6,
		ChatFormat: ChatMLFormat, DefaultLR: 5e-4, TieEmbeddings: true,
	},
	"gemma-7b": {
		Arch: ArchGemma, Name: "Gemma 7B", Params: "7B",
		HiddenDim: 3072, IntermediateDim: 24576, NumLayers: 28,
		NumHeads: 16, NumKVHeads: 16, HeadDim: 256, VocabSize: 256000,
		RoPEBase: 10000, MaxSeqLen: 8192,
		Activation: "geglu", NormType: "rmsnorm", NormEps: 1e-6,
		ChatFormat: ChatMLFormat, DefaultLR: 3e-4, TieEmbeddings: true,
	},

	// ---- Qwen2 Family ----
	"qwen2-7b": {
		Arch: ArchQwen, Name: "Qwen2 7B", Params: "7B",
		HiddenDim: 3584, IntermediateDim: 18944, NumLayers: 28,
		NumHeads: 28, NumKVHeads: 4, HeadDim: 128, VocabSize: 152064,
		RoPEBase: 1000000, MaxSeqLen: 131072,
		Activation: "silu", NormType: "rmsnorm", NormEps: 1e-6,
		ChatFormat: ChatMLFormat, DefaultLR: 2e-4,
	},
	"qwen2-72b": {
		Arch: ArchQwen, Name: "Qwen2 72B", Params: "72B",
		HiddenDim: 8192, IntermediateDim: 29568, NumLayers: 80,
		NumHeads: 64, NumKVHeads: 8, HeadDim: 128, VocabSize: 152064,
		RoPEBase: 1000000, MaxSeqLen: 131072,
		Activation: "silu", NormType: "rmsnorm", NormEps: 1e-5,
		ChatFormat: ChatMLFormat, DefaultLR: 1e-4,
	},

	// ---- DeepSeek Family ----
	"deepseek-v3": {
		Arch: ArchDeepSeek, Name: "DeepSeek V3", Params: "671B",
		HiddenDim: 7168, IntermediateDim: 18432, NumLayers: 61,
		NumHeads: 128, NumKVHeads: 128, HeadDim: 128, VocabSize: 129280,
		RoPEBase: 10000, RoPEScaling: RoPEScaleYaRN, MaxSeqLen: 163840,
		Activation: "silu", NormType: "rmsnorm", NormEps: 1e-6,
		IsMoE: true, NumExperts: 256, NumActiveExperts: 8,
		ChatFormat: ChatMLFormat, DefaultLR: 7.3e-5,
	},

	// ---- Phi Family ----
	"phi3-mini": {
		Arch: ArchPhi, Name: "Phi-3 Mini", Params: "3.8B",
		HiddenDim: 3072, IntermediateDim: 8192, NumLayers: 32,
		NumHeads: 32, NumKVHeads: 32, HeadDim: 96, VocabSize: 32064,
		RoPEBase: 10000, MaxSeqLen: 131072,
		Activation: "silu", NormType: "rmsnorm", NormEps: 1e-5,
		ChatFormat: ChatMLFormat, DefaultLR: 3e-4,
	},
}

// GetModelConfig looks up a model configuration by name.
func GetModelConfig(name string) (ModelConfig, bool) {
	config, ok := ModelRegistry[name]
	return config, ok
}

// ListModels returns all registered model names.
func ListModels() []string {
	names := make([]string, 0, len(ModelRegistry))
	for name := range ModelRegistry {
		names = append(names, name)
	}
	return names
}

// GQARatio returns the KV head grouping ratio (numHeads / numKVHeads).
// 1 = standard MHA, >1 = Grouped Query Attention.
func (mc ModelConfig) GQARatio() int {
	if mc.NumKVHeads <= 0 {
		return 1
	}
	return mc.NumHeads / mc.NumKVHeads
}

// MemoryEstimateMB estimates VRAM needed in MB for inference (FP16).
func (mc ModelConfig) MemoryEstimateMB() int {
	// Rough estimate: 2 bytes per parameter (FP16).
	paramsPerLayer := mc.HiddenDim*mc.IntermediateDim*3 + // FFN (gate, up, down)
		mc.HiddenDim*mc.NumHeads*mc.HeadDim*3 + // QKV projections
		mc.HiddenDim*mc.NumHeads*mc.HeadDim // Output projection

	totalParams := paramsPerLayer * mc.NumLayers
	totalParams += mc.VocabSize * mc.HiddenDim * 2 // Embedding + LM head

	if mc.IsMoE {
		// MoE multiplies FFN parameters by number of experts.
		expertParams := mc.HiddenDim * mc.IntermediateDim * 3 * (mc.NumExperts - 1)
		totalParams += expertParams
	}

	return totalParams * 2 / (1024 * 1024) // FP16 bytes -> MB
}
