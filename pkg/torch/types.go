package torch

import (
	"encoding/json"
	"time"
)

// ---------- Engine Options ----------

// EngineOpts configures how a GGUF model is loaded and run.
type EngineOpts struct {
	// ContextSize is the max context window (n_ctx). Default: 2048.
	ContextSize int `json:"context_size" yaml:"context_size"`
	// Threads is the number of CPU threads to use. Default: runtime.NumCPU().
	Threads int `json:"threads" yaml:"threads"`
	// GPULayers is how many layers to offload to GPU. 0 = CPU only.
	GPULayers int `json:"gpu_layers" yaml:"gpu_layers"`
	// FlashAttention enables flash attention for faster inference.
	// Auto-enabled when GPULayers > 0 (Phase 3). Set NoFlashAttention to override.
	FlashAttention bool `json:"flash_attention" yaml:"flash_attention"`
	// NoFlashAttention disables the auto-enable of flash attention on GPU mode.
	NoFlashAttention bool `json:"no_flash_attention,omitempty" yaml:"no_flash_attention"`
	// UseMlock locks model weights in RAM to prevent OS swapping. Default: false.
	UseMlock bool `json:"use_mlock" yaml:"use_mlock"`
	// NoMmap disables mmap for model loading. Required on WSL2 where the 9P
	// filesystem triggers SIGBUS during mmap page faults on large GGUF files
	// read from /mnt/ paths. Forces read() instead of mmap(). Default: false.
	NoMmap bool `json:"no_mmap,omitempty" yaml:"no_mmap"`
	// NumaStrategy controls NUMA memory policy (0=disabled, 1=distribute, 2=isolate). Default: 0.
	NumaStrategy int `json:"numa_strategy" yaml:"numa_strategy"`
	// BatchSize is the logical batch size for prompt processing. Default: 2048.
	BatchSize int `json:"batch_size" yaml:"batch_size"`
	// KVCacheType controls quantization of the KV cache. Options: "f16" (default), "q8_0", "q4_0".
	// q8_0 halves KV cache VRAM with minimal quality loss. q4_0 quarters it.
	KVCacheType string `json:"kv_cache_type,omitempty" yaml:"kv_cache_type"`
	// DefragThreshold triggers KV cache defragmentation when fragmentation exceeds this ratio.
	// Default: -1 (disabled). Recommended: 0.1 for long-running sessions.
	DefragThreshold float32 `json:"defrag_threshold,omitempty" yaml:"defrag_threshold"`
	// MaxSlots is the number of concurrent inference slots for continuous batching.
	// Default: 1 (sequential mode). Higher values enable multi-request batching.
	MaxSlots int `json:"max_slots,omitempty" yaml:"max_slots"`
	// Backend selects the GPU compute backend. Options: "auto" (default), "cuda", "vulkan", "cpu".
	// "auto" tries CUDA first, then Vulkan, then HIP, then CPU-only.
	Backend string `json:"backend,omitempty" yaml:"backend"`

	// PrefixCacheSize sets the maximum number of KV states to cache for identical system prompts.
	// Default: 16. Set to 0 to disable prefix caching.
	PrefixCacheSize int `json:"prefix_cache_size,omitempty" yaml:"prefix_cache_size"`

	// --- Speculative Decoding (Phase 3 Stretch) ---

	// DraftModelPath is the path to a small draft model for speculative decoding.
	// When set, the draft model generates candidate tokens that the main model verifies
	// in a single batch pass. Matching tokens are accepted for free, giving N tokens
	// for the cost of 1 main-model forward pass. If empty, standard sequential generation is used.
	DraftModelPath string `json:"draft_model,omitempty" yaml:"draft_model"`
	// DraftGPULayers is GPU layers for the draft model. Default: same as GPULayers.
	DraftGPULayers int `json:"draft_gpu_layers,omitempty" yaml:"draft_gpu_layers"`
	// SpeculativeTokens is how many tokens the draft model predicts ahead per step. Default: 5.
	// Higher values = more speculative work but bigger wins when predictions match.
	SpeculativeTokens int `json:"speculative_tokens,omitempty" yaml:"speculative_tokens"`

	// --- Multi-GPU ---

	// TensorSplit sets how to divide model VRAM usage across multiple GPUs.
	// Format is a comma-separated list of values (e.g. "0.5,0.5" or "10,20").
	// When empty, all compute/memory is mapped to the main GPU.
	TensorSplit string `json:"tensor_split,omitempty" yaml:"tensor_split"`

	// MultiGPU configures multi-GPU tensor parallelism or layer splitting.
	// If nil, auto-detected at startup. Set to override auto-detection.
	MultiGPU *MultiGPUConfig `json:"multi_gpu,omitempty" yaml:"multi_gpu"`

	// KeepAlive is the TTL in seconds before an idle model is unloaded from VRAM.
	// 0 = 5 minutes default. -1 = infinite (never auto-unload based on TTL).
	KeepAlive int `json:"keep_alive,omitempty" yaml:"keep_alive"`
}

// CompletionParams controls inference behavior per request.
type CompletionParams struct {
	MaxTokens      int      `json:"max_tokens,omitempty"`
	Temperature    float64  `json:"temperature,omitempty"`
	TopP           float64  `json:"top_p,omitempty"`
	Stop           []string `json:"stop,omitempty"`

	// Grammar is a GBNF grammar string that constrains the output format.
	// When set, llama.cpp only allows tokens that satisfy the grammar rules.
	// Use built-in constants: GrammarJSON, GrammarJSONArray, GrammarBool.
	// Or write custom GBNF for domain-specific formats.
	Grammar        string   `json:"grammar,omitempty"`

	// ResponseFormat is a shorthand for common grammar constraints.
	// Accepted values: "json", "json_array", "bool", "list".
	// When set, the appropriate built-in grammar is auto-applied.
	// If both Grammar and ResponseFormat are set, Grammar takes priority.
	ResponseFormat string   `json:"response_format,omitempty"`
}

// ---------- Model Manager Types ----------

// ModelEntry represents a cached GGUF model file.
type ModelEntry struct {
	Name     string    `json:"name"`
	Path     string    `json:"path"`
	Size     int64     `json:"size"`
	LastUsed time.Time `json:"last_used"`
}

// ---------- OpenAI-Compatible API Types ----------

// ChatRequest is the incoming request body for /v1/chat/completions.
type ChatRequest struct {
	Model          string          `json:"model"`
	Messages       []ChatMessage   `json:"messages"`
	MaxTokens      int             `json:"max_tokens,omitempty"`
	Temperature    *float64        `json:"temperature,omitempty"`
	TopP           *float64        `json:"top_p,omitempty"`
	Stop           []string        `json:"stop,omitempty"`
	Stream         bool            `json:"stream,omitempty"`
	ResponseFormat *ResponseFormat `json:"response_format,omitempty"`
}

// ChatMessage is a single message in a chat conversation.
// Handles both OpenAI text format (content as string) and vision format
// (content as array of content parts with text + image_url entries).
type ChatMessage struct {
	Role      string        `json:"role"`
	Content   string        `json:"content"`
	ImageData []ContentPart `json:"-"` // populated from content array for vision
}

// UnmarshalJSON handles both string content and array content (OpenAI vision format).
func (m *ChatMessage) UnmarshalJSON(data []byte) error {
	// Try the simple string-content format first.
	type plainMessage struct {
		Role    string `json:"role"`
		Content string `json:"content"`
	}
	var plain plainMessage
	if err := json.Unmarshal(data, &plain); err == nil && plain.Content != "" {
		m.Role = plain.Role
		m.Content = plain.Content
		return nil
	}

	// Try array-content format (OpenAI vision API).
	type arrayMessage struct {
		Role    string          `json:"role"`
		Content json.RawMessage `json:"content"`
	}
	var arrMsg arrayMessage
	if err := json.Unmarshal(data, &arrMsg); err != nil {
		return err
	}
	m.Role = arrMsg.Role

	var parts []ContentPart
	if err := json.Unmarshal(arrMsg.Content, &parts); err != nil {
		// Content was neither string nor array - try raw string again.
		var s string
		if err2 := json.Unmarshal(arrMsg.Content, &s); err2 == nil {
			m.Content = s
			return nil
		}
		return err
	}

	// Split parts into text content and image data.
	for _, part := range parts {
		if part.Type == "text" {
			if m.Content != "" {
				m.Content += "\n"
			}
			m.Content += part.Text
		} else {
			m.ImageData = append(m.ImageData, part)
		}
	}
	return nil
}

// ContentPart represents a single content block in a multi-modal message.
// Type is either "text" or "image_url".
type ContentPart struct {
	Type     string    `json:"type"`
	Text     string    `json:"text,omitempty"`
	ImageURL *ImageURL `json:"image_url,omitempty"`
}

// ImageURL contains the URL (or base64 data URI) for an image.
type ImageURL struct {
	URL    string `json:"url"`
	Detail string `json:"detail,omitempty"` // "auto", "low", "high"
}

// ChatResponse is the response body for /v1/chat/completions.
type ChatResponse struct {
	ID      string       `json:"id"`
	Object  string       `json:"object"`
	Created int64        `json:"created"`
	Model   string       `json:"model"`
	Choices []ChatChoice `json:"choices"`
	Usage   ChatUsage    `json:"usage"`
}

// ChatChoice is a single completion choice.
type ChatChoice struct {
	Index        int         `json:"index"`
	Message      ChatMessage `json:"message"`
	FinishReason string      `json:"finish_reason"`
}

// ChatUsage tracks token consumption for a request.
type ChatUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// ModelsResponse is the response body for GET /v1/models.
type ModelsResponse struct {
	Object string      `json:"object"`
	Data   []ModelInfo `json:"data"`
}

// ModelInfo describes a single model available on this server.
type ModelInfo struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	OwnedBy string `json:"owned_by"`
}

// HealthResponse is the response body for GET /health.
type HealthResponse struct {
	Status    string `json:"status"`
	ModelName string `json:"model_name,omitempty"`
	Uptime    string `json:"uptime,omitempty"`
}
