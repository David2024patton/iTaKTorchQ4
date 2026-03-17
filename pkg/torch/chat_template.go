// chat_template.go provides automatic chat template detection and rendering.
//
// Three-level template resolution:
//   1. GGUF metadata (via llama_model_chat_template FFI) - works for all GGUF models
//   2. Local tokenizer_config.json - works for HuggingFace models stored on disk
//   3. HuggingFace API fetch - fetches tokenizer_config.json from HF Hub by repo name
//   4. Model name heuristic - matches "llama", "qwen", "gemma" etc. in the filename
//
// Falls back through each level until a template is found.
package torch

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// TemplateType identifies which chat template format the model uses.
type TemplateType int

const (
	TemplatePlain   TemplateType = iota
	TemplateChatML
	TemplateLlama3
	TemplateMistral
	TemplateGemma
	TemplatePhi
	TemplateDeepSeek
	TemplateCommand
)

// ChatTemplate holds the detected template type and renders prompts.
type ChatTemplate struct {
	Type     TemplateType
	RawJinja string
	Name     string
}

// DetectChatTemplate analyzes a Jinja2 template string and returns the type.
func DetectChatTemplate(jinjaTemplate string) *ChatTemplate {
	if jinjaTemplate == "" {
		return &ChatTemplate{Type: TemplatePlain, Name: "plain"}
	}
	t := jinjaTemplate
	switch {
	case strings.Contains(t, "start_header_id"):
		return &ChatTemplate{Type: TemplateLlama3, RawJinja: t, Name: "llama3"}
	case strings.Contains(t, "<start_of_turn>"):
		return &ChatTemplate{Type: TemplateGemma, RawJinja: t, Name: "gemma"}
	case strings.Contains(t, "[INST]"):
		return &ChatTemplate{Type: TemplateMistral, RawJinja: t, Name: "mistral"}
	case strings.Contains(t, "<|system|>") && strings.Contains(t, "<|end|>"):
		return &ChatTemplate{Type: TemplatePhi, RawJinja: t, Name: "phi"}
	case strings.Contains(t, "<|begin_of_sentence|>"):
		return &ChatTemplate{Type: TemplateDeepSeek, RawJinja: t, Name: "deepseek"}
	case strings.Contains(t, "START_OF_TURN"):
		return &ChatTemplate{Type: TemplateCommand, RawJinja: t, Name: "command-r"}
	case strings.Contains(t, "im_start"):
		return &ChatTemplate{Type: TemplateChatML, RawJinja: t, Name: "chatml"}
	default:
		return &ChatTemplate{Type: TemplatePlain, RawJinja: t, Name: "plain"}
	}
}

// Apply renders messages using the detected chat template format.
func (ct *ChatTemplate) Apply(messages []ChatMessage) string {
	switch ct.Type {
	case TemplateChatML:
		return applyChatML(messages)
	case TemplateLlama3:
		return applyLlama3(messages)
	case TemplateMistral:
		return applyMistral(messages)
	case TemplateGemma:
		return applyGemma(messages)
	case TemplatePhi:
		return applyPhi(messages)
	case TemplateDeepSeek:
		return applyDeepSeek(messages)
	case TemplateCommand:
		return applyCommand(messages)
	default:
		return applyPlain(messages)
	}
}

func (ct *ChatTemplate) String() string { return ct.Name }

// applyChatML: <|im_start|>role\ncontent<|im_end|>\n
func applyChatML(msgs []ChatMessage) string {
	var sb strings.Builder
	for _, m := range msgs {
		fmt.Fprintf(&sb, "<|im_start|>%s\n%s<|im_end|>\n", m.Role, m.Content)
	}
	sb.WriteString("<|im_start|>assistant\n")
	return sb.String()
}

// applyLlama3: <|begin_of_text|><|start_header_id|>role<|end_header_id|>\n\ncontent<|eot_id|>
func applyLlama3(msgs []ChatMessage) string {
	var sb strings.Builder
	sb.WriteString("<|begin_of_text|>")
	for _, m := range msgs {
		fmt.Fprintf(&sb, "<|start_header_id|>%s<|end_header_id|>\n\n%s<|eot_id|>", m.Role, m.Content)
	}
	sb.WriteString("<|start_header_id|>assistant<|end_header_id|>\n\n")
	return sb.String()
}

// applyMistral: <s>[INST] system\n\nuser [/INST] assistant</s>
func applyMistral(msgs []ChatMessage) string {
	var sb strings.Builder
	sb.WriteString("<s>")
	sysPrompt := ""
	var nonSys []ChatMessage
	for _, m := range msgs {
		if m.Role == "system" {
			sysPrompt = m.Content
		} else {
			nonSys = append(nonSys, m)
		}
	}
	for i, m := range nonSys {
		if m.Role == "user" {
			sb.WriteString("[INST] ")
			if i == 0 && sysPrompt != "" {
				sb.WriteString(sysPrompt)
				sb.WriteString("\n\n")
			}
			sb.WriteString(m.Content)
			sb.WriteString(" [/INST]")
		} else if m.Role == "assistant" {
			sb.WriteString(" ")
			sb.WriteString(m.Content)
			sb.WriteString("</s>")
		}
	}
	return sb.String()
}

// applyGemma: <start_of_turn>role\ncontent<end_of_turn>\n
func applyGemma(msgs []ChatMessage) string {
	var sb strings.Builder
	for _, m := range msgs {
		role := m.Role
		if role == "assistant" {
			role = "model"
		}
		fmt.Fprintf(&sb, "<start_of_turn>%s\n%s<end_of_turn>\n", role, m.Content)
	}
	sb.WriteString("<start_of_turn>model\n")
	return sb.String()
}

// applyPhi: wraps each message with role-specific tags and end marker
func applyPhi(msgs []ChatMessage) string {
	var sb strings.Builder
	for _, m := range msgs {
		tag := m.Role
		fmt.Fprintf(&sb, "<|%s|>\n%s<|end|>\n", tag, m.Content)
	}
	sb.WriteString("<|assistant|>\n")
	return sb.String()
}

// applyDeepSeek: sentence-level wrapping with role markers
func applyDeepSeek(msgs []ChatMessage) string {
	var sb strings.Builder
	sb.WriteString("<|begin_of_sentence|>")
	for _, m := range msgs {
		switch m.Role {
		case "system":
			fmt.Fprintf(&sb, "%s\n\n", m.Content)
		case "user":
			fmt.Fprintf(&sb, "<|User|>%s\n\n", m.Content)
		case "assistant":
			fmt.Fprintf(&sb, "<|Assistant|>%s<|end_of_sentence|>", m.Content)
		}
	}
	sb.WriteString("<|Assistant|>")
	return sb.String()
}

// applyCommand: Cohere Command-R format with uppercase turn markers
func applyCommand(msgs []ChatMessage) string {
	var sb strings.Builder
	for _, m := range msgs {
		role := strings.ToUpper(m.Role)
		if role == "ASSISTANT" {
			role = "CHATBOT"
		}
		fmt.Fprintf(&sb, "<|START_OF_TURN|>%s\n%s<|END_OF_TURN|>", role, m.Content)
	}
	sb.WriteString("<|START_OF_TURN|>CHATBOT\n")
	return sb.String()
}

// applyPlain: simple "Role: content" fallback for unknown templates
func applyPlain(msgs []ChatMessage) string {
	var sb strings.Builder
	for _, m := range msgs {
		switch m.Role {
		case "system":
			fmt.Fprintf(&sb, "System: %s\n\n", m.Content)
		case "user":
			fmt.Fprintf(&sb, "User: %s\n\n", m.Content)
		case "assistant":
			fmt.Fprintf(&sb, "Assistant: %s\n\n", m.Content)
		default:
			fmt.Fprintf(&sb, "%s: %s\n\n", m.Role, m.Content)
		}
	}
	sb.WriteString("Assistant: ")
	return sb.String()
}

// ---------- HuggingFace Template Resolution ----------

// tokenizerConfig represents the relevant fields from tokenizer_config.json.
type tokenizerConfig struct {
	ChatTemplate interface{} `json:"chat_template"` // string or []object
}

// LoadTemplateFromTokenizerConfig reads a tokenizer_config.json file and
// extracts the chat_template field. This is the standard location for
// HuggingFace transformer models.
func LoadTemplateFromTokenizerConfig(configPath string) (*ChatTemplate, error) {
	data, err := os.ReadFile(configPath)
	if err != nil {
		return nil, err
	}

	var config tokenizerConfig
	if err := json.Unmarshal(data, &config); err != nil {
		return nil, fmt.Errorf("parse tokenizer_config.json: %w", err)
	}

	template := extractChatTemplate(config.ChatTemplate)
	if template == "" {
		return nil, fmt.Errorf("no chat_template field in tokenizer_config.json")
	}

	return DetectChatTemplate(template), nil
}

// FetchTemplateFromHF fetches tokenizer_config.json from HuggingFace Hub
// and extracts the chat template. Uses the raw file API endpoint.
//
// Example repo: "meta-llama/Llama-3.1-8B-Instruct"
func FetchTemplateFromHF(repo string) (*ChatTemplate, error) {
	url := fmt.Sprintf("https://huggingface.co/%s/raw/main/tokenizer_config.json", repo)

	client := &http.Client{Timeout: 10 * time.Second}
	resp, err := client.Get(url)
	if err != nil {
		return nil, fmt.Errorf("fetch tokenizer_config.json: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("HF API returned %d for %s", resp.StatusCode, repo)
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("read response: %w", err)
	}

	var config tokenizerConfig
	if err := json.Unmarshal(body, &config); err != nil {
		return nil, fmt.Errorf("parse response: %w", err)
	}

	template := extractChatTemplate(config.ChatTemplate)
	if template == "" {
		return nil, fmt.Errorf("no chat_template in HF tokenizer_config.json")
	}

	return DetectChatTemplate(template), nil
}

// InferTemplateFromModelName guesses the chat template from the model filename.
// This is the last-resort fallback when no metadata or config file is available.
func InferTemplateFromModelName(modelName string) *ChatTemplate {
	lower := strings.ToLower(modelName)
	switch {
	case strings.Contains(lower, "llama-3") || strings.Contains(lower, "llama3"):
		return &ChatTemplate{Type: TemplateLlama3, Name: "llama3 (inferred)"}
	case strings.Contains(lower, "gemma"):
		return &ChatTemplate{Type: TemplateGemma, Name: "gemma (inferred)"}
	case strings.Contains(lower, "mistral") || strings.Contains(lower, "mixtral"):
		return &ChatTemplate{Type: TemplateMistral, Name: "mistral (inferred)"}
	case strings.Contains(lower, "phi"):
		return &ChatTemplate{Type: TemplatePhi, Name: "phi (inferred)"}
	case strings.Contains(lower, "deepseek"):
		return &ChatTemplate{Type: TemplateDeepSeek, Name: "deepseek (inferred)"}
	case strings.Contains(lower, "command"):
		return &ChatTemplate{Type: TemplateCommand, Name: "command-r (inferred)"}
	case strings.Contains(lower, "qwen") || strings.Contains(lower, "yi-"):
		return &ChatTemplate{Type: TemplateChatML, Name: "chatml (inferred)"}
	default:
		return &ChatTemplate{Type: TemplatePlain, Name: "plain"}
	}
}

// ResolveTemplate tries all available sources to find the correct chat template:
//   1. ggufTemplate: raw Jinja2 from GGUF model metadata (already extracted)
//   2. modelPath: check for tokenizer_config.json next to the model file
//   3. modelName: infer from the model filename as last resort
func ResolveTemplate(ggufTemplate, modelPath, modelName string) *ChatTemplate {
	// Level 1: GGUF metadata (most reliable for GGUF models).
	if ggufTemplate != "" {
		ct := DetectChatTemplate(ggufTemplate)
		if ct.Type != TemplatePlain {
			ct.Name += " (gguf)"
			return ct
		}
	}

	// Level 2: Local tokenizer_config.json (for HF models on disk).
	if modelPath != "" {
		modelDir := filepath.Dir(modelPath)
		configPath := filepath.Join(modelDir, "tokenizer_config.json")
		if ct, err := LoadTemplateFromTokenizerConfig(configPath); err == nil {
			ct.Name += " (tokenizer_config)"
			fmt.Printf("[iTaK Torch] Chat template from tokenizer_config.json: %s\n", ct.Name)
			return ct
		}
	}

	// Level 3: Model name heuristic (last resort).
	if modelName != "" {
		ct := InferTemplateFromModelName(modelName)
		if ct.Type != TemplatePlain {
			fmt.Printf("[iTaK Torch] Chat template inferred from model name: %s\n", ct.Name)
			return ct
		}
	}

	return &ChatTemplate{Type: TemplatePlain, Name: "plain (no template found)"}
}

// extractChatTemplate handles the polymorphic chat_template field in
// tokenizer_config.json. It can be either a plain string or an array
// of objects like [{"name": "default", "template": "..."}].
func extractChatTemplate(raw interface{}) string {
	if raw == nil {
		return ""
	}

	// Case 1: Simple string.
	if s, ok := raw.(string); ok {
		return s
	}

	// Case 2: Array of template objects (HF's multi-template format).
	// Pick the "default" template, or the first one if no default.
	if arr, ok := raw.([]interface{}); ok {
		var firstTemplate string
		for _, item := range arr {
			if obj, ok := item.(map[string]interface{}); ok {
				tmpl, _ := obj["template"].(string)
				name, _ := obj["name"].(string)
				if name == "default" || name == "" {
					return tmpl
				}
				if firstTemplate == "" {
					firstTemplate = tmpl
				}
			}
		}
		return firstTemplate
	}

	return ""
}

// ---------- Prefix Rendering (for Prefix Caching) ----------

// RenderPrefix renders only the system message portion of the prompt using
// the detected chat template format. Used for prefix caching to extract the
// stable prefix that stays constant across different user queries.
// Returns empty string if no system messages are present.
func (ct *ChatTemplate) RenderPrefix(messages []ChatMessage) string {
	var sysOnly []ChatMessage
	for _, m := range messages {
		if m.Role == "system" {
			sysOnly = append(sysOnly, m)
		}
	}
	if len(sysOnly) == 0 {
		return ""
	}

	switch ct.Type {
	case TemplateChatML:
		return renderPrefixChatML(sysOnly)
	case TemplateLlama3:
		return renderPrefixLlama3(sysOnly)
	case TemplateMistral:
		return renderPrefixMistral(sysOnly)
	case TemplateGemma:
		return renderPrefixGemma(sysOnly)
	case TemplatePhi:
		return renderPrefixPhi(sysOnly)
	case TemplateDeepSeek:
		return renderPrefixDeepSeek(sysOnly)
	case TemplateCommand:
		return renderPrefixCommand(sysOnly)
	default:
		return renderPrefixPlain(sysOnly)
	}
}

// Prefix renderers: same structure as Apply renderers but WITHOUT the
// trailing assistant prompt. Only system messages are rendered.

func renderPrefixChatML(msgs []ChatMessage) string {
	var sb strings.Builder
	for _, m := range msgs {
		fmt.Fprintf(&sb, "<|im_start|>%s\n%s<|im_end|>\n", m.Role, m.Content)
	}
	return sb.String()
}

func renderPrefixLlama3(msgs []ChatMessage) string {
	var sb strings.Builder
	sb.WriteString("<|begin_of_text|>")
	for _, m := range msgs {
		fmt.Fprintf(&sb, "<|start_header_id|>%s<|end_header_id|>\n\n%s<|eot_id|>", m.Role, m.Content)
	}
	return sb.String()
}

func renderPrefixMistral(msgs []ChatMessage) string {
	// Mistral prepends the system prompt to the first [INST] block.
	// The prefix is the opening tag + raw system content.
	var sb strings.Builder
	sb.WriteString("<s>")
	for _, m := range msgs {
		sb.WriteString(m.Content)
	}
	return sb.String()
}

func renderPrefixGemma(msgs []ChatMessage) string {
	var sb strings.Builder
	for _, m := range msgs {
		role := m.Role
		if role == "assistant" {
			role = "model"
		}
		fmt.Fprintf(&sb, "<start_of_turn>%s\n%s<end_of_turn>\n", role, m.Content)
	}
	return sb.String()
}

func renderPrefixPhi(msgs []ChatMessage) string {
	var sb strings.Builder
	for _, m := range msgs {
		fmt.Fprintf(&sb, "<|%s|>\n%s<|end|>\n", m.Role, m.Content)
	}
	return sb.String()
}

func renderPrefixDeepSeek(msgs []ChatMessage) string {
	var sb strings.Builder
	sb.WriteString("<|begin_of_sentence|>")
	for _, m := range msgs {
		fmt.Fprintf(&sb, "%s\n\n", m.Content)
	}
	return sb.String()
}

func renderPrefixCommand(msgs []ChatMessage) string {
	var sb strings.Builder
	for _, m := range msgs {
		role := strings.ToUpper(m.Role)
		fmt.Fprintf(&sb, "<|START_OF_TURN|>%s\n%s<|END_OF_TURN|>", role, m.Content)
	}
	return sb.String()
}

func renderPrefixPlain(msgs []ChatMessage) string {
	var sb strings.Builder
	for _, m := range msgs {
		fmt.Fprintf(&sb, "System: %s\n\n", m.Content)
	}
	return sb.String()
}

// BuildPrefixPrompt renders only system messages using the plain format.
// Companion to BuildPrompt in server.go for prefix caching without a chat template.
func BuildPrefixPrompt(messages []ChatMessage) string {
	var sb strings.Builder
	for _, m := range messages {
		if m.Role == "system" {
			sb.WriteString(fmt.Sprintf("System: %s\n\n", m.Content))
		}
	}
	return sb.String()
}
