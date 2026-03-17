// tool_calling.go implements function/tool calling parsing for agent use cases.
//
// WHAT: Modern LLMs can output structured "tool calls" that instruct the
// caller to execute a function and return the result. This file parses
// model output to detect tool call patterns and extract the function
// name and arguments.
//
// FORMATS: Supports multiple tool call formats:
//   - OpenAI style:  {"name": "func", "arguments": {...}}
//   - Hermes style:  <tool_call>{"name": "func", "arguments": {...}}</tool_call>
//   - Qwen style:    <|tool_call|>func(args)<|/tool_call|>
//   - Llama style:   [TOOL_CALLS] [{"name": "func", "arguments": {...}}]
package native

import (
	"encoding/json"
	"fmt"
	"strings"
)

// ToolCall represents a parsed function/tool call from model output.
type ToolCall struct {
	ID        string          `json:"id"`
	Name      string          `json:"name"`
	Arguments json.RawMessage `json:"arguments"`
}

// ToolCallResult is what the caller sends back after executing a tool.
type ToolCallResult struct {
	ToolCallID string `json:"tool_call_id"`
	Content    string `json:"content"`
}

// ToolDefinition describes a tool available to the model.
type ToolDefinition struct {
	Name        string                 `json:"name"`
	Description string                 `json:"description"`
	Parameters  map[string]interface{} `json:"parameters"`
}

// ToolCallParser detects and extracts tool calls from model output.
type ToolCallParser struct {
	Tools []ToolDefinition
}

// NewToolCallParser creates a parser with the given tool definitions.
func NewToolCallParser(tools []ToolDefinition) *ToolCallParser {
	return &ToolCallParser{Tools: tools}
}

// Parse attempts to extract tool calls from model output text.
// Returns extracted calls and the remaining non-tool text.
func (p *ToolCallParser) Parse(output string) ([]ToolCall, string) {
	// Try each format in order of specificity.

	// 1. Hermes format: <tool_call>...</tool_call>
	if calls, remaining, ok := p.parseHermes(output); ok {
		return calls, remaining
	}

	// 2. Qwen format: <|tool_call|>...<|/tool_call|>
	if calls, remaining, ok := p.parseQwen(output); ok {
		return calls, remaining
	}

	// 3. Llama format: [TOOL_CALLS] [...]
	if calls, remaining, ok := p.parseLlama(output); ok {
		return calls, remaining
	}

	// 4. Raw JSON object with "name" and "arguments" keys.
	if calls, remaining, ok := p.parseRawJSON(output); ok {
		return calls, remaining
	}

	return nil, output
}

// HasToolCalls does a quick check if the output likely contains tool calls.
func (p *ToolCallParser) HasToolCalls(output string) bool {
	markers := []string{
		"<tool_call>", "<|tool_call|>", "[TOOL_CALLS]",
		`"name"`, `"arguments"`,
	}
	for _, m := range markers {
		if strings.Contains(output, m) {
			return true
		}
	}
	return false
}

func (p *ToolCallParser) parseHermes(output string) ([]ToolCall, string, bool) {
	const startTag = "<tool_call>"
	const endTag = "</tool_call>"

	var calls []ToolCall
	remaining := output

	for {
		start := strings.Index(remaining, startTag)
		if start == -1 {
			break
		}
		end := strings.Index(remaining[start:], endTag)
		if end == -1 {
			break
		}
		end += start

		jsonStr := remaining[start+len(startTag) : end]
		jsonStr = strings.TrimSpace(jsonStr)

		var call ToolCall
		if err := json.Unmarshal([]byte(jsonStr), &call); err == nil {
			if call.ID == "" {
				call.ID = fmt.Sprintf("call_%d", len(calls))
			}
			calls = append(calls, call)
		}

		remaining = remaining[:start] + remaining[end+len(endTag):]
	}

	if len(calls) > 0 {
		return calls, strings.TrimSpace(remaining), true
	}
	return nil, output, false
}

func (p *ToolCallParser) parseQwen(output string) ([]ToolCall, string, bool) {
	const startTag = "<|tool_call|>"
	const endTag = "<|/tool_call|>"

	var calls []ToolCall
	remaining := output

	for {
		start := strings.Index(remaining, startTag)
		if start == -1 {
			break
		}
		end := strings.Index(remaining[start:], endTag)
		if end == -1 {
			break
		}
		end += start

		content := remaining[start+len(startTag) : end]
		content = strings.TrimSpace(content)

		// Try JSON first.
		var call ToolCall
		if err := json.Unmarshal([]byte(content), &call); err == nil {
			if call.ID == "" {
				call.ID = fmt.Sprintf("call_%d", len(calls))
			}
			calls = append(calls, call)
		} else {
			// Try func(args) format.
			if paren := strings.Index(content, "("); paren > 0 {
				name := content[:paren]
				args := content[paren:]
				call = ToolCall{
					ID:        fmt.Sprintf("call_%d", len(calls)),
					Name:      strings.TrimSpace(name),
					Arguments: json.RawMessage(args),
				}
				calls = append(calls, call)
			}
		}

		remaining = remaining[:start] + remaining[end+len(endTag):]
	}

	if len(calls) > 0 {
		return calls, strings.TrimSpace(remaining), true
	}
	return nil, output, false
}

func (p *ToolCallParser) parseLlama(output string) ([]ToolCall, string, bool) {
	const marker = "[TOOL_CALLS]"

	idx := strings.Index(output, marker)
	if idx == -1 {
		return nil, output, false
	}

	jsonPart := strings.TrimSpace(output[idx+len(marker):])
	remaining := strings.TrimSpace(output[:idx])

	var calls []ToolCall
	if err := json.Unmarshal([]byte(jsonPart), &calls); err == nil {
		for i := range calls {
			if calls[i].ID == "" {
				calls[i].ID = fmt.Sprintf("call_%d", i)
			}
		}
		return calls, remaining, true
	}

	return nil, output, false
}

func (p *ToolCallParser) parseRawJSON(output string) ([]ToolCall, string, bool) {
	trimmed := strings.TrimSpace(output)

	// Single tool call object.
	if strings.HasPrefix(trimmed, "{") {
		var call ToolCall
		if err := json.Unmarshal([]byte(trimmed), &call); err == nil && call.Name != "" {
			if call.ID == "" {
				call.ID = "call_0"
			}
			return []ToolCall{call}, "", true
		}
	}

	// Array of tool calls.
	if strings.HasPrefix(trimmed, "[") {
		var calls []ToolCall
		if err := json.Unmarshal([]byte(trimmed), &calls); err == nil && len(calls) > 0 && calls[0].Name != "" {
			for i := range calls {
				if calls[i].ID == "" {
					calls[i].ID = fmt.Sprintf("call_%d", i)
				}
			}
			return calls, "", true
		}
	}

	return nil, output, false
}

// FormatToolsPrompt generates the tool description section for a system prompt.
func FormatToolsPrompt(tools []ToolDefinition) string {
	if len(tools) == 0 {
		return ""
	}

	var sb strings.Builder
	sb.WriteString("You have access to the following tools:\n\n")

	for _, tool := range tools {
		sb.WriteString(fmt.Sprintf("### %s\n", tool.Name))
		sb.WriteString(fmt.Sprintf("%s\n", tool.Description))
		if len(tool.Parameters) > 0 {
			paramsJSON, _ := json.MarshalIndent(tool.Parameters, "", "  ")
			sb.WriteString(fmt.Sprintf("Parameters: %s\n", string(paramsJSON)))
		}
		sb.WriteString("\n")
	}

	sb.WriteString("To call a tool, output: <tool_call>{\"name\": \"tool_name\", \"arguments\": {...}}</tool_call>\n")
	return sb.String()
}
