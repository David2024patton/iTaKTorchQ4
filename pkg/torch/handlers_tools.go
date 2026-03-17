// handlers_tools.go implements OpenAI-compatible tool/function calling.
//
// When a request includes "tools" with function definitions, Torch uses
// grammar-constrained decoding to force the model to output valid JSON
// tool_calls that match the function signatures. This guarantees 100%
// schema compliance even on smaller models.
package torch

import (
	"encoding/json"
	"fmt"
	"strings"
)

// ---------- Tool Calling Types ----------

// ToolDefinition describes a tool the model can call.
type ToolDefinition struct {
	Type     string       `json:"type"` // "function"
	Function FunctionDef  `json:"function"`
}

// FunctionDef describes a callable function.
type FunctionDef struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters,omitempty"` // JSON Schema
}

// ToolCall represents a tool invocation requested by the model.
type ToolCall struct {
	ID       string       `json:"id"`
	Type     string       `json:"type"` // "function"
	Function FunctionCall `json:"function"`
}

// FunctionCall holds the function name and arguments.
type FunctionCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"` // JSON string
}

// ToolChatRequest extends ChatRequest with tool definitions.
type ToolChatRequest struct {
	ChatRequest
	Tools      []ToolDefinition `json:"tools,omitempty"`
	ToolChoice interface{}      `json:"tool_choice,omitempty"` // "auto", "none", or specific
}

// ToolChatResponse extends ChatResponse with tool calls.
type ToolChatResponse struct {
	ChatResponse
}

// ToolChatChoice extends ChatChoice with tool_calls.
type ToolChatChoice struct {
	Index        int         `json:"index"`
	Message      ToolMessage `json:"message"`
	FinishReason string      `json:"finish_reason"`
}

// ToolMessage extends ChatMessage with tool_calls.
type ToolMessage struct {
	Role      string     `json:"role"`
	Content   *string    `json:"content"` // null when tool_calls present
	ToolCalls []ToolCall `json:"tool_calls,omitempty"`
}

// ---------- Grammar Generation ----------

// BuildToolCallGrammar generates a GBNF grammar that constrains model output
// to valid tool_calls JSON matching the provided function signatures.
//
// The grammar forces the model to output exactly:
// {"name": "<one of the function names>", "arguments": {<valid json matching schema>}}
func BuildToolCallGrammar(tools []ToolDefinition) string {
	if len(tools) == 0 {
		return ""
	}

	var b strings.Builder

	// Root rule: a JSON object with "name" and "arguments" fields.
	b.WriteString("root ::= \"{\" ws ")
	b.WriteString(`"\"name\"" ws ":" ws name-value ws "," ws `)
	b.WriteString(`"\"arguments\"" ws ":" ws arguments-value ws `)
	b.WriteString(`"}"`)
	b.WriteString("\n\n")

	// Name rule: must be one of the function names.
	b.WriteString("name-value ::= ")
	for i, tool := range tools {
		if i > 0 {
			b.WriteString(" | ")
		}
		b.WriteString(fmt.Sprintf(`"\"" "%s" "\""`, tool.Function.Name))
	}
	b.WriteString("\n\n")

	// Arguments rule: valid JSON object.
	b.WriteString("arguments-value ::= \"{\" ws (argument-pair (ws \",\" ws argument-pair)*)? ws \"}\"\n")
	b.WriteString("argument-pair ::= string ws \":\" ws value\n\n")

	// Standard JSON value types.
	b.WriteString("value ::= string | number | object | array | \"true\" | \"false\" | \"null\"\n")
	b.WriteString("string ::= \"\\\"\" ([^\"\\\\] | \"\\\\\" [\"\\\\/bfnrt])* \"\\\"\"\n")
	b.WriteString("number ::= \"-\"? [0-9]+ (\".\" [0-9]+)? ([eE] [\"+\" \"-\"]? [0-9]+)?\n")
	b.WriteString("object ::= \"{\" ws (string ws \":\" ws value (ws \",\" ws string ws \":\" ws value)*)? ws \"}\"\n")
	b.WriteString("array ::= \"[\" ws (value (ws \",\" ws value)*)? ws \"]\"\n")
	b.WriteString("ws ::= [ \\t\\n]*\n")

	return b.String()
}

// ParseToolCalls extracts tool calls from model output.
// The model should output JSON like: {"name":"func_name","arguments":{...}}
func ParseToolCalls(output string, tools []ToolDefinition) ([]ToolCall, error) {
	output = strings.TrimSpace(output)

	// Try parsing as a single tool call.
	var single struct {
		Name      string          `json:"name"`
		Arguments json.RawMessage `json:"arguments"`
	}

	if err := json.Unmarshal([]byte(output), &single); err != nil {
		return nil, fmt.Errorf("failed to parse tool call: %w", err)
	}

	// Validate the function name exists in tools.
	validName := false
	for _, t := range tools {
		if t.Function.Name == single.Name {
			validName = true
			break
		}
	}
	if !validName {
		return nil, fmt.Errorf("model called unknown function %q", single.Name)
	}

	return []ToolCall{
		{
			ID:   fmt.Sprintf("call_%s", single.Name),
			Type: "function",
			Function: FunctionCall{
				Name:      single.Name,
				Arguments: string(single.Arguments),
			},
		},
	}, nil
}

// InjectToolSystemPrompt prepends tool definitions to the system message
// so the model knows what tools are available.
func InjectToolSystemPrompt(messages []ChatMessage, tools []ToolDefinition) []ChatMessage {
	if len(tools) == 0 {
		return messages
	}

	// Build tool descriptions.
	var toolDescs []string
	for _, t := range tools {
		desc := fmt.Sprintf("- %s", t.Function.Name)
		if t.Function.Description != "" {
			desc += fmt.Sprintf(": %s", t.Function.Description)
		}
		if len(t.Function.Parameters) > 0 {
			desc += fmt.Sprintf("\n  Parameters: %s", string(t.Function.Parameters))
		}
		toolDescs = append(toolDescs, desc)
	}

	toolPrompt := fmt.Sprintf(
		"You have access to the following tools:\n%s\n\n"+
			"To call a tool, respond with a JSON object containing 'name' and 'arguments' fields.\n"+
			"Example: {\"name\": \"function_name\", \"arguments\": {\"param\": \"value\"}}",
		strings.Join(toolDescs, "\n"),
	)

	// Prepend to or create system message.
	result := make([]ChatMessage, len(messages))
	copy(result, messages)

	if len(result) > 0 && result[0].Role == "system" {
		result[0].Content = result[0].Content + "\n\n" + toolPrompt
	} else {
		result = append([]ChatMessage{{Role: "system", Content: toolPrompt}}, result...)
	}

	return result
}
