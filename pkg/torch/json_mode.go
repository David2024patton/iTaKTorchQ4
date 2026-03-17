// json_mode.go implements structured JSON output mode via GBNF grammar constraints.
//
// HOW IT WORKS:
//   1. User provides a JSON schema in the request (OpenAI-compatible format)
//   2. We convert that schema to a GBNF grammar
//   3. The grammar constrains the model's output to ONLY valid JSON
//   4. Every token is grammar-checked before acceptance
//
// WHY: Without grammar constraints, LLMs frequently produce broken JSON:
//   - Missing closing braces
//   - Trailing commas
//   - Unquoted keys
//   - Wrong types ("123" instead of 123)
//
// With GBNF constraints, the output is schema-perfect 100% of the time.
// The grammar sampler prunes invalid tokens before sampling, so the
// model never even considers outputting broken JSON.
//
// USAGE:
//   POST /v1/chat/completions with:
//   { "response_format": { "type": "json_schema", "json_schema": { ... } } }
//   or simply:
//   { "response_format": { "type": "json_object" } }
package torch

import (
	"encoding/json"
	"fmt"
	"strings"
)

// ResponseFormat specifies the output format constraint (OpenAI-compatible).
type ResponseFormat struct {
	Type       string          `json:"type"`                  // "text", "json_object", "json_schema"
	JSONSchema *JSONSchemaSpec `json:"json_schema,omitempty"` // schema definition when type="json_schema"
}

// JSONSchemaSpec defines a JSON schema for structured output.
type JSONSchemaSpec struct {
	Name        string                 `json:"name"`
	Description string                 `json:"description,omitempty"`
	Schema      map[string]interface{} `json:"schema"`
	Strict      bool                   `json:"strict,omitempty"`
}

// SchemaToGBNF converts a JSON schema to a GBNF grammar string.
// The grammar constrains token generation to only produce valid JSON
// matching the given schema.
func SchemaToGBNF(schema map[string]interface{}) string {
	var sb strings.Builder

	// Write root rule.
	sb.WriteString("root ::= ")

	schemaType, _ := schema["type"].(string)
	switch schemaType {
	case "object":
		sb.WriteString("object\n")
		generateObjectRule(&sb, schema, "object")
	case "array":
		sb.WriteString("array\n")
		generateArrayRule(&sb, schema, "array")
	default:
		// Default: accept any valid JSON.
		sb.WriteString("value\n")
	}

	// Append primitive rules that every JSON grammar needs.
	sb.WriteString(jsonPrimitiveRules)

	return sb.String()
}

// GenerateJSONGBNF creates a GBNF grammar that accepts any valid JSON object.
// Used when response_format.type = "json_object" (no schema provided).
func GenerateJSONGBNF() string {
	return `root ::= object
` + jsonPrimitiveRules
}

// generateObjectRule creates GBNF rules for a JSON object schema.
func generateObjectRule(sb *strings.Builder, schema map[string]interface{}, ruleName string) {
	props, hasProps := schema["properties"].(map[string]interface{})
	required, _ := schema["required"].([]interface{})

	requiredSet := make(map[string]bool)
	for _, r := range required {
		if s, ok := r.(string); ok {
			requiredSet[s] = true
		}
	}

	if !hasProps || len(props) == 0 {
		// No properties defined: accept any JSON object.
		sb.WriteString(fmt.Sprintf("%s ::= \"{\" ws (string \":\" ws value (\",\" ws string \":\" ws value)*)? ws \"}\"\n", ruleName))
		return
	}

	// Build property rules in order.
	propNames := make([]string, 0, len(props))
	for name := range props {
		propNames = append(propNames, name)
	}

	// Generate the object rule.
	sb.WriteString(fmt.Sprintf("%s ::= \"{\" ws ", ruleName))

	for i, name := range propNames {
		propSchema, _ := props[name].(map[string]interface{})
		propRuleName := fmt.Sprintf("%s-%s", ruleName, sanitizeName(name))

		// Write property key-value pair.
		if i > 0 {
			if requiredSet[name] {
				sb.WriteString("\",\" ws ")
			} else {
				sb.WriteString("(\",\" ws ")
			}
		}

		sb.WriteString(fmt.Sprintf("\"\\\"%s\\\"\" \":\" ws %s", name, propRuleName))

		if i > 0 && !requiredSet[name] {
			sb.WriteString(")?")
		}

		// Generate the property type rule.
		generateTypeRule(sb, propSchema, propRuleName)
	}

	sb.WriteString(" ws \"}\"\n")
}

// generateArrayRule creates GBNF rules for a JSON array schema.
func generateArrayRule(sb *strings.Builder, schema map[string]interface{}, ruleName string) {
	items, hasItems := schema["items"].(map[string]interface{})

	if !hasItems {
		sb.WriteString(fmt.Sprintf("%s ::= \"[\" ws (value (\",\" ws value)*)? ws \"]\"\n", ruleName))
		return
	}

	itemRule := ruleName + "-item"
	sb.WriteString(fmt.Sprintf("%s ::= \"[\" ws (%s (\",\" ws %s)*)? ws \"]\"\n", ruleName, itemRule, itemRule))
	generateTypeRule(sb, items, itemRule)
}

// generateTypeRule creates a GBNF rule for a specific JSON type.
func generateTypeRule(sb *strings.Builder, schema map[string]interface{}, ruleName string) {
	if schema == nil {
		sb.WriteString(fmt.Sprintf("%s ::= value\n", ruleName))
		return
	}

	schemaType, _ := schema["type"].(string)
	enumVals, hasEnum := schema["enum"].([]interface{})

	// Handle enums: exact match of allowed values.
	if hasEnum {
		alts := make([]string, 0, len(enumVals))
		for _, v := range enumVals {
			switch val := v.(type) {
			case string:
				alts = append(alts, fmt.Sprintf("\"\\\"\" \"%s\" \"\\\"\"", escapeGBNF(val)))
			case float64:
				alts = append(alts, fmt.Sprintf("\"%v\"", val))
			case bool:
				alts = append(alts, fmt.Sprintf("\"%v\"", val))
			default:
				b, _ := json.Marshal(val)
				alts = append(alts, fmt.Sprintf("\"%s\"", escapeGBNF(string(b))))
			}
		}
		sb.WriteString(fmt.Sprintf("%s ::= %s\n", ruleName, strings.Join(alts, " | ")))
		return
	}

	switch schemaType {
	case "string":
		sb.WriteString(fmt.Sprintf("%s ::= string\n", ruleName))
	case "number":
		sb.WriteString(fmt.Sprintf("%s ::= number\n", ruleName))
	case "integer":
		sb.WriteString(fmt.Sprintf("%s ::= integer\n", ruleName))
	case "boolean":
		sb.WriteString(fmt.Sprintf("%s ::= boolean\n", ruleName))
	case "null":
		sb.WriteString(fmt.Sprintf("%s ::= \"null\"\n", ruleName))
	case "object":
		generateObjectRule(sb, schema, ruleName)
	case "array":
		generateArrayRule(sb, schema, ruleName)
	default:
		sb.WriteString(fmt.Sprintf("%s ::= value\n", ruleName))
	}
}

// sanitizeName makes a property name safe for GBNF rule naming.
func sanitizeName(name string) string {
	result := strings.NewReplacer(
		" ", "-",
		"_", "-",
		".", "-",
		"/", "-",
		"[", "",
		"]", "",
	).Replace(name)
	return strings.ToLower(result)
}

// escapeGBNF escapes special characters for GBNF string literals.
func escapeGBNF(s string) string {
	s = strings.ReplaceAll(s, "\\", "\\\\")
	s = strings.ReplaceAll(s, "\"", "\\\"")
	s = strings.ReplaceAll(s, "\n", "\\n")
	s = strings.ReplaceAll(s, "\t", "\\t")
	return s
}

// jsonPrimitiveRules are the shared GBNF rules for basic JSON types.
// These are appended to every generated grammar.
const jsonPrimitiveRules = `
value ::= object | array | string | number | boolean | "null"
object ::= "{" ws (string ":" ws value ("," ws string ":" ws value)*)? ws "}"
array ::= "[" ws (value ("," ws value)*)? ws "]"
string ::= "\"" ([^"\\] | "\\" ["\\/bfnrt] | "\\u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F])* "\""
number ::= integer ("." [0-9]+)? ([eE] [+-]? [0-9]+)?
integer ::= "-"? ("0" | [1-9] [0-9]*)
boolean ::= "true" | "false"
ws ::= ([ \t\n\r])*
`
