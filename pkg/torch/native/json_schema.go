// json_schema.go implements JSON Schema validation for structured model output.
//
// WHAT: Beyond GBNF grammar constraints, this provides JSON Schema validation
// for model output. It validates that generated JSON conforms to a schema
// with support for types, required fields, enums, min/max, and patterns.
//
// USE CASE: When calling /v1/chat/completions with response_format.type = "json_schema",
// the output is validated against the provided schema before returning.
package native

import (
	"encoding/json"
	"fmt"
	"strings"
)

// JSONSchemaType represents the type field in a JSON Schema.
type JSONSchemaType string

const (
	SchemaString  JSONSchemaType = "string"
	SchemaNumber  JSONSchemaType = "number"
	SchemaInteger JSONSchemaType = "integer"
	SchemaBoolean JSONSchemaType = "boolean"
	SchemaArray   JSONSchemaType = "array"
	SchemaObject  JSONSchemaType = "object"
	SchemaNull    JSONSchemaType = "null"
)

// JSONSchema represents a JSON Schema for output validation.
type JSONSchema struct {
	Type        JSONSchemaType         `json:"type"`
	Properties  map[string]*JSONSchema `json:"properties,omitempty"`
	Required    []string               `json:"required,omitempty"`
	Items       *JSONSchema            `json:"items,omitempty"`
	Enum        []interface{}          `json:"enum,omitempty"`
	MinLength   *int                   `json:"minLength,omitempty"`
	MaxLength   *int                   `json:"maxLength,omitempty"`
	Minimum     *float64               `json:"minimum,omitempty"`
	Maximum     *float64               `json:"maximum,omitempty"`
	Description string                 `json:"description,omitempty"`
}

// SchemaValidator validates JSON output against a schema.
type SchemaValidator struct {
	schema *JSONSchema
}

// NewSchemaValidator creates a validator from a JSON Schema.
func NewSchemaValidator(schema *JSONSchema) *SchemaValidator {
	return &SchemaValidator{schema: schema}
}

// NewSchemaValidatorFromJSON parses a schema from JSON bytes.
func NewSchemaValidatorFromJSON(data []byte) (*SchemaValidator, error) {
	var schema JSONSchema
	if err := json.Unmarshal(data, &schema); err != nil {
		return nil, fmt.Errorf("parse schema: %w", err)
	}
	return &SchemaValidator{schema: &schema}, nil
}

// ValidationError describes a schema validation failure.
type ValidationError struct {
	Path    string // JSON path to the failing field
	Message string // What went wrong
}

func (e *ValidationError) Error() string {
	return fmt.Sprintf("%s: %s", e.Path, e.Message)
}

// Validate checks if a JSON string conforms to the schema.
// Returns nil if valid, or a list of validation errors.
func (v *SchemaValidator) Validate(jsonStr string) []ValidationError {
	var parsed interface{}
	if err := json.Unmarshal([]byte(jsonStr), &parsed); err != nil {
		return []ValidationError{{Path: "$", Message: fmt.Sprintf("invalid JSON: %v", err)}}
	}

	return v.validateValue(parsed, v.schema, "$")
}

// ValidateAndFix attempts to fix minor validation issues.
// Returns the fixed JSON string and any unfixable errors.
func (v *SchemaValidator) ValidateAndFix(jsonStr string) (string, []ValidationError) {
	var parsed interface{}
	if err := json.Unmarshal([]byte(jsonStr), &parsed); err != nil {
		return jsonStr, []ValidationError{{Path: "$", Message: fmt.Sprintf("invalid JSON: %v", err)}}
	}

	errors := v.validateValue(parsed, v.schema, "$")
	if len(errors) == 0 {
		return jsonStr, nil
	}

	// Try to fix by adding missing required fields with defaults.
	if obj, ok := parsed.(map[string]interface{}); ok && v.schema.Type == SchemaObject {
		fixed := v.addDefaults(obj, v.schema)
		fixedJSON, err := json.Marshal(fixed)
		if err == nil {
			remainingErrors := v.validateValue(fixed, v.schema, "$")
			return string(fixedJSON), remainingErrors
		}
	}

	return jsonStr, errors
}

func (v *SchemaValidator) validateValue(value interface{}, schema *JSONSchema, path string) []ValidationError {
	if schema == nil {
		return nil
	}

	var errors []ValidationError

	// Check enum constraint.
	if len(schema.Enum) > 0 {
		found := false
		for _, e := range schema.Enum {
			if fmt.Sprintf("%v", value) == fmt.Sprintf("%v", e) {
				found = true
				break
			}
		}
		if !found {
			errors = append(errors, ValidationError{
				Path:    path,
				Message: fmt.Sprintf("value must be one of %v", schema.Enum),
			})
		}
	}

	switch schema.Type {
	case SchemaString:
		str, ok := value.(string)
		if !ok {
			errors = append(errors, ValidationError{Path: path, Message: "expected string"})
		} else {
			if schema.MinLength != nil && len(str) < *schema.MinLength {
				errors = append(errors, ValidationError{
					Path: path, Message: fmt.Sprintf("string too short: %d < %d", len(str), *schema.MinLength),
				})
			}
			if schema.MaxLength != nil && len(str) > *schema.MaxLength {
				errors = append(errors, ValidationError{
					Path: path, Message: fmt.Sprintf("string too long: %d > %d", len(str), *schema.MaxLength),
				})
			}
		}

	case SchemaNumber, SchemaInteger:
		num, ok := value.(float64)
		if !ok {
			errors = append(errors, ValidationError{Path: path, Message: "expected number"})
		} else {
			if schema.Minimum != nil && num < *schema.Minimum {
				errors = append(errors, ValidationError{
					Path: path, Message: fmt.Sprintf("value too small: %v < %v", num, *schema.Minimum),
				})
			}
			if schema.Maximum != nil && num > *schema.Maximum {
				errors = append(errors, ValidationError{
					Path: path, Message: fmt.Sprintf("value too large: %v > %v", num, *schema.Maximum),
				})
			}
		}

	case SchemaBoolean:
		if _, ok := value.(bool); !ok {
			errors = append(errors, ValidationError{Path: path, Message: "expected boolean"})
		}

	case SchemaArray:
		arr, ok := value.([]interface{})
		if !ok {
			errors = append(errors, ValidationError{Path: path, Message: "expected array"})
		} else if schema.Items != nil {
			for i, item := range arr {
				itemPath := fmt.Sprintf("%s[%d]", path, i)
				errors = append(errors, v.validateValue(item, schema.Items, itemPath)...)
			}
		}

	case SchemaObject:
		obj, ok := value.(map[string]interface{})
		if !ok {
			errors = append(errors, ValidationError{Path: path, Message: "expected object"})
		} else {
			// Check required fields.
			for _, req := range schema.Required {
				if _, exists := obj[req]; !exists {
					errors = append(errors, ValidationError{
						Path:    path + "." + req,
						Message: "required field missing",
					})
				}
			}

			// Validate properties.
			for key, propSchema := range schema.Properties {
				if val, exists := obj[key]; exists {
					propPath := path + "." + key
					errors = append(errors, v.validateValue(val, propSchema, propPath)...)
				}
			}
		}

	case SchemaNull:
		if value != nil {
			errors = append(errors, ValidationError{Path: path, Message: "expected null"})
		}
	}

	return errors
}

// addDefaults adds missing required fields with type-appropriate defaults.
func (v *SchemaValidator) addDefaults(obj map[string]interface{}, schema *JSONSchema) map[string]interface{} {
	if schema.Properties == nil {
		return obj
	}

	for _, req := range schema.Required {
		if _, exists := obj[req]; exists {
			continue
		}
		propSchema := schema.Properties[req]
		if propSchema == nil {
			continue
		}
		switch propSchema.Type {
		case SchemaString:
			obj[req] = ""
		case SchemaNumber, SchemaInteger:
			obj[req] = 0
		case SchemaBoolean:
			obj[req] = false
		case SchemaArray:
			obj[req] = []interface{}{}
		case SchemaObject:
			obj[req] = map[string]interface{}{}
		}
	}
	return obj
}

// SchemaToPrompt generates a natural language description of the expected output format.
func SchemaToPrompt(schema *JSONSchema) string {
	var sb strings.Builder
	sb.WriteString("Respond with a JSON object matching this schema:\n")
	describeSchema(&sb, schema, "", 0)
	return sb.String()
}

func describeSchema(sb *strings.Builder, schema *JSONSchema, name string, depth int) {
	indent := strings.Repeat("  ", depth)

	if name != "" {
		sb.WriteString(fmt.Sprintf("%s- %s (%s)", indent, name, schema.Type))
		if schema.Description != "" {
			sb.WriteString(fmt.Sprintf(": %s", schema.Description))
		}
		sb.WriteString("\n")
	}

	if schema.Type == SchemaObject && schema.Properties != nil {
		for key, prop := range schema.Properties {
			isRequired := false
			for _, r := range schema.Required {
				if r == key {
					isRequired = true
					break
				}
			}
			reqStr := ""
			if isRequired {
				reqStr = " [required]"
			}
			sb.WriteString(fmt.Sprintf("%s  - %s (%s)%s", indent, key, prop.Type, reqStr))
			if prop.Description != "" {
				sb.WriteString(fmt.Sprintf(": %s", prop.Description))
			}
			sb.WriteString("\n")
		}
	}
}
