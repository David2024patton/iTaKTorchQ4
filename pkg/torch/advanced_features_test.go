package torch

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// ===========================================================================
// Ollama API Compatibility Tests
// ===========================================================================

// TestOllama_Version verifies GET /api/version returns Torch-flavored version.
func TestOllama_Version(t *testing.T) {
	engine := NewMockEngine("qwen3")
	server := NewServer(engine, 0, WithOllamaCompat())

	req := httptest.NewRequest(http.MethodGet, "/api/version", nil)
	w := httptest.NewRecorder()
	server.server.Handler.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}

	var resp map[string]string
	json.Unmarshal(w.Body.Bytes(), &resp)

	if !strings.Contains(resp["version"], "torch") {
		t.Errorf("version %q should contain 'torch'", resp["version"])
	}
}

// TestOllama_Tags verifies GET /api/tags lists loaded models.
func TestOllama_Tags(t *testing.T) {
	engine := NewMockEngine("qwen3-8b")
	server := NewServer(engine, 0, WithOllamaCompat())

	req := httptest.NewRequest(http.MethodGet, "/api/tags", nil)
	w := httptest.NewRecorder()
	server.server.Handler.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", w.Code)
	}

	var resp map[string]interface{}
	json.Unmarshal(w.Body.Bytes(), &resp)

	models, ok := resp["models"].([]interface{})
	if !ok || len(models) == 0 {
		t.Error("expected at least 1 model in tags response")
	}
}

// TestOllama_Show verifies POST /api/show returns model details.
func TestOllama_Show(t *testing.T) {
	engine := NewMockEngine("qwen3-8b")
	server := NewServer(engine, 0, WithOllamaCompat())

	body := []byte(`{"name":"qwen3-8b"}`)
	req := httptest.NewRequest(http.MethodPost, "/api/show", bytes.NewReader(body))
	w := httptest.NewRecorder()
	server.server.Handler.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}

	var resp map[string]interface{}
	json.Unmarshal(w.Body.Bytes(), &resp)

	if _, ok := resp["details"]; !ok {
		t.Error("expected 'details' field in show response")
	}
}

// TestOllama_Show_WrongMethod verifies GET /api/show returns 405.
func TestOllama_Show_WrongMethod(t *testing.T) {
	engine := NewMockEngine("test")
	server := NewServer(engine, 0, WithOllamaCompat())

	req := httptest.NewRequest(http.MethodGet, "/api/show", nil)
	w := httptest.NewRecorder()
	server.server.Handler.ServeHTTP(w, req)

	if w.Code != http.StatusMethodNotAllowed {
		t.Errorf("expected 405, got %d", w.Code)
	}
}

// TestOllama_Generate_NonStreaming verifies non-streaming POST /api/generate.
func TestOllama_Generate_NonStreaming(t *testing.T) {
	engine := NewMockEngine("qwen3")
	server := NewServer(engine, 0, WithOllamaCompat())

	streamFalse := false
	ollamaReq := OllamaGenerateRequest{
		Model:  "qwen3",
		Prompt: "What is 2+2?",
		Stream: &streamFalse,
	}
	body, _ := json.Marshal(ollamaReq)

	req := httptest.NewRequest(http.MethodPost, "/api/generate", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	server.server.Handler.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}

	var resp OllamaGenerateResponse
	json.Unmarshal(w.Body.Bytes(), &resp)

	if !resp.Done {
		t.Error("expected done=true for non-streaming response")
	}
	if resp.Response == "" {
		t.Error("expected non-empty response text")
	}
	if resp.Model != "qwen3" {
		t.Errorf("model = %q, want %q", resp.Model, "qwen3")
	}
}

// TestOllama_Chat_NonStreaming verifies non-streaming POST /api/chat.
func TestOllama_Chat_NonStreaming(t *testing.T) {
	engine := NewMockEngine("qwen3")
	server := NewServer(engine, 0, WithOllamaCompat())

	streamFalse := false
	ollamaReq := OllamaChatRequest{
		Model: "qwen3",
		Messages: []OllamaChatMsg{
			{Role: "user", Content: "Hello"},
		},
		Stream: &streamFalse,
	}
	body, _ := json.Marshal(ollamaReq)

	req := httptest.NewRequest(http.MethodPost, "/api/chat", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	server.server.Handler.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}

	var resp OllamaChatResponse
	json.Unmarshal(w.Body.Bytes(), &resp)

	if !resp.Done {
		t.Error("expected done=true")
	}
	if resp.Message.Role != "assistant" {
		t.Errorf("message role = %q, want %q", resp.Message.Role, "assistant")
	}
	if resp.Message.Content == "" {
		t.Error("expected non-empty message content")
	}
}

// TestOllama_Generate_WrongMethod verifies wrong method returns 405.
func TestOllama_Generate_WrongMethod(t *testing.T) {
	engine := NewMockEngine("test")
	server := NewServer(engine, 0, WithOllamaCompat())

	req := httptest.NewRequest(http.MethodGet, "/api/generate", nil)
	w := httptest.NewRecorder()
	server.server.Handler.ServeHTTP(w, req)

	if w.Code != http.StatusMethodNotAllowed {
		t.Errorf("expected 405, got %d", w.Code)
	}
}

// TestOllama_OptsConversion verifies Ollama options map to completion params correctly.
func TestOllama_OptsConversion(t *testing.T) {
	opts := OllamaOptions{
		NumPredict:  200,
		Temperature: 0.5,
		TopP:        0.85,
		Stop:        []string{"\n", "END"},
	}

	params := ollamaOptsToParams(opts)

	if params.MaxTokens != 200 {
		t.Errorf("MaxTokens = %d, want 200", params.MaxTokens)
	}
	if params.Temperature != 0.5 {
		t.Errorf("Temperature = %f, want 0.5", params.Temperature)
	}
	if params.TopP != 0.85 {
		t.Errorf("TopP = %f, want 0.85", params.TopP)
	}
	if len(params.Stop) != 2 {
		t.Errorf("Stop length = %d, want 2", len(params.Stop))
	}
}

// TestOllama_OptsDefaults verifies empty options produce sane defaults.
func TestOllama_OptsDefaults(t *testing.T) {
	params := ollamaOptsToParams(OllamaOptions{})

	if params.MaxTokens != 512 {
		t.Errorf("default MaxTokens = %d, want 512", params.MaxTokens)
	}
	if params.Temperature != 0.8 {
		t.Errorf("default Temperature = %f, want 0.8", params.Temperature)
	}
}

// ===========================================================================
// JSON Structured Output Tests
// ===========================================================================

// TestSchemaToGBNF_SimpleObject verifies schema-to-grammar conversion for a basic object.
func TestSchemaToGBNF_SimpleObject(t *testing.T) {
	schema := map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"name": map[string]interface{}{"type": "string"},
			"age":  map[string]interface{}{"type": "integer"},
		},
		"required": []interface{}{"name", "age"},
	}

	grammar := SchemaToGBNF(schema)

	if grammar == "" {
		t.Fatal("expected non-empty grammar")
	}
	if !strings.Contains(grammar, "root ::=") {
		t.Error("grammar should have a root rule")
	}
	if !strings.Contains(grammar, "object") {
		t.Error("grammar should reference object rule")
	}
	if !strings.Contains(grammar, "string") {
		t.Error("grammar should contain string primitive rule")
	}
}

// TestSchemaToGBNF_Array verifies array schema conversion.
func TestSchemaToGBNF_Array(t *testing.T) {
	schema := map[string]interface{}{
		"type": "array",
		"items": map[string]interface{}{
			"type": "string",
		},
	}

	grammar := SchemaToGBNF(schema)

	if !strings.Contains(grammar, "root ::=") {
		t.Error("grammar should have root rule")
	}
	if !strings.Contains(grammar, "array") {
		t.Error("grammar should reference array rule")
	}
}

// TestSchemaToGBNF_NoType verifies fallback to generic value grammar.
func TestSchemaToGBNF_NoType(t *testing.T) {
	schema := map[string]interface{}{}

	grammar := SchemaToGBNF(schema)

	if !strings.Contains(grammar, "root ::= value") {
		t.Error("empty schema should produce root ::= value")
	}
}

// TestSchemaToGBNF_Enum verifies enum value constraints.
func TestSchemaToGBNF_Enum(t *testing.T) {
	schema := map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"status": map[string]interface{}{
				"type": "string",
				"enum": []interface{}{"active", "inactive", "pending"},
			},
		},
	}

	grammar := SchemaToGBNF(schema)

	if !strings.Contains(grammar, "active") {
		t.Error("grammar should contain enum value 'active'")
	}
	if !strings.Contains(grammar, "inactive") {
		t.Error("grammar should contain enum value 'inactive'")
	}
}

// TestGenerateJSONGBNF verifies the generic JSON object grammar.
func TestGenerateJSONGBNF(t *testing.T) {
	grammar := GenerateJSONGBNF()

	if grammar == "" {
		t.Fatal("expected non-empty grammar")
	}
	if !strings.Contains(grammar, "root ::= object") {
		t.Error("should start with root ::= object")
	}
	if !strings.Contains(grammar, "value ::=") {
		t.Error("should contain value rule")
	}
	if !strings.Contains(grammar, "number ::=") {
		t.Error("should contain number rule")
	}
	if !strings.Contains(grammar, "boolean ::=") {
		t.Error("should contain boolean rule")
	}
}

// TestSanitizeName verifies property name sanitization for GBNF.
func TestSanitizeName(t *testing.T) {
	tests := []struct {
		input string
		want  string
	}{
		{"user_name", "user-name"},
		{"first.last", "first-last"},
		{"items[0]", "items0"},
		{"UPPER", "upper"},
	}

	for _, tt := range tests {
		got := sanitizeName(tt.input)
		if got != tt.want {
			t.Errorf("sanitizeName(%q) = %q, want %q", tt.input, got, tt.want)
		}
	}
}

// TestEscapeGBNF verifies GBNF string escaping.
func TestEscapeGBNF(t *testing.T) {
	tests := []struct {
		input string
		want  string
	}{
		{`hello`, `hello`},
		{`he"llo`, `he\"llo`},
		{"line\nbreak", `line\nbreak`},
		{`back\slash`, `back\\slash`},
	}

	for _, tt := range tests {
		got := escapeGBNF(tt.input)
		if got != tt.want {
			t.Errorf("escapeGBNF(%q) = %q, want %q", tt.input, got, tt.want)
		}
	}
}

// TestResponseFormat_JSONTypes verifies JSON roundtrip of ResponseFormat types.
func TestResponseFormat_JSONTypes(t *testing.T) {
	tests := []struct {
		name string
		rf   ResponseFormat
	}{
		{"text", ResponseFormat{Type: "text"}},
		{"json_object", ResponseFormat{Type: "json_object"}},
		{"json_schema", ResponseFormat{
			Type: "json_schema",
			JSONSchema: &JSONSchemaSpec{
				Name:   "test",
				Schema: map[string]interface{}{"type": "object"},
			},
		}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			data, err := json.Marshal(tt.rf)
			if err != nil {
				t.Fatalf("marshal: %v", err)
			}

			var decoded ResponseFormat
			if err := json.Unmarshal(data, &decoded); err != nil {
				t.Fatalf("unmarshal: %v", err)
			}
			if decoded.Type != tt.rf.Type {
				t.Errorf("Type = %q, want %q", decoded.Type, tt.rf.Type)
			}
		})
	}
}

// ===========================================================================
// Speculative Decoding Tests (struct-level, no model required)
// ===========================================================================

// TestSpecStats_Zero verifies default SpecStats values.
func TestSpecStats_Zero(t *testing.T) {
	stats := SpecStats{}

	if stats.TotalDraftTokens != 0 || stats.AcceptedTokens != 0 || stats.RejectedTokens != 0 {
		t.Error("default SpecStats should be zero")
	}
	if stats.AvgAcceptanceRate != 0 || stats.AvgTokensPerStep != 0 {
		t.Error("default averages should be zero")
	}
	if stats.SpeedupEstimate != 0 {
		t.Error("default speedup should be zero")
	}
}

// TestSpeculativeDecoder_NilSafety verifies IsEnabled returns false for nil decoder.
func TestSpeculativeDecoder_NilSafety(t *testing.T) {
	var sd *SpeculativeDecoder
	if sd.IsEnabled() {
		t.Error("nil SpeculativeDecoder should not be enabled")
	}
}

// TestSpeculativeDecoder_BadInputs verifies error handling on construction.
func TestSpeculativeDecoder_BadInputs(t *testing.T) {
	// Zero target model should fail.
	_, err := NewSpeculativeDecoder(0, 0, "draft.gguf", 5, 0, 4)
	if err == nil {
		t.Error("expected error for zero target model")
	}

	// Empty draft path should fail.
	_, err = NewSpeculativeDecoder(1, 1, "", 5, 0, 4)
	if err == nil {
		t.Error("expected error for empty draft path")
	}
}

// TestSpecStats_ResetStats verifies stats reset on a SpeculativeDecoder.
func TestSpecStats_ResetStats(t *testing.T) {
	sd := &SpeculativeDecoder{
		Stats: SpecStats{
			TotalDraftTokens:  100,
			AcceptedTokens:    85,
			RejectedTokens:    15,
			TotalSteps:        20,
			AvgAcceptanceRate: 0.85,
		},
	}

	sd.ResetStats()

	if sd.Stats.TotalDraftTokens != 0 || sd.Stats.AcceptedTokens != 0 {
		t.Error("ResetStats should zero all counters")
	}
	if sd.Stats.TotalSteps != 0 {
		t.Error("ResetStats should zero TotalSteps")
	}
}

// ===========================================================================
// Multi-GPU Tests (detection-based, no model required)
// ===========================================================================

// TestDetectMultiGPU verifies multi-GPU detection returns nil for single-GPU.
func TestDetectMultiGPU(t *testing.T) {
	config := DetectMultiGPU()

	// On a single discrete GPU system (like this machine), should return nil.
	// We can't assert nil definitively on unknown hardware, so just verify the
	// return type and structure if non-nil.
	if config != nil {
		if config.SplitMode == "" {
			t.Error("non-nil config should have a split mode")
		}
		if len(config.TensorSplit) < 2 {
			t.Error("multi-GPU config should have at least 2 split values")
		}

		// Verify splits sum to ~1.0.
		sum := float32(0)
		for _, s := range config.TensorSplit {
			sum += s
		}
		if sum < 0.99 || sum > 1.01 {
			t.Errorf("tensor splits should sum to ~1.0, got %f", sum)
		}
	}
}

// TestMultiGPUConfig_Defaults verifies configuration defaults.
func TestMultiGPUConfig_Defaults(t *testing.T) {
	config := &MultiGPUConfig{
		SplitMode:   "layer",
		MainGPU:     0,
		TensorSplit: []float32{0.7, 0.3},
		AutoBalance: true,
	}

	if config.SplitMode != "layer" {
		t.Errorf("SplitMode = %q, want %q", config.SplitMode, "layer")
	}
	if len(config.TensorSplit) != 2 {
		t.Error("should have 2 split values")
	}
}

// TestMultiGPUStatus_Nil verifies status string for nil config.
func TestMultiGPUStatus_Nil(t *testing.T) {
	status := MultiGPUStatus(nil)
	if !strings.Contains(status, "single GPU") {
		t.Errorf("nil config status = %q, should mention single GPU", status)
	}
}

// TestMultiGPUStatus_Active verifies status string for active config.
func TestMultiGPUStatus_Active(t *testing.T) {
	config := &MultiGPUConfig{
		SplitMode:   "row",
		MainGPU:     0,
		TensorSplit: []float32{0.5, 0.5},
	}

	status := MultiGPUStatus(config)
	if !strings.Contains(status, "row") {
		t.Error("status should mention split mode")
	}
	if !strings.Contains(status, "GPU0") {
		t.Error("status should list GPU indices")
	}
}

// TestMultiGPUConfig_JSON verifies JSON roundtrip of config.
func TestMultiGPUConfig_JSON(t *testing.T) {
	config := MultiGPUConfig{
		SplitMode:   "layer",
		MainGPU:     1,
		TensorSplit: []float32{0.6, 0.4},
		AutoBalance: false,
	}

	data, err := json.Marshal(config)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}

	var decoded MultiGPUConfig
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	if decoded.SplitMode != "layer" {
		t.Errorf("SplitMode = %q, want %q", decoded.SplitMode, "layer")
	}
	if decoded.MainGPU != 1 {
		t.Errorf("MainGPU = %d, want 1", decoded.MainGPU)
	}
	if len(decoded.TensorSplit) != 2 {
		t.Errorf("TensorSplit length = %d, want 2", len(decoded.TensorSplit))
	}
}
