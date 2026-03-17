package native

import (
	"context"
	"strings"
	"testing"
)

// TestSpeculativeDecoder logic verifies that drafting and verifying
// produces exactly the same output as running the target model directly,
// just doing it faster under the hood.
func TestSpeculativeDecoder_Complete(t *testing.T) {
	// 1. Create a "Large" Target model.
	// We use the NewNativeEngine which auto-initializes with random weights.
	targetModel := NewNativeEngine("Target-8B", 64, 128, 4, 2)

	// 2. Create a "Small" Draft model.
	// Since weights are randomly initialized based on seeds, if we use the same size,
	// they will have identical weights and predict the exact same things, making the draft 100% accurate.
	// To test rejection, we make the draft model slightly different (different dimensions)
	// so it gets some tokens wrong and forces a rejection.
	draftModel := NewNativeEngine("Draft-1.5B", 64, 64, 2, 1)

	decoder := NewSpeculativeDecoder(targetModel, draftModel)

	if !strings.Contains(decoder.ModelName(), "Target-8B") || !strings.Contains(decoder.ModelName(), "Draft-1.5B") {
		t.Errorf("ModelName didn't contain expected strings: %s", decoder.ModelName())
	}

	messages := []ChatMessage{
		{Role: "user", Content: "Hello world"},
	}

	params := CompletionParams{MaxTokens: 10}

	// 3. Run speculative decoding
	ctx := context.Background()
	result, err := decoder.Complete(ctx, messages, params)
	if err != nil {
		t.Fatalf("Speculative decoder failed: %v", err)
	}

	if result == "" {
		t.Error("Speculative decoder returned empty string")
	}

	stats := decoder.GetStats()
	if stats.TotalRequests != 1 {
		t.Errorf("Expected 1 request, got %d", stats.TotalRequests)
	}
	if stats.LastMetrics == nil {
		t.Fatal("Expected metrics, got nil")
	}

	if stats.LastMetrics.CompletionTokens != 10 {
		t.Errorf("Expected 10 completion tokens, got %d", stats.LastMetrics.CompletionTokens)
	}

	// 4. Run the target model alone (the ground truth)
	// We need a fresh target model because running updates its internal state if we had state (we don't here,
	// but it's good practice). Actually, NativeEngine is stateless between requests so we can reuse targetModel.
	// BUT we need to isolate the start point.
	groundTruthModel := NewNativeEngine("Target-8B", 64, 128, 4, 2)
	truthResult, err := groundTruthModel.Complete(ctx, messages, params)
	if err != nil {
		t.Fatalf("Target model failed: %v", err)
	}

	// 5. Assert exact match!
	// Speculative decoding MUST be mathematically identical to the target model in greedy mode.
	if result != truthResult {
		t.Errorf("Speculative output mismatch!\nExpected: %q\nGot:      %q", truthResult, result)
	}
}
