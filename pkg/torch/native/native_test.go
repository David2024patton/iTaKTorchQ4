package native

import (
	"context"
	"math"
	"testing"
)

// ===========================================================================
// Tensor Basics
// ===========================================================================

func TestTensor_NewAndGetSet(t *testing.T) {
	tensor := NewTensor([]int{2, 3})

	if tensor.Size() != 6 {
		t.Fatalf("Size() = %d, want 6", tensor.Size())
	}

	tensor.Set([]int{0, 1}, 3.14)
	got := tensor.Get([]int{0, 1})
	if got != 3.14 {
		t.Errorf("Get([0,1]) = %f, want 3.14", got)
	}

	// Verify other elements are still zero.
	if tensor.Get([]int{0, 0}) != 0 {
		t.Error("Get([0,0]) should be 0")
	}
}

func TestTensor_Add(t *testing.T) {
	a := NewTensorFrom([]int{3}, []float32{1, 2, 3})
	b := NewTensorFrom([]int{3}, []float32{4, 5, 6})
	c := Add(a, b)

	expected := []float32{5, 7, 9}
	for i, want := range expected {
		if c.Data[i] != want {
			t.Errorf("Add()[%d] = %f, want %f", i, c.Data[i], want)
		}
	}
}

func TestTensor_Mul(t *testing.T) {
	a := NewTensorFrom([]int{3}, []float32{2, 3, 4})
	b := NewTensorFrom([]int{3}, []float32{5, 6, 7})
	c := Mul(a, b)

	expected := []float32{10, 18, 28}
	for i, want := range expected {
		if c.Data[i] != want {
			t.Errorf("Mul()[%d] = %f, want %f", i, c.Data[i], want)
		}
	}
}

func TestTensor_Scale(t *testing.T) {
	a := NewTensorFrom([]int{3}, []float32{1, 2, 3})
	b := Scale(a, 2.0)

	expected := []float32{2, 4, 6}
	for i, want := range expected {
		if b.Data[i] != want {
			t.Errorf("Scale()[%d] = %f, want %f", i, b.Data[i], want)
		}
	}
}

// ===========================================================================
// Activation Functions
// ===========================================================================

func TestReLU(t *testing.T) {
	input := NewTensorFrom([]int{5}, []float32{-3, -1, 0, 1, 3})
	output := ReLU(input)

	expected := []float32{0, 0, 0, 1, 3}
	for i, want := range expected {
		if output.Data[i] != want {
			t.Errorf("ReLU()[%d] = %f, want %f", i, output.Data[i], want)
		}
	}
}

func TestSoftmax(t *testing.T) {
	input := NewTensorFrom([]int{3}, []float32{1, 2, 3})
	output := Softmax(input)

	// Softmax values should sum to ~1.0.
	var sum float32
	for _, v := range output.Data {
		sum += v
	}
	if math.Abs(float64(sum-1.0)) > 0.001 {
		t.Errorf("Softmax sum = %f, want ~1.0", sum)
	}

	// Largest input should have largest softmax output.
	if output.Data[2] <= output.Data[1] || output.Data[1] <= output.Data[0] {
		t.Errorf("Softmax should preserve relative order: %v", output.Data)
	}
}

func TestRMSNorm(t *testing.T) {
	input := NewTensorFrom([]int{1, 4}, []float32{2, 4, 6, 8})
	weight := NewTensorFrom([]int{4}, []float32{1, 1, 1, 1})
	output := RMSNorm(input, weight, 1e-6)

	// Output should have roughly unit RMS (since weight is all ones).
	var sumSq float32
	for _, v := range output.Data {
		sumSq += v * v
	}
	rms := math.Sqrt(float64(sumSq / 4.0))
	if math.Abs(rms-1.0) > 0.01 {
		t.Errorf("RMSNorm: output RMS = %f, expected ~1.0", rms)
	}
}

// ===========================================================================
// Matrix Multiplication
// ===========================================================================

func TestMatMul_Identity(t *testing.T) {
	// Multiplying by identity should return the original matrix.
	a := NewTensorFrom([]int{2, 2}, []float32{1, 2, 3, 4})
	identity := NewTensorFrom([]int{2, 2}, []float32{1, 0, 0, 1})
	result := MatMul(a, identity)

	expected := []float32{1, 2, 3, 4}
	for i, want := range expected {
		if result.Data[i] != want {
			t.Errorf("MatMul(A, I)[%d] = %f, want %f", i, result.Data[i], want)
		}
	}
}

func TestMatMul_Basic(t *testing.T) {
	a := NewTensorFrom([]int{2, 3}, []float32{1, 2, 3, 4, 5, 6})
	b := NewTensorFrom([]int{3, 2}, []float32{7, 8, 9, 10, 11, 12})
	result := MatMul(a, b)

	// [2,3] x [3,2] = [2,2]
	if result.Shape[0] != 2 || result.Shape[1] != 2 {
		t.Fatalf("shape = %v, want [2,2]", result.Shape)
	}

	// Manual: row 0 = [1*7+2*9+3*11, 1*8+2*10+3*12] = [58, 64]
	// Manual: row 1 = [4*7+5*9+6*11, 4*8+5*10+6*12] = [139, 154]
	expected := []float32{58, 64, 139, 154}
	for i, want := range expected {
		if result.Data[i] != want {
			t.Errorf("MatMul[%d] = %f, want %f", i, result.Data[i], want)
		}
	}
}

func TestMatVecMul(t *testing.T) {
	a := NewTensorFrom([]int{2, 3}, []float32{1, 2, 3, 4, 5, 6})
	v := NewTensorFrom([]int{3}, []float32{1, 1, 1})
	result := MatVecMul(a, v)

	// Row sums: [1+2+3, 4+5+6] = [6, 15]
	expected := []float32{6, 15}
	for i, want := range expected {
		if result.Data[i] != want {
			t.Errorf("MatVecMul[%d] = %f, want %f", i, result.Data[i], want)
		}
	}
}

// ===========================================================================
// Attention
// ===========================================================================

func TestCausalMask(t *testing.T) {
	mask := CausalMask(3)

	// Row 0: F T T (can see only position 0)
	if mask[0] {
		t.Error("mask[0,0] should be false (can see self)")
	}
	if !mask[1] {
		t.Error("mask[0,1] should be true (can't see future)")
	}
	if !mask[2] {
		t.Error("mask[0,2] should be true (can't see future)")
	}

	// Row 2: F F F (can see all past positions)
	if mask[6] || mask[7] || mask[8] {
		t.Error("row 2 should have no masking")
	}
}

func TestAttention_OutputShape(t *testing.T) {
	seqLen := 4
	headDim := 8
	q := NewTensor([]int{seqLen, headDim})
	k := NewTensor([]int{seqLen, headDim})
	v := NewTensor([]int{seqLen, headDim})
	mask := CausalMask(seqLen)

	output := Attention(q, k, v, mask)

	if output.Shape[0] != seqLen || output.Shape[1] != headDim {
		t.Errorf("Attention output shape = %v, want [%d, %d]", output.Shape, seqLen, headDim)
	}
}

// ===========================================================================
// Native Engine (End-to-End)
// ===========================================================================

func TestNativeEngine_Complete(t *testing.T) {
	// Tiny model: vocab=100, hidden=32, 2 heads, 1 layer.
	engine := NewNativeEngine("test-tiny", 100, 32, 2, 1)

	messages := []ChatMessage{
		{Role: "user", Content: "hello"},
	}
	result, err := engine.Complete(context.Background(), messages, CompletionParams{MaxTokens: 5})
	if err != nil {
		t.Fatalf("Complete error: %v", err)
	}
	if result == "" {
		t.Error("expected non-empty result")
	}
	if len(result) != 5 {
		t.Errorf("expected 5 characters (5 tokens), got %d", len(result))
	}
	t.Logf("Generated: %q", result)
}

func TestNativeEngine_Stats(t *testing.T) {
	engine := NewNativeEngine("stats-test", 50, 16, 2, 1)

	messages := []ChatMessage{{Role: "user", Content: "test"}}
	engine.Complete(context.Background(), messages, CompletionParams{MaxTokens: 3})
	engine.Complete(context.Background(), messages, CompletionParams{MaxTokens: 3})

	stats := engine.GetStats()
	if stats.TotalRequests != 2 {
		t.Errorf("TotalRequests = %d, want 2", stats.TotalRequests)
	}
	if stats.TotalTokensGen != 6 {
		t.Errorf("TotalTokensGen = %d, want 6", stats.TotalTokensGen)
	}
	if stats.LastMetrics == nil {
		t.Fatal("LastMetrics should not be nil")
	}
	if stats.LastMetrics.CompletionTokens != 3 {
		t.Errorf("expected 3 completion tokens, got %d", stats.LastMetrics.CompletionTokens)
	}
	t.Logf("Tokens/sec: %.1f (may be 0 or Inf on fast CPUs)", stats.LastMetrics.TokensPerSecond)
}

func TestNativeEngine_Close(t *testing.T) {
	engine := NewNativeEngine("close-test", 50, 16, 2, 1)
	engine.Close()

	_, err := engine.Complete(context.Background(), []ChatMessage{{Role: "user", Content: "test"}}, CompletionParams{MaxTokens: 1})
	if err == nil {
		t.Error("expected error after Close()")
	}
}

func TestNativeEngine_ModelName(t *testing.T) {
	engine := NewNativeEngine("my-engine", 50, 16, 2, 1)
	if engine.ModelName() != "my-engine" {
		t.Errorf("ModelName() = %q, want %q", engine.ModelName(), "my-engine")
	}
}
