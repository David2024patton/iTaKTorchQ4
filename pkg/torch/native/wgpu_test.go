//go:build wgpu

package native

import (
	"math"
	"testing"
)

// TestWebGPU_MatMul verifies GPU matrix multiplication matches CPU reference.
func TestWebGPU_MatMul(t *testing.T) {
	gpu := NewGPUBackend()
	defer gpu.Close()

	if !gpu.IsAvailable() {
		t.Skip("No WebGPU device available")
	}

	// 4x3 * 3x2 = 4x2
	a := &Tensor{
		Data:  []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
		Shape: []int{4, 3},
	}
	b := &Tensor{
		Data:  []float32{1, 2, 3, 4, 5, 6},
		Shape: []int{3, 2},
	}

	cpuResult := MatMul(a, b)
	gpuResult := gpu.MatMulGPU(a, b)

	if len(cpuResult.Data) != len(gpuResult.Data) {
		t.Fatalf("CPU %d elements, GPU %d elements", len(cpuResult.Data), len(gpuResult.Data))
	}

	tolerance := float32(0.01)
	for i := range cpuResult.Data {
		diff := float32(math.Abs(float64(cpuResult.Data[i] - gpuResult.Data[i])))
		if diff > tolerance {
			t.Errorf("element %d: CPU=%f GPU=%f diff=%f", i, cpuResult.Data[i], gpuResult.Data[i], diff)
		}
	}
}

// TestWebGPU_MatMul_Identity verifies GPU matmul with identity matrix.
func TestWebGPU_MatMul_Identity(t *testing.T) {
	gpu := NewGPUBackend()
	defer gpu.Close()

	if !gpu.IsAvailable() {
		t.Skip("No WebGPU device available")
	}

	// 3x3 identity * 3x3 input = input.
	identity := &Tensor{
		Data:  []float32{1, 0, 0, 0, 1, 0, 0, 0, 1},
		Shape: []int{3, 3},
	}
	input := &Tensor{
		Data:  []float32{1, 2, 3, 4, 5, 6, 7, 8, 9},
		Shape: []int{3, 3},
	}

	result := gpu.MatMulGPU(identity, input)

	for i := range input.Data {
		if result.Data[i] != input.Data[i] {
			t.Errorf("identity matmul: result[%d] = %f, want %f", i, result.Data[i], input.Data[i])
		}
	}
}

// TestWebGPU_FallbackWhenNotAvailable verifies CPU fallback works.
func TestWebGPU_FallbackWhenNotAvailable(t *testing.T) {
	// Create a stub backend (not initialized, so not available).
	gpu := &GPUBackend{available: false}

	a := &Tensor{
		Data:  []float32{1, 2, 3, 4},
		Shape: []int{2, 2},
	}
	b := &Tensor{
		Data:  []float32{5, 6, 7, 8},
		Shape: []int{2, 2},
	}

	// Should fall back to CPU MatMul.
	result := gpu.MatMulGPU(a, b)

	expected := MatMul(a, b)
	for i := range expected.Data {
		if result.Data[i] != expected.Data[i] {
			t.Errorf("fallback result[%d] = %f, want %f", i, result.Data[i], expected.Data[i])
		}
	}
}
