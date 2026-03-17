package torch

import (
	"fmt"
	"testing"
)

func TestDetectGPUs(t *testing.T) {
	inv := DetectGPUs()

	t.Logf("GPU Inventory:")
	t.Logf("  HasNVIDIA: %v", inv.HasNVIDIA)
	t.Logf("  HasAMD:    %v", inv.HasAMD)
	t.Logf("  HasIntel:  %v", inv.HasIntel)
	t.Logf("  BestDiscreteIdx: %d", inv.BestDiscreteIdx)
	t.Logf("  BestVRAMMiB:     %d", inv.BestVRAMMiB)

	for _, gpu := range inv.GPUs {
		t.Logf("  GPU: %s", gpu)
	}

	if len(inv.GPUs) == 0 {
		t.Log("  (no GPUs detected - this is OK on headless/CI)")
	}
}

func TestAutoConfigSmallModel(t *testing.T) {
	// Simulate a 1GB model (e.g., TinyLlama Q4)
	ac := DetectAutoConfig("")
	t.Logf("AutoConfig (no model): %s", ac)

	if ac.Threads < 1 {
		t.Errorf("Threads should be >= 1, got %d", ac.Threads)
	}
}

func TestAutoConfigWithModel(t *testing.T) {
	// Use the actual benchmark model if available.
	modelPath := "e:\\.agent\\GOAgent\\models\\Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
	ac := DetectAutoConfig(modelPath)
	t.Logf("AutoConfig (8B model): %s", ac)

	fmt.Printf("\n=== AUTO-CONFIG RECOMMENDATION ===\n")
	fmt.Printf("Backend:      %s\n", ac.Backend)
	fmt.Printf("GPU Layers:   %d\n", ac.GPULayers)
	fmt.Printf("Threads:      %d\n", ac.Threads)
	fmt.Printf("GPU Device:   %s\n", ac.GPUDevice)
	fmt.Printf("Model Fits:   %v\n", ac.ModelFitsGPU)
	fmt.Printf("Lib Dir:      %s\n", ac.LibDir)
	fmt.Printf("Reason:       %s\n", ac.Reason)
}

func TestRecommendLibPath(t *testing.T) {
	ac := DetectAutoConfig("")
	libPath := RecommendLibPath(ac)
	t.Logf("Recommended lib path: %q (backend=%s)", libPath, ac.Backend)
}
