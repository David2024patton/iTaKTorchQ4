package torch

import (
	"runtime"
	"testing"
)

func TestDetectOptimalThreads_SmallModel(t *testing.T) {
	// Small model (<1GB) should use fewer threads.
	threads := DetectOptimalThreads(500, 0) // 500MB model, no GPU
	logical := runtime.NumCPU()

	if threads >= logical {
		t.Errorf("small model should not use all %d logical cores, got %d", logical, threads)
	}
	if threads < 1 {
		t.Errorf("threads must be >= 1, got %d", threads)
	}
	t.Logf("small model (500MB, no GPU): %d threads (of %d logical)", threads, logical)
}

func TestDetectOptimalThreads_LargeModel(t *testing.T) {
	// Large model (>8GB) should use more threads than small model.
	threads := DetectOptimalThreads(16000, 0) // 16GB model, no GPU
	logical := runtime.NumCPU()

	if threads < 1 {
		t.Errorf("threads must be >= 1, got %d", threads)
	}
	if threads > logical {
		t.Errorf("threads must be <= %d logical cores, got %d", logical, threads)
	}
	t.Logf("large model (16GB, no GPU): %d threads (of %d logical)", threads, logical)
}

func TestDetectOptimalThreads_GPUOffload(t *testing.T) {
	// GPU offload should reduce CPU threads.
	noGPU := DetectOptimalThreads(4000, 0)
	withGPU := DetectOptimalThreads(4000, 32)

	if withGPU > noGPU {
		t.Errorf("GPU offload should not increase threads: noGPU=%d, withGPU=%d", noGPU, withGPU)
	}
	t.Logf("4GB model: noGPU=%d threads, withGPU=%d threads", noGPU, withGPU)
}

func TestDetectOptimalThreads_ZeroModel(t *testing.T) {
	// When model size is unknown (0), should still return valid threads.
	threads := DetectOptimalThreads(0, 0)
	if threads < 1 {
		t.Errorf("threads must be >= 1, got %d", threads)
	}
	t.Logf("unknown model: %d threads", threads)
}

func TestDetectCPUTopology(t *testing.T) {
	topo := DetectCPUTopology()

	if topo.LogicalCores < 1 {
		t.Errorf("LogicalCores must be >= 1, got %d", topo.LogicalCores)
	}
	if topo.PhysicalCores < 1 {
		t.Errorf("PhysicalCores must be >= 1, got %d", topo.PhysicalCores)
	}
	if topo.PhysicalCores > topo.LogicalCores {
		t.Errorf("PhysicalCores (%d) > LogicalCores (%d)", topo.PhysicalCores, topo.LogicalCores)
	}
	t.Logf("CPU topology: %+v", topo)
}

func TestThreadAutoTuneReason(t *testing.T) {
	topo := CPUTopology{
		LogicalCores:  32,
		PhysicalCores: 16,
		HasHT:         true,
		Platform:      "linux/amd64",
	}
	reason := ThreadAutoTuneReason(8, topo, 500, 32)
	if reason == "" {
		t.Error("expected non-empty reason string")
	}
	t.Logf("Reason: %s", reason)
}

func TestFallbackPhysical(t *testing.T) {
	tests := []struct {
		logical  int
		expected int
	}{
		{32, 16},
		{16, 8},
		{4, 2},
		{2, 1},
		{1, 1}, // minimum clamp
	}
	for _, tc := range tests {
		got := fallbackPhysical(tc.logical)
		if got != tc.expected {
			t.Errorf("fallbackPhysical(%d) = %d, want %d", tc.logical, got, tc.expected)
		}
	}
}
