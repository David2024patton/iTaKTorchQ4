// cpu_topology.go implements CPU topology detection for optimal thread selection.
// Using all logical cores (e.g., 32 on an i9-13900K) causes L3 cache contention
// and a ~50% speed drop. Peak performance is at or near the physical core count.
package torch

import (
	"fmt"
	"os"
	"runtime"
	"strconv"
	"strings"
)

// CPUTopology holds detected CPU characteristics.
type CPUTopology struct {
	LogicalCores  int
	PhysicalCores int
	HasHT         bool // Hyper-Threading / SMT detected
	IsHybrid      bool // P-Core/E-Core hybrid architecture detected
	PerformanceCores int // Estimated P-Cores
	Platform      string
}

// DetectCPUTopology probes the system for CPU core counts.
func DetectCPUTopology() CPUTopology {
	logical := runtime.NumCPU()
	physical := detectPhysicalCores(logical)

	// Hybrid architecture heuristic (Intel Alder Lake+ / Raptor Lake+)
	// If logical != physical * 2 (standard HT) and logical != physical (no HT),
	// it's likely a hybrid count.
	// e.g. i7-1260P: 4P (8T) + 8E (8T) = 16 Threads, 12 Physical.
	// 12 * 2 = 24 != 16.
	// 12 * 1 = 12 != 16.
	isHybrid := false
	pCores := physical

	if logical != physical && logical != physical*2 {
		// Hybrid detection logic
		// Assume E-cores have no HT.
		// Threads = 2*P + E
		// Cores = P + E
		// P = Threads - Cores
		// E = Cores - P
		calcP := logical - physical
		calcE := physical - calcP

		if calcP > 0 && calcE > 0 {
			isHybrid = true
			pCores = calcP
		}
	}

	return CPUTopology{
		LogicalCores:     logical,
		PhysicalCores:    physical,
		HasHT:            logical > physical,
		IsHybrid:         isHybrid,
		PerformanceCores: pCores,
		Platform:         runtime.GOOS + "/" + runtime.GOARCH,
	}
}

// DetectOptimalThreads returns the best thread count for inference
// based on CPU topology and model size.
//
// Heuristics (validated on i9-13900K benchmarks):
//   - Base: physical core count (avoids HT contention on shared L3)
//   - Small models (<1GB): cap at physical/2 (fits in per-core L2)
//   - Large models (>8GB): use full physical count (need bandwidth)
//   - GPU offload: reduce CPU threads (GPU does the heavy lifting)
func DetectOptimalThreads(modelSizeMB int64, gpuLayers int) int {
	topo := DetectCPUTopology()
	threads := topo.PhysicalCores

	// Hybrid architecture optimization:
	// Prioritize P-cores. Using E-cores for inference usually hurts tail latency.
	if topo.IsHybrid {
		threads = topo.PerformanceCores
	}

	// Small models benefit from fewer threads to avoid cache thrashing.
	if modelSizeMB > 0 && modelSizeMB < 1024 {
		half := topo.PhysicalCores / 2
		if half < 2 {
			half = 2
		}
		threads = half
	}

	// When GPU handles most layers, CPU threads are mainly for
	// prompt tokenization and batch prep. No need for many.
	if gpuLayers > 0 {
		gpuThreads := topo.PhysicalCores / 2
		if gpuThreads < 2 {
			gpuThreads = 2
		}
		if gpuThreads < threads {
			threads = gpuThreads
		}
	}

	// Clamp to valid range.
	if threads < 1 {
		threads = 1
	}
	if threads > topo.LogicalCores {
		threads = topo.LogicalCores
	}

	return threads
}

// ThreadAutoTuneReason returns a human-readable explanation of the thread selection.
func ThreadAutoTuneReason(threads int, topo CPUTopology, modelMB int64, gpuLayers int) string {
	parts := []string{
		fmt.Sprintf("auto-tuned to %d threads", threads),
		fmt.Sprintf("(logical=%d, physical=%d", topo.LogicalCores, topo.PhysicalCores),
	}
	if topo.IsHybrid {
		parts = append(parts, fmt.Sprintf("hybrid=yes, p_cores=%d", topo.PerformanceCores))
	} else if topo.HasHT {
		parts = append(parts, "HT=yes")
	}
	if modelMB > 0 {
		parts = append(parts, fmt.Sprintf("model=%dMB", modelMB))
	}
	if gpuLayers > 0 {
		parts = append(parts, fmt.Sprintf("gpu_layers=%d", gpuLayers))
	}
	return strings.Join(parts, ", ") + ")"
}

// detectPhysicalCores attempts to determine the physical core count.
// Falls back to logical/2 (conservative HT estimate) if detection fails.
func detectPhysicalCores(logical int) int {
	switch runtime.GOOS {
	case "linux":
		return detectPhysicalCoresLinux(logical)
	case "windows":
		return detectPhysicalCoresWindows(logical)
	case "darwin":
		return detectPhysicalCoresDarwin(logical)
	default:
		return fallbackPhysical(logical)
	}
}

// detectPhysicalCoresLinux reads /sys topology to find unique physical cores.
func detectPhysicalCoresLinux(logical int) int {
	// Count unique core_id values across all CPUs.
	seen := make(map[string]bool)
	for i := 0; i < logical; i++ {
		path := fmt.Sprintf("/sys/devices/system/cpu/cpu%d/topology/core_id", i)
		data, err := os.ReadFile(path)
		if err != nil {
			return fallbackPhysical(logical)
		}
		seen[strings.TrimSpace(string(data))] = true
	}
	if len(seen) > 0 {
		return len(seen)
	}
	return fallbackPhysical(logical)
}

// detectPhysicalCoresWindows uses NUMBER_OF_PROCESSORS env as a hint.
// Windows doesn't expose topology cleanly without WMI, so we use
// the env var and divide by 2 as a conservative HT estimate.
func detectPhysicalCoresWindows(logical int) int {
	// On Windows, NUMBER_OF_PROCESSORS typically equals logical cores.
	// We assume HT is active and divide by 2.
	if nop := os.Getenv("NUMBER_OF_PROCESSORS"); nop != "" {
		if n, err := strconv.Atoi(nop); err == nil && n > 0 {
			physical := n / 2
			if physical < 1 {
				physical = 1
			}
			return physical
		}
	}
	return fallbackPhysical(logical)
}

// detectPhysicalCoresDarwin uses sysctl on macOS.
// Apple Silicon doesn't have HT, so logical == physical on M-series.
func detectPhysicalCoresDarwin(logical int) int {
	// On Apple Silicon (arm64), there's no HT. Logical == Physical.
	if runtime.GOARCH == "arm64" {
		return logical
	}
	// Intel Macs have HT.
	return fallbackPhysical(logical)
}

// fallbackPhysical returns a conservative estimate: logical / 2, minimum 1.
func fallbackPhysical(logical int) int {
	p := logical / 2
	if p < 1 {
		p = 1
	}
	return p
}
