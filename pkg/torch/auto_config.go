// auto_config.go determines the optimal backend, GPU layers, and thread count
// based on detected hardware and model characteristics.
//
// Decision flowchart:
//
//  1. Detect GPUs and system RAM.
//  2. Estimate model VRAM requirements from file size.
//  3. Pick the best backend:
//     a. If discrete NVIDIA GPU with enough VRAM -> try CUDA first, fall back to Vulkan.
//     b. If discrete AMD GPU with enough VRAM -> Vulkan.
//     c. If only iGPU -> CPU-only (iGPU is slower than CPU for LLMs per our benchmarks).
//     d. If Apple Silicon -> Metal.
//     e. Otherwise -> CPU-only.
//  4. Determine GPU layers: all-or-nothing (partial offload is slower than CPU per benchmarks).
//  5. Auto-tune thread count based on CPU topology.
package torch

import (
	"fmt"
	"os"
	"runtime"
	"strings"
)

// AutoConfig holds the recommended configuration for a given model + hardware combo.
type AutoConfig struct {
	Backend      string // "cuda", "vulkan", "metal", "cpu"
	GPULayers    int    // 0 = CPU-only, 999 = all layers on GPU
	Threads      int    // Optimal CPU thread count
	GPUDevice    string // e.g. "Vulkan0", "" for auto
	Reason       string // Human-readable explanation of the decision
	LibDir       string // Recommended library directory suffix (e.g. "_cuda", "_vulkan", "")
	ModelFitsGPU bool   // Whether the entire model fits in GPU VRAM
}

// String returns a compact one-line summary.
func (ac AutoConfig) String() string {
	return fmt.Sprintf("backend=%s gpu_layers=%d threads=%d device=%s fits_gpu=%v | %s",
		ac.Backend, ac.GPULayers, ac.Threads, ac.GPUDevice, ac.ModelFitsGPU, ac.Reason)
}

// DetectAutoConfig probes the system and returns the optimal configuration
// for the given model file. If modelPath is empty, returns a generic recommendation.
func DetectAutoConfig(modelPath string) AutoConfig {
	gpus := DetectGPUs()
	topo := DetectCPUTopology()

	// Estimate model VRAM requirement (model weights + KV cache overhead).
	// Rule of thumb: VRAM needed ~ 1.2x model file size (weights + KV cache + workspace).
	var modelSizeMB int64
	if modelPath != "" {
		if fi, err := os.Stat(modelPath); err == nil {
			modelSizeMB = fi.Size() / (1024 * 1024)
		}
	}
	vramNeededMiB := int64(float64(modelSizeMB) * 1.3) // 30% overhead for KV cache + compute buffers

	var reasons []string
	reasons = append(reasons, fmt.Sprintf("model=%.1fGB", float64(modelSizeMB)/1024))
	reasons = append(reasons, fmt.Sprintf("vram_needed~%dMiB", vramNeededMiB))
	reasons = append(reasons, fmt.Sprintf("cpu=%dP/%dL cores", topo.PhysicalCores, topo.LogicalCores))

	for _, g := range gpus.GPUs {
		reasons = append(reasons, fmt.Sprintf("gpu[%d]=%s(%dMiB)", g.Index, g.Vendor, g.VRAMMiB))
	}

	ac := AutoConfig{}

	// Decision tree.
	switch {
	// Case 1: Discrete NVIDIA GPU with enough VRAM -> full GPU offload.
	case gpus.HasNVIDIA && gpus.BestVRAMMiB > 0 && gpus.BestVRAMMiB >= vramNeededMiB:
		// Vulkan is faster than CUDA on modern llama.cpp (b8398+):
		// - Vulkan: 465 tok/s, 737MB VRAM, no graph warmup overhead
		// - CUDA:   428 tok/s, 996MB VRAM, CUDA graph warmup per batch
		// Fall back to CUDA if Vulkan libs aren't available.
		ac.Backend = "vulkan"
		ac.LibDir = "_vulkan"
		ac.GPULayers = 999
		ac.ModelFitsGPU = true
		ac.Threads = DetectOptimalThreads(modelSizeMB, 999)
		reasons = append(reasons, fmt.Sprintf("NVIDIA %dMiB >= needed %dMiB -> full GPU offload (Vulkan preferred, 8%% faster than CUDA on b8398)", gpus.BestVRAMMiB, vramNeededMiB))

	// Case 2: Discrete NVIDIA GPU but model too large for VRAM.
	case gpus.HasNVIDIA && gpus.BestVRAMMiB > 0 && gpus.BestVRAMMiB < vramNeededMiB:
		// Per benchmarks: partial GPU offload is worse than CPU-only due to PCIe overhead.
		// Run CPU-only instead of split.
		ac.Backend = "cpu"
		ac.LibDir = ""
		ac.GPULayers = 0
		ac.ModelFitsGPU = false
		ac.Threads = DetectOptimalThreads(modelSizeMB, 0)
		reasons = append(reasons, fmt.Sprintf("NVIDIA %dMiB < needed %dMiB -> CPU-only (split offload is slower)", gpus.BestVRAMMiB, vramNeededMiB))

	// Case 3: Discrete AMD GPU with dedicated VRAM -> Vulkan full offload.
	case gpus.HasAMD && gpus.BestDiscreteIdx >= 0 && gpus.BestVRAMMiB >= vramNeededMiB:
		ac.Backend = "vulkan"
		ac.LibDir = "_vulkan"
		ac.GPULayers = 999
		ac.ModelFitsGPU = true
		ac.Threads = DetectOptimalThreads(modelSizeMB, 999)
		reasons = append(reasons, fmt.Sprintf("AMD discrete %dMiB >= needed %dMiB -> full Vulkan offload", gpus.BestVRAMMiB, vramNeededMiB))

	// Case 4: Apple Silicon -> Metal (unified memory, always fits).
	case runtime.GOOS == "darwin" && runtime.GOARCH == "arm64":
		ac.Backend = "metal"
		ac.LibDir = "_metal"
		ac.GPULayers = 999
		ac.ModelFitsGPU = true // UMA - GPU and CPU share same memory
		ac.Threads = DetectOptimalThreads(modelSizeMB, 999)
		reasons = append(reasons, "Apple Silicon UMA -> full Metal offload")

	// Case 5: Only iGPU (Intel/AMD integrated) -> CPU-only.
	// Benchmark proved iGPU is SLOWER than CPU for LLM inference.
	case (gpus.HasIntel || gpus.HasAMD) && gpus.BestDiscreteIdx == -1:
		ac.Backend = "cpu"
		ac.LibDir = ""
		ac.GPULayers = 0
		ac.ModelFitsGPU = false
		ac.Threads = DetectOptimalThreads(modelSizeMB, 0)
		reasons = append(reasons, "iGPU only (no discrete GPU) -> CPU-only (iGPU is slower than CPU for LLMs)")

	// Case 6: No GPU detected -> CPU-only.
	default:
		ac.Backend = "cpu"
		ac.LibDir = ""
		ac.GPULayers = 0
		ac.ModelFitsGPU = false
		ac.Threads = DetectOptimalThreads(modelSizeMB, 0)
		reasons = append(reasons, "no compatible GPU detected -> CPU-only")
	}

	ac.Reason = strings.Join(reasons, " | ")
	return ac
}

// ApplyAutoConfig applies auto-detected settings to EngineOpts, respecting
// any user overrides. Fields that are already set by the user are not modified.
func ApplyAutoConfig(opts *EngineOpts, modelPath string) AutoConfig {
	ac := DetectAutoConfig(modelPath)

	// Only apply auto-config for fields the user hasn't explicitly set.
	if opts.Backend == "" || opts.Backend == "auto" {
		opts.Backend = ac.Backend
	}

	if opts.GPULayers == 0 && ac.GPULayers > 0 {
		// User didn't specify GPU layers and we recommend GPU offload.
		// Only auto-offload if the model fits. Never auto-split.
		if ac.ModelFitsGPU {
			opts.GPULayers = ac.GPULayers
		}
	}

	if opts.Threads == 0 {
		opts.Threads = ac.Threads
	}

	// Auto-enable flash attention when using GPU (20-30% faster attention).
	// Flash attention uses fused Q/K/V kernels that reduce VRAM bandwidth.
	// Only auto-enable; never auto-disable if user explicitly set it.
	if opts.GPULayers > 0 && !opts.NoFlashAttention && !opts.FlashAttention {
		opts.FlashAttention = true
		fmt.Println("[iTaK Torch] Auto-enabled: Flash Attention (GPU detected)")
	}

	// Auto-enable mlock when GPU offloading to prevent OS swapping model pages.
	// Swapped pages cause 100x latency spikes when accessed during inference.
	if opts.GPULayers > 0 && !opts.UseMlock {
		opts.UseMlock = true
		fmt.Println("[iTaK Torch] Auto-enabled: mlock (prevents page swaps during inference)")
	}

	// Auto-set defrag threshold for KV cache maintenance.
	// Prevents gradual KV cache fragmentation in long-running sessions.
	if opts.DefragThreshold <= 0 {
		opts.DefragThreshold = 0.1
	}

	// Auto-tune batch size based on available VRAM.
	if opts.BatchSize == 0 {
		gpuInv := DetectGPUs()
		var modelMB int64
		if modelPath != "" {
			if fi, err := os.Stat(modelPath); err == nil {
				modelMB = fi.Size() / (1024 * 1024)
			}
		}
		opts.BatchSize = TuneBatchSize(modelMB, ac.ModelFitsGPU, gpuInv.BestVRAMMiB, opts.ContextSize)
	}

	return ac
}

// RecommendLibPath returns the best library directory path based on AutoConfig.
// It checks which directories actually exist and falls back gracefully.
func RecommendLibPath(ac AutoConfig) string {
	platformDir := runtime.GOOS + "_" + runtime.GOARCH

	// Build candidates list based on recommended backend.
	var candidates []string

	switch ac.Backend {
	case "cuda":
		candidates = []string{
			"./lib/" + platformDir + "_cuda",
			"./lib/" + platformDir + "_vulkan", // Fallback: Vulkan works on NVIDIA too
			"./lib/" + platformDir,             // Last resort: CPU-only
		}
	case "vulkan":
		candidates = []string{
			"./lib/" + platformDir + "_vulkan",
			"./lib/" + platformDir + "_cuda", // Fallback: CUDA works too on NVIDIA
			"./lib/" + platformDir,
		}
	case "metal":
		candidates = []string{
			"./lib/" + platformDir + "_metal",
			"./lib/" + platformDir,
		}
	default: // "cpu"
		candidates = []string{
			"./lib/" + platformDir,
			"./lib/" + platformDir + "_vulkan", // Vulkan CPU fallback is fine
			"./lib/" + platformDir + "_cuda",
		}
	}

	// Also check user home directory.
	if home, err := os.UserHomeDir(); err == nil {
		candidates = append(candidates, home+"/.itaktorch/lib")
	}

	for _, c := range candidates {
		if _, err := os.Stat(c); err == nil {
			return c
		}
	}

	return "" // Nothing found
}
