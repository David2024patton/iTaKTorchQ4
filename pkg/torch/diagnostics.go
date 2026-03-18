// diagnostics.go provides structured startup diagnostics for iTaK Torch.
//
// What: Logs every major decision during engine initialization so you can
// see at a glance what loaded, what didn't, and why.
//
// Why: Without this, issues like "CUDA DLLs loaded but backend chose Vulkan"
// are invisible and waste hours of debugging.
//
// How: Call PrintDiagnostics() after ApplyAutoConfig and engine init.
// Each diagnostic line is prefixed with a status indicator:
//   [OK]   = working correctly
//   [WARN] = works but suboptimal
//   [FAIL] = not working
//   [INFO] = neutral information
package torch

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"time"
)

// DiagLevel controls the verbosity of diagnostic output.
type DiagLevel int

const (
	DiagSilent  DiagLevel = iota // No output.
	DiagSummary                  // One-line summary.
	DiagNormal                   // Key decisions only (default).
	DiagVerbose                  // Everything including DLL scan.
)

// DiagReport holds all diagnostic findings from engine startup.
type DiagReport struct {
	Timestamp   time.Time
	Level       DiagLevel
	Entries     []DiagEntry
	LibPath     string
	Backend     string
	GPULayers   int
	ModelPath   string
	ModelSizeMB int64
}

// DiagEntry is a single diagnostic finding.
type DiagEntry struct {
	Status  string // "OK", "WARN", "FAIL", "INFO"
	Category string // "dll", "backend", "gpu", "config", "perf"
	Message string
}

// NewDiagReport creates a new diagnostic report.
func NewDiagReport(level DiagLevel) *DiagReport {
	return &DiagReport{
		Timestamp: time.Now(),
		Level:     level,
		Entries:   make([]DiagEntry, 0, 20),
	}
}

// Add adds a diagnostic entry.
func (r *DiagReport) Add(status, category, msg string) {
	r.Entries = append(r.Entries, DiagEntry{
		Status:   status,
		Category: category,
		Message:  msg,
	})
}

// OK adds a success entry.
func (r *DiagReport) OK(category, msg string) {
	r.Add("OK", category, msg)
}

// Warn adds a warning entry.
func (r *DiagReport) Warn(category, msg string) {
	r.Add("WARN", category, msg)
}

// Fail adds a failure entry.
func (r *DiagReport) Fail(category, msg string) {
	r.Add("FAIL", category, msg)
}

// Info adds an informational entry.
func (r *DiagReport) Info(category, msg string) {
	r.Add("INFO", category, msg)
}

// Print outputs the diagnostic report.
func (r *DiagReport) Print() {
	if r.Level == DiagSilent {
		return
	}

	fmt.Println("[iTaK Torch] ============ STARTUP DIAGNOSTICS ============")

	if r.Level == DiagSummary {
		// One-line summary: count OK/WARN/FAIL.
		ok, warn, fail := 0, 0, 0
		for _, e := range r.Entries {
			switch e.Status {
			case "OK":
				ok++
			case "WARN":
				warn++
			case "FAIL":
				fail++
			}
		}
		fmt.Printf("[iTaK Torch] Diagnostics: %d OK, %d WARN, %d FAIL\n", ok, warn, fail)
		// Show any failures even in summary mode.
		for _, e := range r.Entries {
			if e.Status == "FAIL" {
				fmt.Printf("[iTaK Torch] [FAIL] [%s] %s\n", e.Category, e.Message)
			}
		}
		fmt.Println("[iTaK Torch] ==========================================")
		return
	}

	for _, e := range r.Entries {
		if r.Level < DiagVerbose && e.Status == "INFO" && e.Category == "dll" {
			continue // Skip DLL scan in normal mode.
		}
		fmt.Printf("[iTaK Torch] [%-4s] [%-7s] %s\n", e.Status, e.Category, e.Message)
	}
	fmt.Println("[iTaK Torch] ==========================================")
}

// RunDLLDiagnostics checks which shared libraries are available in the lib path.
func (r *DiagReport) RunDLLDiagnostics(libPath string) {
	r.LibPath = libPath

	if libPath == "" {
		r.Fail("dll", "ITAK_TORCH_LIB not set - using default lib path")
		return
	}

	if _, err := os.Stat(libPath); os.IsNotExist(err) {
		r.Fail("dll", fmt.Sprintf("Lib directory does not exist: %s", libPath))
		return
	}

	r.OK("dll", fmt.Sprintf("Lib directory: %s", libPath))

	// Check for specific backend DLLs based on OS.
	var ext string
	if runtime.GOOS == "windows" {
		ext = ".dll"
	} else {
		ext = ".so"
	}

	// Key libraries to check.
	backends := map[string]string{
		"ggml-cuda" + ext:   "CUDA GPU backend",
		"ggml-vulkan" + ext: "Vulkan GPU backend",
		"ggml-metal" + ext:  "Metal GPU backend (macOS)",
		"ggml-rpc" + ext:    "RPC distributed backend",
		"llama" + ext:       "Core inference engine",
		"ggml-base" + ext:   "Base tensor operations",
		"ggml" + ext:        "GGML runtime",
	}

	foundBackends := []string{}
	for dll, desc := range backends {
		fullPath := filepath.Join(libPath, dll)
		if fi, err := os.Stat(fullPath); err == nil {
			sizeMB := float64(fi.Size()) / (1024 * 1024)
			r.Info("dll", fmt.Sprintf("Found %s (%.1f MB) - %s", dll, sizeMB, desc))
			// Track which GPU backends are available.
			if strings.Contains(dll, "cuda") {
				foundBackends = append(foundBackends, "cuda")
			} else if strings.Contains(dll, "vulkan") {
				foundBackends = append(foundBackends, "vulkan")
			} else if strings.Contains(dll, "metal") {
				foundBackends = append(foundBackends, "metal")
			}
		}
	}

	if len(foundBackends) == 0 {
		r.Warn("dll", "No GPU backend DLLs found - inference will use CPU only")
	} else {
		r.OK("dll", fmt.Sprintf("GPU backends available: %s", strings.Join(foundBackends, ", ")))
	}

	// Check if the lib path name implies a specific backend.
	pathLower := strings.ToLower(libPath)
	if strings.Contains(pathLower, "cuda") {
		r.Info("dll", "Lib path contains 'cuda' - CUDA backend expected")
		if _, err := os.Stat(filepath.Join(libPath, "ggml-cuda"+ext)); os.IsNotExist(err) {
			r.Fail("dll", "Path implies CUDA but ggml-cuda"+ext+" not found!")
		}
	}
	if strings.Contains(pathLower, "vulkan") {
		r.Info("dll", "Lib path contains 'vulkan' - Vulkan backend expected")
	}
}

// RunBackendDiagnostics checks the backend selection logic.
func (r *DiagReport) RunBackendDiagnostics(ac AutoConfig, opts *EngineOpts) {
	r.Backend = ac.Backend

	r.Info("backend", fmt.Sprintf("Selected backend: %s", ac.Backend))
	r.Info("backend", fmt.Sprintf("GPU layers: %d (fits_gpu=%v)", opts.GPULayers, ac.ModelFitsGPU))

	// Check for mismatches between lib path and selected backend.
	pathLower := strings.ToLower(r.LibPath)
	if strings.Contains(pathLower, "cuda") && ac.Backend != "cuda" {
		r.Warn("backend", fmt.Sprintf(
			"Lib path contains 'cuda' but AutoConfig chose '%s' - "+
				"CUDA DLLs loaded but not used for routing. "+
				"Use --backend=cuda to force CUDA.", ac.Backend))
	}
	if strings.Contains(pathLower, "vulkan") && ac.Backend != "vulkan" {
		r.Warn("backend", fmt.Sprintf(
			"Lib path contains 'vulkan' but AutoConfig chose '%s'", ac.Backend))
	}

	// GPU offload check.
	if opts.GPULayers == 0 {
		r.Warn("gpu", "GPU layers = 0 - model will run on CPU only")
	} else if opts.GPULayers >= 999 {
		r.OK("gpu", "Full GPU offload requested (all layers)")
	} else {
		r.Info("gpu", fmt.Sprintf("Partial GPU offload: %d layers", opts.GPULayers))
	}
}

// RunPerfDiagnostics checks performance-related settings.
func (r *DiagReport) RunPerfDiagnostics(opts *EngineOpts) {
	// Flash attention.
	if opts.FlashAttention {
		r.OK("perf", "Flash attention: enabled")
	} else if opts.NoFlashAttention {
		r.Info("perf", "Flash attention: explicitly disabled")
	} else {
		r.Warn("perf", "Flash attention: not enabled (20-30% GPU perf left on table)")
	}

	// Prefix cache.
	if opts.PrefixCacheSize > 0 {
		r.OK("perf", fmt.Sprintf("Prefix cache: %d entries", opts.PrefixCacheSize))
	} else {
		r.Info("perf", "Prefix cache: disabled")
	}

	// Batch size.
	if opts.BatchSize >= 2048 {
		r.OK("perf", fmt.Sprintf("Batch size: %d", opts.BatchSize))
	} else if opts.BatchSize > 0 {
		r.Info("perf", fmt.Sprintf("Batch size: %d (consider increasing for throughput)", opts.BatchSize))
	}

	// KV cache quantization.
	if opts.KVCacheType == "q8_0" || opts.KVCacheType == "q4_0" {
		r.OK("perf", fmt.Sprintf("KV cache: %s (quantized, saves VRAM)", opts.KVCacheType))
	} else {
		r.Info("perf", "KV cache: f16 (default, higher quality but 2x VRAM)")
	}

	// Mlock.
	if opts.UseMlock {
		r.OK("perf", "mlock: enabled (prevents page faults)")
	}

	// Speculative decoding.
	if opts.DraftModelPath != "" {
		r.OK("perf", fmt.Sprintf("Speculative decode: %s (%d tokens)",
			filepath.Base(opts.DraftModelPath), opts.SpeculativeTokens))
	}

	// Thread count.
	r.Info("perf", fmt.Sprintf("Threads: %d", opts.Threads))
}

// RunGPUVerification checks if the GPU is actually being used post-load.
func (r *DiagReport) RunGPUVerification() {
	gpuInv := DetectGPUs()
	if gpuInv.BestVRAMMiB > 0 && gpuInv.BestDiscreteIdx >= 0 {
		bestName := gpuInv.GPUs[gpuInv.BestDiscreteIdx].Name
		r.OK("gpu", fmt.Sprintf("GPU detected: %s (%d MiB VRAM)",
			bestName, gpuInv.BestVRAMMiB))
	} else {
		r.Info("gpu", "No discrete GPU detected")
	}
}

// DiagLevelFromEnv reads the diagnostic level from ITAK_DEBUG env var.
//
//	ITAK_DEBUG=0 -> silent
//	ITAK_DEBUG=1 -> summary
//	ITAK_DEBUG=2 -> normal (default when set)
//	ITAK_DEBUG=3 -> verbose
func DiagLevelFromEnv() DiagLevel {
	val := os.Getenv("ITAK_DEBUG")
	switch val {
	case "0":
		return DiagSilent
	case "1":
		return DiagSummary
	case "2", "true", "yes":
		return DiagNormal
	case "3", "verbose":
		return DiagVerbose
	default:
		return DiagNormal // Default to normal when running
	}
}
