package main

import (
	"bufio"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"path/filepath"
	"runtime"
	"runtime/debug"
	"runtime/pprof"
	"strings"
	"syscall"
	"time"

	_ "net/http/pprof" // Registers /debug/pprof/* handlers for live profiling.

	"github.com/David2024patton/iTaKTorch/pkg/torch"
	"github.com/David2024patton/iTaKTorch/pkg/torch/native"
)

const defaultPort = 41934
const defaultCacheDir = "" // empty = let puller use os.UserHomeDir()/.torch/models/

func main() {
	if len(os.Args) < 2 {
		printUsage()
		os.Exit(1)
	}

	switch os.Args[1] {
	case "serve":
		cmdServe(os.Args[2:])
	case "models", "list", "ls":
		cmdModels()
	case "catalog":
		cmdCatalog()
	case "recommend":
		cmdRecommend()
	case "scan", "s":
		cmdScan(os.Args[2:])
	case "convert", "c":
		cmdConvert(os.Args[2:])
	case "clean":
		cmdClean()
	case "rm", "delete":
		cmdRemove(os.Args[2:])
	case "nuke", "delete-all":
		cmdNuke()
	case "pull":
		cmdPull(os.Args[2:])
	case "ollama-pull", "opull":
		cmdOllamaPull(os.Args[2:])
	case "hf-pull", "hpull":
		cmdHFPull(os.Args[2:])
	case "bench":
		cmdBench(os.Args[2:])
	case "chat":
		cmdChat(os.Args[2:])
	case "help", "--help", "-h":
		printUsage()
	default:
		fmt.Fprintf(os.Stderr, "Unknown command: %s\n\n", os.Args[1])
		printUsage()
		os.Exit(1)
	}
}

func printBanner() {
	fmt.Println("")
	fmt.Println("\033[38;5;202m  ========================================\033[0m")
	fmt.Println("\033[1;38;5;202m       iTaK Torch - Local LLM Engine\033[0m")
	fmt.Println("\033[2;38;5;202m     Intelligent Task Automation Kernel\033[0m")
	fmt.Println("\033[38;5;202m  ========================================\033[0m")
	fmt.Println("")
}

func printUsage() {
	printBanner()
	fmt.Println("  \033[1mGo-Native LLM Inference Engine\033[0m")
	fmt.Println(`
Usage:
  torch <command> [options]

Commands:
  serve       Load a GGUF model and start the inference server
  models      List cached models (alias: list, ls)
  scan        Search your hard drives for .gguf files and track their folders
  convert     Manually convert a discovered HuggingFace Safetensors model to .gguf
  clean       Clear all watched directories to empty the scan list
  rm          Permanently delete a tracked model from your hard drive (alias: delete)
  nuke        DANGER: Permanently delete ALL tracked models from your hard drive
  catalog     Show all available models with family and hardware info
  recommend   Detect your hardware and recommend compatible models
  pull        Download a model from the curated catalog by name
  ollama-pull Download a model from the Ollama registry (alias: opull)
  hf-pull     Download a model from HuggingFace Hub (alias: hpull)

Examples:
  torch serve --model ./models/qwen3-0.6b.gguf --port 11434
  torch serve --models-dir ~/.torch/models --max-models 2 --port 11434
  torch serve --model main.gguf --draft-model draft.gguf --speculative-tokens 5
  torch serve --model ./model.gguf --ollama-api=false  (disable Ollama compat)
  torch ollama-pull nemotron-3-nano:30b-a3b-q4_K_M
  torch recommend
  torch pull qwen3-0.6b-q4_k_m
  torch list`)
}

func cmdServe(args []string) {
	fs := flag.NewFlagSet("serve", flag.ExitOnError)
	modelPath := fs.String("model", "", "Path to GGUF model file (single-model mode)")
	mmprojPath := fs.String("mmproj", "", "Path to multimodal projector GGUF (for vision models)")
	modelsDir := fs.String("models-dir", "", "Directory of GGUF models (multi-model mode)")
	maxModels := fs.Int("max-models", 1, "Max models loaded simultaneously in multi-model mode")
	port := fs.Int("port", defaultPort, "Port to listen on")
	useMock := fs.Bool("mock", false, "Use mock engine (for testing without a real model)")
	ctxSize := fs.Int("ctx", 2048, "Context window size")
	threads := fs.Int("threads", 0, "Number of CPU threads (0 = auto-detect)")
	gpuLayers := fs.Int("gpu-layers", 0, "Layers to offload to GPU (0=CPU, -1=all, N=specific count)")
	flashAttn := fs.Bool("flash-attn", true, "Enable flash attention for faster inference")
	useMlock := fs.Bool("mlock", false, "Lock model in RAM to prevent OS swapping")
	numaStrategy := fs.Int("numa", 0, "NUMA strategy (0=disabled, 1=distribute, 2=isolate)")
	batchSize := fs.Int("batch", 2048, "Logical batch size for prompt processing")
	kvCacheType := fs.String("kv-cache-type", "f16", "KV cache quantization: f16 (default), q8_0 (50% less VRAM), q4_0 (75% less)")
	defragThreshold := fs.Float64("defrag-threshold", -1, "KV cache defrag threshold (0.0-1.0, -1=disabled, 0.1=recommended)")
	maxSlots := fs.Int("max-slots", 1, "Concurrent inference slots for continuous batching (1=sequential, 4=recommended)")
	backend := fs.String("backend", "auto", "GPU backend: auto (default), cuda, vulkan, cpu")

	// Speculative decoding flags (Phase 3 Stretch).
	draftModel := fs.String("draft-model", "", "Path to draft GGUF model for speculative decoding")
	draftGPULayers := fs.Int("draft-gpu-layers", 0, "GPU layers for draft model (0 = same as --gpu-layers)")
	specTokens := fs.Int("speculative-tokens", 5, "Number of tokens to speculate ahead per step")
	
	tensorSplit := fs.String("tensor-split", "", "Comma-separated split of VRAM usage across GPUs (e.g. 0.5,0.5)")
	
	prefixCacheSize := fs.Int("prefix-cache-size", 16, "Max cached KV states for identical system prompts (0=disabled)")
	enableOllama := fs.Bool("enable-ollama", false, "Enable Ollama model pull endpoint (/v1/models/pull/ollama)")
	ollamaAPI := fs.Bool("ollama-api", true, "Enable Ollama-compatible API routes (/api/generate, /api/chat, etc.)")
	noMmap := fs.Bool("no-mmap", false, "Disable mmap (required on WSL2 for /mnt/ paths)")
	pgoCapture := fs.String("pgo-capture", "", "Save CPU profile to file on shutdown (for PGO builds)")
	goGC := fs.Int("gogc", 100, "Garbage collection target percentage (100=default, 200=less GC/more GC, off=no GC)")
	tokenFile := fs.String("token-file", "", "Path to a JSON file containing an array of raw token IDs to inject (bypasses tokenizer)")
	imageFile := fs.String("image", "", "Path to an image file for multimodal inference (VLM boundary)")

	fs.Parse(args)

	// Apply GOGC tuning.
	if *goGC != 100 {
		if *goGC <= 0 {
			debug.SetGCPercent(-1) // "off"
			fmt.Println("[gotensor] Garbage collection disabled (GOGC=off)")
		} else {
			debug.SetGCPercent(*goGC)
			fmt.Printf("[gotensor] Garbage collection target set to %d%%\n", *goGC)
		}
	}

	opts := torch.EngineOpts{
		ContextSize:       *ctxSize,
		Threads:           *threads,
		GPULayers:         *gpuLayers,
		FlashAttention:    *flashAttn,
		NoFlashAttention:  !*flashAttn,
		UseMlock:          *useMlock,
		NumaStrategy:      *numaStrategy,
		BatchSize:         *batchSize,
		KVCacheType:       *kvCacheType,
		DefragThreshold:   float32(*defragThreshold),
		MaxSlots:          *maxSlots,
		Backend:           *backend,
		DraftModelPath:    *draftModel,
		DraftGPULayers:    *draftGPULayers,
		SpeculativeTokens: *specTokens,
		TensorSplit:       *tensorSplit,
		PrefixCacheSize:   *prefixCacheSize,
		NoMmap:            *noMmap,
	}

	var engine torch.Engine
	var serverOpts []torch.ServerOption

	if *useMock {
		mockName := "mock-model"
		if *modelPath != "" {
			mockName = *modelPath
		}
		engine = torch.NewMockEngine(mockName)
		fmt.Println("[iTaK Torch] Using mock engine (no real inference)")
	} else if *modelsDir != "" {
		// Multi-model mode: use ModelRegistry to dynamically load/unload models.
		fmt.Printf("[iTaK Torch] Multi-model mode: dir=%s max_loaded=%d\n", *modelsDir, *maxModels)
		registry, err := torch.NewModelRegistry(*modelsDir, *maxModels, opts)
		if err != nil {
			fmt.Fprintf(os.Stderr, "[iTaK Torch] Failed to initialize model registry: %v\n", err)
			os.Exit(1)
		}

		// Use a mock engine as placeholder (registry handles real engines).
		engine = torch.NewMockEngine("registry-placeholder")
		serverOpts = append(serverOpts, torch.WithRegistry(registry))
	} else if *modelPath != "" {
		fmt.Printf("[iTaK Torch] Loading model: %s\n", *modelPath)

		// ----- Auto-Convert Safetensors to GGUF -----
		cacheDir := defaultCacheDir
		if cacheDir == "" {
			if home, err := os.UserHomeDir(); err == nil {
				cacheDir = home + "/.torch/models/"
			} else {
				cacheDir = "./models"
			}
		}
		
		convertedPath, err := torch.AutoConvert(*modelPath, cacheDir)
		if err != nil {
			fmt.Fprintf(os.Stderr, "[iTaK Torch] Safetensors Auto-Conversion failed: %v\n", err)
			os.Exit(1)
		}
		*modelPath = convertedPath // Transparently handoff the new .gguf path
		// --------------------------------------------

		fmt.Printf("[iTaK Torch] Config: ctx=%d threads=%d gpu_layers=%d flash_attn=%v batch=%d\n",
			*ctxSize, *threads, *gpuLayers, *flashAttn, *batchSize)

		if *mmprojPath != "" {
			// Vision model: load both text model and multimodal projector.
			fmt.Printf("[iTaK Torch] Loading mmproj: %s\n", *mmprojPath)
			engine, err = torch.NewVisionEngine(*modelPath, *mmprojPath, opts)
		} else {
			// Text-only model.
			engine, err = torch.NewTorchEngine(*modelPath, opts)
		}
		if err != nil {
			fmt.Fprintf(os.Stderr, "[iTaK Torch] Failed to load model: %v\n", err)
			os.Exit(1)
		}
		fmt.Printf("[iTaK Torch] Model loaded successfully: %s\n", engine.ModelName())
	} else {
		fmt.Fprintf(os.Stderr, "Error: --model, --models-dir, or --mock is required\n")
		fmt.Fprintf(os.Stderr, "Usage: torch serve --model <path.gguf> --port <port>\n")
		fmt.Fprintf(os.Stderr, "       torch serve --models-dir <dir> --max-models 2 --port <port>\n")
		fmt.Fprintf(os.Stderr, "       torch serve --mock --port <port>\n")
		os.Exit(1)
	}

	// Token bypass execution. Run immediately and exit without starting HTTP server.
	if *tokenFile != "" {
		data, err := os.ReadFile(*tokenFile)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error reading token file: %v\n", err)
			os.Exit(1)
		}
		var inputTokens []int32
		if err := json.Unmarshal(data, &inputTokens); err != nil {
			fmt.Fprintf(os.Stderr, "Error parsing JSON token file: %v\n", err)
			os.Exit(1)
		}
		fmt.Printf("[iTaK Torch] Exact Token ID Injection: executing %d raw tokens\n", len(inputTokens))
		
		out, err := engine.GenerateTokens(context.Background(), inputTokens, torch.CompletionParams{})
		if err != nil {
			fmt.Fprintf(os.Stderr, "Token generation failed: %v\n", err)
			os.Exit(1)
		}
		
		fmt.Println("\n" + out)
		engine.Close()
		os.Exit(0)
	}

	// Multimodal VLM execution. Run immediately and exit.
	if *imageFile != "" {
		fmt.Printf("[iTaK Torch] Unified Multimodal Boundary: Processing image %s\n", *imageFile)
		
		// In a real usage, we would process this through vision_engine.go or vlm.go.
		// For the CLI direct injection, we trigger a standard Complete with the image path.
		messages := []torch.ChatMessage{
			{Role: "user", Content: "Describe this image."},
		}
		
		if visEngine, ok := engine.(*torch.VisionEngine); ok {
			// Extract image content by directly building the prompt with the image.
			// Note: extraction normally happens from Base64 or URLs in messages.
			// Since we just have a file path, we'll format it as an [img-...] tag.
			messages[0].Content = fmt.Sprintf("[img-%s] Describe this image.", *imageFile)
			out, err := visEngine.Complete(context.Background(), messages, torch.CompletionParams{})
			if err != nil {
				fmt.Fprintf(os.Stderr, "VLM inference error: %v\n", err)
				os.Exit(1)
			}
			fmt.Println("\n" + out)
		} else {
			fmt.Fprintf(os.Stderr, "Error: --image requires a vision projector loaded via --mmproj\n")
			os.Exit(1)
		}
		
		engine.Close()
		os.Exit(0)
	}

	if *enableOllama {
		cacheDir := defaultCacheDir
		if *modelsDir != "" {
			cacheDir = *modelsDir
		}
		serverOpts = append(serverOpts, torch.WithOllamaPuller(cacheDir))
		fmt.Println("[iTaK Torch] Ollama pull endpoint enabled")
	}

	// Ollama API compat: /api/generate, /api/chat, /api/tags, /api/show, /api/version.
	if *ollamaAPI {
		serverOpts = append(serverOpts, torch.WithOllamaCompat())
		fmt.Println("[iTaK Torch] Ollama-compatible API enabled (/api/generate, /api/chat, /api/tags)")
	}

	server := torch.NewServer(engine, *port, serverOpts...)

	// Graceful shutdown on Ctrl+C.
	stop := make(chan os.Signal, 1)
	signal.Notify(stop, syscall.SIGINT, syscall.SIGTERM)

	// PGO: start CPU profiling if --pgo-capture is set.
	var pgoFile *os.File
	if *pgoCapture != "" {
		var err error
		pgoFile, err = os.Create(*pgoCapture)
		if err != nil {
			fmt.Fprintf(os.Stderr, "[iTaK Torch] PGO: failed to create profile: %v\n", err)
			os.Exit(1)
		}
		pprof.StartCPUProfile(pgoFile)
		fmt.Printf("[iTaK Torch] PGO: recording CPU profile to %s (stop server to save)\n", *pgoCapture)
	}

	go func() {
		<-stop
		fmt.Println("\n[iTaK Torch] Shutting down...")

		// PGO: stop profiling and flush to disk.
		if pgoFile != nil {
			pprof.StopCPUProfile()
			pgoFile.Close()
			fmt.Printf("[iTaK Torch] PGO: CPU profile saved to %s\n", *pgoCapture)
			fmt.Println("[iTaK Torch] PGO: rebuild with: go build -pgo=" + *pgoCapture + " ./cmd/torch/")
		}

		server.Stop()
		engine.Close()
	}()

	if err := server.Start(); err != nil {
		fmt.Fprintf(os.Stderr, "Server error: %v\n", err)
		os.Exit(1)
	}
}

func cmdModels() {
	mgr, err := torch.NewModelManager(defaultCacheDir)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}

	models, err := mgr.List()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error listing models: %v\n", err)
		os.Exit(1)
	}

	if len(models) == 0 {
		fmt.Println("No cached models found.")
		fmt.Printf("Cache directory: %s\n", mgr.CacheDir())
		fmt.Println("Run 'torch catalog' to see available models.")
		return
	}

	printBanner()
	fmt.Printf("\033[1mCached models\033[0m (%s):\n\n", mgr.CacheDir())

	maxLen := 40
	for _, m := range models {
		if len(m.Name) > maxLen && len(m.Name) <= 55 {
			maxLen = len(m.Name)
		} else if len(m.Name) > 55 {
			maxLen = 55
		}
	}

	fmt.Printf("  \033[36m%-*s  %-10s  %-12s\033[0m\n", maxLen, "NAME", "SIZE", "LAST USED")
	fmt.Printf("  \033[36m%-*s  %-10s  %-12s\033[0m\n", maxLen, strings.Repeat("-", maxLen), strings.Repeat("-", 10), strings.Repeat("-", 12))
	
	for _, m := range models {
		sizeMB := m.Size / 1024 / 1024
		
		name := m.Name
		if len(name) > maxLen {
			name = name[:maxLen-3] + "..."
		}

		if strings.HasSuffix(m.Name, " (HF)") {
			fmt.Printf("  \033[33m%-*s  %6d MB  %s\033[0m\n", maxLen, name, sizeMB, m.LastUsed.Format("2006-01-02"))
		} else {
			fmt.Printf("  %-*s  %6d MB  %s\n", maxLen, name, sizeMB, m.LastUsed.Format("2006-01-02"))
		}
	}
	fmt.Println()
}

func cmdCatalog() {
	catalog := torch.CuratedModels()
	printBanner()
	fmt.Println("  \033[1mModel Catalog\033[0m")
	fmt.Println()

	// Group by family.
	lastFamily := ""
	fmt.Printf("  \033[36m%-30s  %-8s  %-8s  %-10s  %-8s  %-5s  %s\033[0m\n", "NAME", "PARAMS", "SIZE", "ROLE", "FAMILY", "DRAFT", "NOTES")
	fmt.Printf("  \033[36m%-30s  %-8s  %-8s  %-10s  %-8s  %-5s  %s\033[0m\n", strings.Repeat("-", 30), strings.Repeat("-", 8), strings.Repeat("-", 8), strings.Repeat("-", 10), strings.Repeat("-", 8), strings.Repeat("-", 5), strings.Repeat("-", 5))
	for _, m := range catalog {
		if m.Family != lastFamily {
			if lastFamily != "" {
				fmt.Println()
			}
			lastFamily = m.Family
		}
		draft := ""
		if m.CanDraft {
			draft = "  yes"
		}
		fmt.Printf("  %-30s  %-8s  %-8s  %-10s  %-8s  %-5s  %s\n", m.Name, m.Params, m.Size, m.Role, m.Family, draft, m.Notes)
	}
	fmt.Println()
	fmt.Println("Pull a model: torch pull <name>")
	fmt.Println("See what fits your hardware: torch recommend")
}

func cmdRecommend() {
	// Detect system hardware.
	var memInfo runtime.MemStats
	runtime.ReadMemStats(&memInfo)

	// Use Sys (total memory obtained from OS) as a rough estimate of available RAM.
	// For a more accurate reading we'd use OS-specific APIs, but this is a solid baseline.
	totalRAMMB := int(memInfo.Sys/1024/1024) + 8192 // Add headroom (Go only reports its own usage)

	// Rough heuristic: check if CUDA/HIP/SYCL libs exist to detect GPU.
	hasGPU := false
	vramMB := 0

	specs := torch.SystemSpecs{
		TotalRAMMB:  totalRAMMB,
		TotalVRAMMB: vramMB,
		HasGPU:      hasGPU,
	}

	fmt.Println("torch Hardware Detection")
	fmt.Println("=========================")
	fmt.Printf("  Estimated RAM:  %d MB\n", totalRAMMB)
	if hasGPU {
		fmt.Printf("  GPU VRAM:       %d MB\n", vramMB)
	} else {
		fmt.Printf("  GPU:            Not detected (CPU-only mode)\n")
	}
	fmt.Println()

	// Show models that fit.
	fits := torch.ModelsForHardware(specs)
	if len(fits) == 0 {
		fmt.Println("No models fit your hardware. Try upgrading RAM.")
		return
	}

	fmt.Printf("Models that fit your hardware (%d available):\n\n", len(fits))
	fmt.Printf("  %-30s  %-8s  %-8s  %-10s  %-8s  %s\n", "NAME", "PARAMS", "SIZE", "ROLE", "FAMILY", "NOTES")
	fmt.Printf("  %-30s  %-8s  %-8s  %-10s  %-8s  %s\n", "----", "------", "----", "----", "------", "-----")
	for _, m := range fits {
		fmt.Printf("  %-30s  %-8s  %-8s  %-10s  %-8s  %s\n", m.Name, m.Params, m.Size, m.Role, m.Family, m.Notes)
	}

	// Show speculative decoding pairs.
	pairs := torch.SpeculativePairsForHardware(specs)
	if len(pairs) > 0 {
		fmt.Println()
		fmt.Printf("Speculative Decoding Pairs (%d pairs available):\n", len(pairs))
		fmt.Println("  Draft models predict tokens, main models verify. Same family = compatible tokenizer.")
		fmt.Println()
		fmt.Printf("  %-20s  %-20s  %-8s  %s\n", "DRAFT MODEL", "MAIN MODEL", "FAMILY", "NOTES")
		fmt.Printf("  %-20s  %-20s  %-8s  %s\n", "-----------", "----------", "------", "-----")
		for _, p := range pairs {
			fmt.Printf("  %-20s  %-20s  %-8s  %s\n", p.DraftModel.Name, p.MainModel.Name, p.Family, p.Notes)
		}
		fmt.Println()
		fmt.Println("  Usage: torch serve --model <main.gguf> --draft-model <draft.gguf> --speculative-tokens 5")
	}

	fmt.Println()
	fmt.Println("Pull a model: torch pull <name>")
}

func cmdPull(args []string) {
	if len(args) == 0 {
		fmt.Fprintf(os.Stderr, "Usage: torch pull <model-name>\n")
		fmt.Fprintf(os.Stderr, "Run 'torch catalog' to see available models.\n")
		os.Exit(1)
	}

	name := args[0]

	// Find in catalog.
	var found *torch.ModelIndex
	for _, m := range torch.CuratedModels() {
		if m.Name == name {
			found = &m
			break
		}
	}

	if found == nil {
		fmt.Fprintf(os.Stderr, "Model %q not found in catalog.\n", name)
		fmt.Fprintf(os.Stderr, "Run 'torch catalog' to see available models.\n")
		os.Exit(1)
	}

	cacheDir := defaultCacheDir
	if cacheDir == "" {
		if home, err := os.UserHomeDir(); err == nil {
			cacheDir = filepath.Join(home, ".torch", "models")
		} else {
			cacheDir = "./models"
		}
	}

	puller, err := torch.NewHFPuller(cacheDir, "")
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error creating puller: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("[iTaK Torch] Pulling %s (%s, %s)...\n", found.Name, found.Params, found.Size)
	
	// Temporarily set the progress bar callback to the puller config
	puller.Progress = makeProgressBar(colorBlue)
	
	result, err := puller.Pull(context.Background(), found.URL, found.Name+".gguf")
	if err != nil {
		fmt.Fprintf(os.Stderr, "\nDownload failed: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("\n[iTaK Torch] Model saved to: %s\n", result.LocalPath)
}

// ---------- ANSI color helpers ----------

const (
	colorReset  = "\033[0m"
	colorGreen  = "\033[32m" // Ollama brand
	colorBlue   = "\033[34m" // HuggingFace brand
	colorYellow = "\033[33m"
	colorCyan   = "\033[36m"
	colorBold   = "\033[1m"
)

// makeProgressBar returns a Progress callback that renders a colored terminal
// progress bar with speed and ETA.
//
//	[=========>              ] 45.2% 1234/5678 MB  42.3 MB/s  ETA 2m30s
func makeProgressBar(barColor string) func(downloaded, total int64) {
	startTime := time.Now()
	var lastPrint time.Time

	return func(downloaded, total int64) {
		now := time.Now()
		// Throttle updates to every 500ms.
		if now.Sub(lastPrint) < 500*time.Millisecond && downloaded < total {
			return
		}
		lastPrint = now

		pct := float64(0)
		if total > 0 {
			pct = float64(downloaded) / float64(total) * 100
		}
		dlMB := float64(downloaded) / 1024 / 1024
		totalMB := float64(total) / 1024 / 1024

		// Speed and ETA.
		elapsed := now.Sub(startTime).Seconds()
		speedMBs := float64(0)
		eta := "calculating..."
		if elapsed > 1 {
			speedMBs = dlMB / elapsed
			if speedMBs > 0 {
				remainMB := totalMB - dlMB
				etaSec := remainMB / speedMBs
				if etaSec < 60 {
					eta = fmt.Sprintf("%.0fs", etaSec)
				} else if etaSec < 3600 {
					eta = fmt.Sprintf("%.0fm%.0fs", etaSec/60, float64(int(etaSec)%60))
				} else {
					eta = fmt.Sprintf("%.1fh", etaSec/3600)
				}
			}
		}

		// Build the bar.
		barWidth := 30
		filled := int(pct / 100 * float64(barWidth))
		if filled > barWidth {
			filled = barWidth
		}
		bar := ""
		for i := 0; i < barWidth; i++ {
			if i < filled {
				bar += "="
			} else if i == filled {
				bar += ">"
			} else {
				bar += " "
			}
		}

		fmt.Fprintf(os.Stderr, "\r  %s[%s]%s %5.1f%% %.0f/%.0f MB  %.1f MB/s  ETA %s   ",
			barColor, bar, colorReset, pct, dlMB, totalMB, speedMBs, eta)

		if downloaded >= total {
			fmt.Fprintln(os.Stderr)
		}
	}
}

// ---------- ollama-pull ----------

func cmdOllamaPull(args []string) {
	if len(args) == 0 {
		fmt.Fprintf(os.Stderr, "Usage: torch ollama-pull <model:tag>\n")
		fmt.Fprintf(os.Stderr, "Example: torch ollama-pull nemotron-3-nano:30b-a3b-q4_K_M\n")
		os.Exit(1)
	}

	modelRef := args[0]

	puller, err := torch.NewOllamaPuller(defaultCacheDir)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}

	puller.Progress = makeProgressBar(colorGreen)

	fmt.Printf("%s[iTaK Torch]%s Pulling from %sOllama%s registry: %s%s%s\n",
		colorBold, colorReset, colorGreen, colorReset, colorCyan, modelRef, colorReset)
	result, err := puller.Pull(context.Background(), modelRef)
	if err != nil {
		fmt.Fprintln(os.Stderr)
		fmt.Fprintf(os.Stderr, "Pull failed: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("%s[iTaK Torch]%s Model saved to: %s\n", colorBold, colorReset, result.LocalPath)
	fmt.Printf("%s[iTaK Torch]%s Size: %.1f GB\n", colorBold, colorReset, float64(result.Size)/1024/1024/1024)
}

// ---------- hf-pull ----------

func cmdHFPull(args []string) {
	if len(args) == 0 {
		fmt.Fprintf(os.Stderr, "Usage: torch hf-pull <repo> <filename>\n")
		fmt.Fprintf(os.Stderr, "Example: torch hf-pull Qwen/Qwen3-0.6B-GGUF qwen3-0.6b-q4_k_m.gguf\n")
		os.Exit(1)
	}

	repo := args[0]
	filename := ""
	if len(args) > 1 {
		filename = args[1]
	}

	token := os.Getenv("HF_TOKEN")
	puller, err := torch.NewHFPuller(defaultCacheDir, token)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}

	puller.Progress = makeProgressBar(colorBlue)

	// If no filename given, list GGUF files in the repo.
	if filename == "" {
		fmt.Printf("%s[iTaK Torch]%s Listing GGUF files in %s%s%s...\n",
			colorBold, colorReset, colorBlue, repo, colorReset)
		files, err := puller.ListRepoFiles(repo)
		if err != nil {
			fmt.Fprintf(os.Stderr, "List failed: %v\n", err)
			os.Exit(1)
		}
		if len(files) == 0 {
			fmt.Println("No GGUF files found in this repo.")
			return
		}
		fmt.Println()
		for _, f := range files {
			sizeMB := float64(f.Size) / 1024 / 1024
			if sizeMB > 1024 {
				fmt.Printf("  %s%-50s%s  %.1f GB\n", colorCyan, f.Filename, colorReset, sizeMB/1024)
			} else {
				fmt.Printf("  %s%-50s%s  %.0f MB\n", colorCyan, f.Filename, colorReset, sizeMB)
			}
		}
		fmt.Println()
		fmt.Println("Pull a file: torch hf-pull", repo, "<filename>")
		return
	}

	fmt.Printf("%s[iTaK Torch]%s Pulling from %sHuggingFace%s: %s/%s\n",
		colorBold, colorReset, colorBlue, colorReset, repo, filename)
	result, err := puller.Pull(context.Background(), repo, filename)
	if err != nil {
		fmt.Fprintln(os.Stderr)
		fmt.Fprintf(os.Stderr, "Pull failed: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("%s[iTaK Torch]%s Model saved to: %s\n", colorBold, colorReset, result.LocalPath)
	fmt.Printf("%s[iTaK Torch]%s Size: %.1f GB\n", colorBold, colorReset, float64(result.Size)/1024/1024/1024)
}

// cmdBench runs an enhanced benchmark on a GGUF model file.
//
// Usage:
//
//	torch bench <model.gguf> [flags]
//
// Flags:
//
//	--backend gotensor|ffi    Engine to benchmark (default: ffi if libs found, else gotensor)
//	--mode dense|sparse-70|sparse-90  Sparsity mode for gotensor (default: dense)
//	--json                   Output JSON instead of box art
//	--iterations N           Run N iterations, keep best (default: 3)
//	--gpu-layers N           GPU layers for FFI backend (-1=all, default: -1)
//	--threads N              CPU threads for FFI backend (0=auto, default: 0)
//	--max-tokens N           Max tokens to generate per benchmark prompt (default: 64)
//
// Shows system resources (CPU, RAM, GPU), latency percentiles (P50/P95/P99),
// TTFT, ITL, and power efficiency ratios alongside standard tok/s metrics.
func cmdBench(args []string) {
	if len(args) < 1 {
		fmt.Fprintln(os.Stderr, "Usage: torch bench <model.gguf> [--backend gotensor|ffi] [--json] [--iterations N] [--gpu-layers N]")
		os.Exit(1)
	}

	// Parse args: first positional is the model path, rest are flags.
	ggufPath := args[0]
	mode := "dense"
	backendFlag := "auto"
	jsonOutput := false
	iterations := 3
	gpuLayers := -1    // -1 = offload all
	threads := 0       // 0 = auto
	maxTokens := 64

	for i := 1; i < len(args); i++ {
		switch args[i] {
		case "--backend", "-b":
			if i+1 < len(args) {
				i++
				backendFlag = args[i]
			}
		case "--mode", "-m":
			if i+1 < len(args) {
				i++
				mode = args[i]
			}
		case "--json", "-j":
			jsonOutput = true
		case "--iterations", "-n":
			if i+1 < len(args) {
				i++
				fmt.Sscanf(args[i], "%d", &iterations)
			}
		case "--gpu-layers", "-gl":
			if i+1 < len(args) {
				i++
				fmt.Sscanf(args[i], "%d", &gpuLayers)
			}
		case "--threads", "-t":
			if i+1 < len(args) {
				i++
				fmt.Sscanf(args[i], "%d", &threads)
			}
		case "--max-tokens":
			if i+1 < len(args) {
				i++
				fmt.Sscanf(args[i], "%d", &maxTokens)
			}
		case "dense", "sparse-70", "sparse-90":
			mode = args[i] // Backward compat: positional mode argument.
		}
	}

	// Verify file exists.
	info, err := os.Stat(ggufPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: cannot open %s: %v\n", ggufPath, err)
		os.Exit(1)
	}

	// Auto-convert SafeTensors to GGUF if needed (same as cmdServe).
	if strings.HasSuffix(strings.ToLower(ggufPath), ".safetensors") {
		if !jsonOutput {
			fmt.Printf("[iTaK Torch] SafeTensors detected, auto-converting to GGUF...\n")
		}
		cacheDir := defaultCacheDir
		if cacheDir == "" {
			if home, err := os.UserHomeDir(); err == nil {
				cacheDir = home + "/.torch/models/"
			} else {
				cacheDir = "./models"
			}
		}
		converted, convErr := torch.AutoConvert(ggufPath, cacheDir)
		if convErr != nil {
			fmt.Fprintf(os.Stderr, "[iTaK Torch] SafeTensors auto-conversion failed: %v\n", convErr)
			os.Exit(1)
		}
		ggufPath = converted
		info, err = os.Stat(ggufPath)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error: converted file not found: %v\n", err)
			os.Exit(1)
		}
	}

	modelSizeGB := float64(info.Size()) / (1024 * 1024 * 1024)

	// Auto-detect backend: prefer FFI if libraries are present.
	if backendFlag == "auto" {
		if _, libErr := os.Stat("./lib"); libErr == nil {
			backendFlag = "ffi"
		} else if os.Getenv("ITAK_TORCH_LIB") != "" {
			backendFlag = "ffi"
		} else {
			backendFlag = "gotensor"
		}
	}

	if !jsonOutput {
		fmt.Printf("[iTaK Torch] Benchmark: %s (%.1f GB) backend=%s mode=%s\n",
			filepath.Base(ggufPath), modelSizeGB, backendFlag, mode)
	}

	// Capture system snapshot BEFORE loading.
	snapBefore := native.TakeSnapshot()

	if backendFlag == "ffi" {
		// ============================================================
		// FFI BENCHMARK PATH (llama.cpp via purego/ffi)
		// ============================================================
		benchFFI(ggufPath, modelSizeGB, mode, jsonOutput, iterations, gpuLayers, threads, maxTokens, snapBefore)
	} else {
		// ============================================================
		// GOTENSOR BENCHMARK PATH (pure Go, zero dependencies)
		// ============================================================
		benchGOTensor(ggufPath, modelSizeGB, mode, jsonOutput, iterations, maxTokens, snapBefore)
	}
}

// benchFFI runs the benchmark using the llama.cpp FFI backend.
func benchFFI(ggufPath string, modelSizeGB float64, mode string, jsonOutput bool, iterations, gpuLayers, threads, maxTokens int, snapBefore native.SystemSnapshot) {
	if !jsonOutput {
		fmt.Printf("[iTaK Torch] Loading model via FFI (llama.cpp)...\n")
	}

	opts := torch.EngineOpts{
		GPULayers: gpuLayers,
		Threads:   threads,
		Backend:   "auto",
	}

	loadStart := time.Now()
	engine, loadErr := torch.NewTorchEngine(ggufPath, opts)
	loadDur := time.Since(loadStart)
	if loadErr != nil {
		fmt.Fprintf(os.Stderr, "[iTaK Torch] FFI load failed: %v\n", loadErr)
		fmt.Fprintln(os.Stderr, "Hint: Set ITAK_TORCH_LIB to the directory containing llama.dll/libllama.so")
		os.Exit(1)
	}
	defer engine.Close()

	if !jsonOutput {
		fmt.Printf("[iTaK Torch] FFI model loaded in %v\n", loadDur.Round(time.Millisecond))
		fmt.Printf("[iTaK Torch] Running %d iteration(s) with %d max tokens...\n\n", iterations, maxTokens)
	}

	// Benchmark prompts: different lengths to test prefill and generation speed.
	prompts := []struct {
		Name   string
		Prompt string
	}{
		{"short_gen", "Say hello."},
		{"medium_gen", "Explain the concept of transformer attention mechanisms in neural networks."},
		{"long_gen", "Write a comprehensive explanation of how quantum computing works, covering qubits, superposition, entanglement, and quantum gates. Include examples and practical applications."},
	}

	report := &native.BenchmarkReport{
		Model:          engine.ModelName(),
		Engine:         "iTaK Torch FFI (llama.cpp)",
		Platform:       fmt.Sprintf("%s/%s", runtime.GOOS, runtime.GOARCH),
		Timestamp:      time.Now().Format(time.RFC3339),
		LoadDuration:   loadDur,
		Mode:           mode,
		Backend:        "ffi",
		ModelSizeGB:    modelSizeGB,
		SnapshotBefore: snapBefore,
	}

	// Run each prompt multiple iterations.
	for _, p := range prompts {
		var bestTPS float64
		var bestResult native.BenchmarkResult

		for iter := 0; iter < iterations; iter++ {
			if !jsonOutput && iterations > 1 {
				fmt.Printf("  %s [%d/%d] ...", p.Name, iter+1, iterations)
			} else if !jsonOutput {
				fmt.Printf("  %s ...", p.Name)
			}

			messages := []torch.ChatMessage{
				{Role: "user", Content: p.Prompt},
			}
			params := torch.CompletionParams{MaxTokens: maxTokens}

			genStart := time.Now()
			output, genErr := engine.Complete(context.Background(), messages, params)
			genDur := time.Since(genStart)

			if genErr != nil {
				if !jsonOutput {
					fmt.Printf(" ERROR: %v\n", genErr)
				}
				continue
			}

			// Get stats from the engine.
			stats := engine.GetStats()
			tokGenerated := 0
			tps := 0.0
			var ttft time.Duration

			if stats.LastMetrics != nil {
				tokGenerated = stats.LastMetrics.CompletionTokens
				tps = stats.LastMetrics.TokensPerSecond
				ttft = stats.LastMetrics.PromptDuration
			}
			if tokGenerated == 0 {
				tokGenerated = len(output) / 4 // rough char-to-token estimate
			}
			if tps == 0 && genDur.Seconds() > 0 {
				tps = float64(tokGenerated) / genDur.Seconds()
			}

			if !jsonOutput {
				fmt.Printf(" %.1f tok/s (%d tokens in %v)\n", tps, tokGenerated, genDur.Round(time.Millisecond))
			}

			if tps > bestTPS {
				bestTPS = tps
				bestResult = native.BenchmarkResult{
					Name:            p.Name,
					TokensGenerated: tokGenerated,
					TotalDuration:   genDur.Round(time.Millisecond).String(),
					GenDuration:     genDur.Round(time.Millisecond).String(),
					TokensPerSec:    tps,
					TTFT:            ttft,
				}
			}
		}

		if bestTPS > 0 {
			report.Results = append(report.Results, bestResult)
		}
	}

	// Post-benchmark snapshot.
	snapAfter := native.TakeSnapshot()
	report.SnapshotAfter = snapAfter
	report.Delta = native.ComputeDelta(snapBefore, snapAfter)

	// Compute efficiency from best result.
	bestTPS := 0.0
	for _, r := range report.Results {
		if r.TokensPerSec > bestTPS {
			bestTPS = r.TokensPerSec
		}
	}
	report.Efficiency = native.ComputeEfficiency(bestTPS, report.Delta)

	// Output.
	if jsonOutput {
		data, _ := json.MarshalIndent(report, "", "  ")
		fmt.Println(string(data))
	} else {
		fmt.Println()
		report.PrintRich()
	}

	// Save JSON report.
	reportPath := ggufPath + ".benchmark.json"
	if err := report.SaveJSON(reportPath); err != nil && !jsonOutput {
		fmt.Fprintf(os.Stderr, "[iTaK Torch] Warning: could not save report: %v\n", err)
	}
}

// benchGOTensor runs the benchmark using the pure Go GOTensor engine.
func benchGOTensor(ggufPath string, modelSizeGB float64, mode string, jsonOutput bool, iterations, maxTokens int, snapBefore native.SystemSnapshot) {
	if !jsonOutput {
		fmt.Printf("[iTaK Torch] Loading model via GOTensor (pure Go)...\n")
	}

	loadStart := time.Now()
	engine, loadErr := native.NewNativeEngineFromGGUF(ggufPath)
	loadDur := time.Since(loadStart)
	if loadErr != nil {
		fmt.Fprintf(os.Stderr, "\nLoad failed: %v\n", loadErr)
		os.Exit(1)
	}
	defer engine.Close()

	if !jsonOutput {
		fmt.Printf("[iTaK Torch] GOTensor loaded in %v\n", loadDur)
	}

	// Enable sparse if requested.
	switch mode {
	case "sparse-70":
		engine.EnableSparse(0.7)
	case "sparse-90":
		engine.EnableSparse(0.9)
	}

	if !jsonOutput {
		fmt.Printf("[iTaK Torch] Running GOTensor benchmark (%d iteration(s))...\n\n", iterations)
	}

	var bestReport *native.BenchmarkReport
	bench := native.NewBenchmark(engine)

	for iter := 0; iter < iterations; iter++ {
		if iterations > 1 && !jsonOutput {
			fmt.Printf("--- Iteration %d/%d ---\n", iter+1, iterations)
		}
		report := bench.RunAll()
		report.LoadDuration = loadDur
		report.Mode = mode
		report.ModelSizeGB = modelSizeGB
		report.Backend = "gotensor"

		if bestReport == nil {
			bestReport = report
		} else {
			bestTPS := 0.0
			for _, r := range report.Results {
				if r.TokensPerSec > bestTPS {
					bestTPS = r.TokensPerSec
				}
			}
			currentBestTPS := 0.0
			for _, r := range bestReport.Results {
				if r.TokensPerSec > currentBestTPS {
					currentBestTPS = r.TokensPerSec
				}
			}
			if bestTPS > currentBestTPS {
				bestReport = report
			}
		}
	}

	// Capture final snapshots.
	snapAfter := native.TakeSnapshot()
	bestReport.SnapshotBefore = snapBefore
	bestReport.SnapshotAfter = snapAfter
	bestReport.Delta = native.ComputeDelta(snapBefore, snapAfter)

	// Output.
	if jsonOutput {
		data, _ := json.MarshalIndent(bestReport, "", "  ")
		fmt.Println(string(data))
	} else {
		fmt.Println()
		bestReport.PrintRich()
	}

	// Save JSON report.
	reportPath := ggufPath + ".benchmark.json"
	if err := bestReport.SaveJSON(reportPath); err != nil && !jsonOutput {
		fmt.Fprintf(os.Stderr, "[iTaK Torch] Warning: could not save report: %v\n", err)
	}
}


// cmdChat provides an interactive REPL with optional benchmark output.
// Usage: torch chat <model.gguf> [--bench] [--sparse N]
func cmdChat(args []string) {
	if len(args) < 1 {
		fmt.Fprintln(os.Stderr, "Usage: torch chat <path-to-gguf> [--bench] [--sparse 70] [--gpu]")
		os.Exit(1)
	}

	ggufPath := args[0]
	benchMode := false
	sparsity := float32(0)
	useGPU := false

	// Parse optional flags.
	for i := 1; i < len(args); i++ {
		switch args[i] {
		case "--bench", "-b":
			benchMode = true
		case "--gpu", "-g":
			useGPU = true
		case "--sparse", "-s":
			if i+1 < len(args) {
				i++
				var s float64
				fmt.Sscanf(args[i], "%f", &s)
				sparsity = float32(s / 100.0)
			}
		}
	}

	// Verify file exists.
	info, err := os.Stat(ggufPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("\033[36m[iTaK Torch] Loading %s (%.1f GB)...\033[0m\n",
		filepath.Base(ggufPath), float64(info.Size())/(1024*1024*1024))

	loadStart := time.Now()
	engine, err := native.NewNativeEngineFromGGUF(ggufPath)
	loadDur := time.Since(loadStart)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Load failed: %v\n", err)
		os.Exit(1)
	}
	defer engine.Close()

	// Set benchmark mode.
	engine.BenchMode = benchMode
	engine.SetLoadDuration(loadDur)

	// Enable GPU compute if requested.
	if useGPU {
		engine.UseGPU()
	}

	// Enable sparse if requested.
	if sparsity > 0 {
		engine.EnableSparse(sparsity)
	}

	fmt.Printf("\033[32mLoaded in %v. Type your message (or 'quit' to exit).\033[0m\n", loadDur)
	if benchMode {
		fmt.Println("\033[33mBenchmark mode ON - metrics shown after each response.\033[0m")
	}
	fmt.Println()

	// Interactive REPL.
	scanner := bufio.NewScanner(os.Stdin)
	for {
		fmt.Print("\033[36myou> \033[0m")
		if !scanner.Scan() {
			break
		}
		input := scanner.Text()
		if input == "" {
			continue
		}
		if input == "quit" || input == "exit" {
			break
		}

		// Special commands.
		if input == "/bench" {
			engine.BenchMode = !engine.BenchMode
			if engine.BenchMode {
				fmt.Println("\033[33mBenchmark mode ON\033[0m")
			} else {
				fmt.Println("\033[33mBenchmark mode OFF\033[0m")
			}
			continue
		}
		if len(input) > 8 && input[:8] == "/sparse " {
			var s float64
			fmt.Sscanf(input[8:], "%f", &s)
			engine.EnableSparse(float32(s / 100.0))
			continue
		}
		if input == "/dense" {
			engine.SparseConfig.Enabled = false
			fmt.Println("\033[33mSwitched to dense mode\033[0m")
			continue
		}
		if input == "/stats" {
			stats := engine.GetStats()
			fmt.Printf("Requests: %d, Tokens generated: %d\n", stats.TotalRequests, stats.TotalTokensGen)
			if stats.LastMetrics != nil {
				fmt.Printf("Last: %.1f tok/s, %v total\n", stats.LastMetrics.TokensPerSecond, stats.LastMetrics.TotalDuration)
			}
			continue
		}

		// Run inference.
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
		output, err := engine.Complete(ctx, []native.ChatMessage{
			{Role: "user", Content: input},
		}, native.CompletionParams{MaxTokens: 64})
		cancel()

		if err != nil {
			fmt.Printf("\033[31mError: %v\033[0m\n", err)
		} else {
			fmt.Printf("\033[32mmodel> \033[0m%s\n", output)
		}
		fmt.Println()
	}

	fmt.Println("\033[36mGoodbye!\033[0m")
}

// ---------- scan ----------

func getDrives() []string {
	var drives []string
	if runtime.GOOS == "windows" {
		for _, drive := range "ABCDEFGHIJKLMNOPQRSTUVWXYZ" {
			path := string(drive) + ":\\"
			if _, err := os.Stat(path); err == nil {
				drives = append(drives, path)
			}
		}
	} else {
		drives = append(drives, "/")
	}
	return drives
}

func cmdScan(args []string) {
	var targets []string
	if len(args) == 0 {
		targets = getDrives()
		fmt.Printf("[iTaK Torch] No path provided. Auto-detecting drives to scan: %v\n", targets)
		fmt.Println("[iTaK Torch] This full-system scan may take a few minutes...")
	} else {
		targets = []string{args[0]}
	}

	foundDirs := make(map[string]bool)
	modelCount := 0

	for _, startDir := range targets {
		fmt.Printf("[iTaK Torch] Scanning for .gguf files in: %s\n", startDir)
		
		err := filepath.WalkDir(startDir, func(path string, d os.DirEntry, err error) error {
			if err != nil {
				if d != nil && d.IsDir() {
					return filepath.SkipDir
				}
				return nil
			}
			
			// Optional progress spinner (shows briefly)
			if modelCount%50 == 0 {
				fmt.Printf("\r\033[K[\033[36m...\033[0m] Scanning files... [ %s ]", d.Name())
			}

			nameLower := strings.ToLower(d.Name())
		if !d.IsDir() && (strings.HasSuffix(nameLower, ".gguf") || strings.HasSuffix(nameLower, ".safetensors")) {
			dir := filepath.Dir(path)
			if !foundDirs[dir] {
				foundDirs[dir] = true
				if strings.HasSuffix(nameLower, ".safetensors") {
					fmt.Printf("\r\033[K[\033[33mHF Safetensors\033[0m] Found model directory: %s\n", dir)
				} else {
					fmt.Printf("\r\033[K[\033[32miTaK Torch\033[0m] Found model directory: %s\n", dir)
				}
			}
			modelCount++
		}
		return nil
	})

		// Clear the status line
		fmt.Print("\r\033[K")

		if err != nil {
			fmt.Fprintf(os.Stderr, "Error scanning %s: %v\n", startDir, err)
		}
	}

	if len(foundDirs) == 0 {
		fmt.Println("No .gguf files found.")
		return
	}

	fmt.Printf("\n[iTaK Torch] Found %d models across %d directories.\n", modelCount, len(foundDirs))
	
	newDirs := 0
	for dir := range foundDirs {
		if err := torch.AddWatchedDir(dir); err != nil {
			fmt.Fprintf(os.Stderr, "Failed to watch directory %s: %v\n", dir, err)
		} else {
			newDirs++
		}
	}

	fmt.Printf("[iTaK Torch] Added %d new directories to your watched list.\n", newDirs)
	fmt.Println("You can now run 'torch list' to see all your discovered models.")
}

// ---------- disk management ----------

func cmdClean() {
	if err := torch.ClearWatchedDirs(); err != nil {
		fmt.Fprintf(os.Stderr, "Error clearing watched directories: %v\n", err)
		os.Exit(1)
	}
	fmt.Println("\033[32mSuccessfully cleared all watched directories.\033[0m")
	fmt.Println("Your 'torch list' is now reset to your default cache.")
}

func cmdRemove(args []string) {
	if len(args) == 0 {
		fmt.Fprintf(os.Stderr, "Usage: torch rm <model_name>\n")
		os.Exit(1)
	}

	modelName := args[0]
	mgr, err := torch.NewModelManager(defaultCacheDir)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to initialize model manager: %v\n", err)
		os.Exit(1)
	}

	path, err := mgr.GetPath(modelName)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Model not found: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("\033[31mWARNING: This will permanently delete the file at:\033[0m\n  %s\n", path)
	fmt.Print("Are you sure you want to delete this model? [y/N]: ")
	reader := bufio.NewReader(os.Stdin)
	ans, _ := reader.ReadString('\n')
	ans = strings.TrimSpace(strings.ToLower(ans))
	if ans != "y" && ans != "yes" {
		fmt.Println("Aborted.")
		return
	}

	if err := mgr.Remove(modelName); err != nil {
		fmt.Fprintf(os.Stderr, "Error deleting model: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("\033[32mSuccessfully deleted %s\033[0m\n", modelName)
}

func cmdNuke() {
	mgr, err := torch.NewModelManager(defaultCacheDir)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to initialize model manager: %v\n", err)
		os.Exit(1)
	}

	models, err := mgr.List()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error gathering models: %v\n", err)
		os.Exit(1)
	}

	if len(models) == 0 {
		fmt.Println("No models tracked. Nothing to delete.")
		return
	}

	fmt.Printf("\033[31;1mCRITICAL WARNING: You are about to permanently delete %d models from your hard drive.\033[0m\n", len(models))
	fmt.Println("This includes ALL models found across ALL scanned directories and your default cache.")
	fmt.Println("This action CANNOT BE UNDONE.")
	fmt.Print("Type 'NUKE' to confirm deletion of all models: ")
	
	reader := bufio.NewReader(os.Stdin)
	ans, _ := reader.ReadString('\n')
	if strings.TrimSpace(ans) != "NUKE" {
		fmt.Println("Aborted.")
		return
	}

	deleted := 0
	for _, m := range models {
		fmt.Printf("Deleting %s... ", m.Path)
		if err := os.Remove(m.Path); err != nil {
			fmt.Printf("Error: %v\n", err)
		} else {
			fmt.Println("Done.")
			deleted++
		}
	}
	fmt.Printf("\n\033[32mSuccessfully deleted %d models.\033[0m\n", deleted)
}

func cmdConvert(args []string) {
	if len(args) == 0 {
		fmt.Fprintf(os.Stderr, "Usage: torch convert <model_name>\n")
		fmt.Fprintf(os.Stderr, "Example: torch convert \"Qwen (HF)\"\n")
		os.Exit(1)
	}

	modelName := args[0]
	mgr, err := torch.NewModelManager(defaultCacheDir)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to initialize model manager: %v\n", err)
		os.Exit(1)
	}

	path, err := mgr.GetPath(modelName)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Model not found: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("[iTaK Torch] Starting manual conversion for: %s\n", modelName)
	fmt.Printf("[iTaK Torch] Target directory: %s\n", path)
	
	convertedPath, convErr := torch.AutoConvert(path, defaultCacheDir)
	if convErr != nil {
		fmt.Fprintf(os.Stderr, "Error during conversion: %v\n", convErr)
		os.Exit(1)
	}

	fmt.Printf("\n\033[32mSuccessfully converted model to GGUF format!\033[0m\n")
	fmt.Printf("File saved to: %s\n", convertedPath)
	fmt.Printf("You can now run: torch serve \"%s\"\n", filepath.Base(convertedPath))
}
