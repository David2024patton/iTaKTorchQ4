// benchmark.go implements automated performance benchmarking for the Torch engine.
//
// WHAT: Runs a series of inference benchmarks (prompt processing, token
// generation, different context lengths) and reports metrics like tokens/sec,
// latency percentiles (P50/P95/P99), TTFT, ITL, memory usage, GPU resources,
// and power efficiency ratios.
//
// WHY: The benchmark flag should show MORE data than Ollama's --verbose.
// Ollama gives you 3 lines: eval count, duration, rate. Torch gives you
// the full picture: model info, load time, generation speed, latency
// distribution, system resources, GPU metrics, and efficiency ratios.
//
// USAGE:
//
//	bench := NewBenchmark(engine)
//	results := bench.RunAll()
//	results.PrintRich()
//	results.SaveJSON("benchmark_results.json")
package native

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"runtime"
	"sort"
	"strings"
	"time"
)

// Benchmark runs performance tests against an engine.
type Benchmark struct {
	engine *NativeEngine
}

// NewBenchmark creates a benchmark runner.
func NewBenchmark(engine *NativeEngine) *Benchmark {
	return &Benchmark{engine: engine}
}

// BenchmarkResult holds results from one benchmark run.
type BenchmarkResult struct {
	Name            string  `json:"name"`
	TokensGenerated int     `json:"tokens_generated"`
	PromptTokens    int     `json:"prompt_tokens"`
	TotalDuration   string  `json:"total_duration"`
	PromptDuration  string  `json:"prompt_duration"`
	GenDuration     string  `json:"gen_duration"`
	TokensPerSec    float64 `json:"tokens_per_sec"`
	PromptTPS       float64 `json:"prompt_tokens_per_sec"`
	MemoryMB        float64 `json:"memory_mb"`

	// Latency tracking.
	TTFT           time.Duration   `json:"ttft_ns"`           // Time to first token
	ITLAvg         time.Duration   `json:"itl_avg_ns"`        // Average inter-token latency
	ITLPercentiles LatencyProfile  `json:"itl_percentiles"`   // P50/P95/P99
	TokenLatencies []time.Duration `json:"-"`                 // Raw per-token latencies
}

// LatencyProfile holds percentile latencies.
type LatencyProfile struct {
	P50 time.Duration `json:"p50_ns"`
	P95 time.Duration `json:"p95_ns"`
	P99 time.Duration `json:"p99_ns"`
}

// BenchmarkReport holds all benchmark results plus system context.
type BenchmarkReport struct {
	Model      string            `json:"model"`
	Engine     string            `json:"engine"`
	Platform   string            `json:"platform"`
	Timestamp  string            `json:"timestamp"`
	Features   map[string]bool   `json:"features"`
	Results    []BenchmarkResult `json:"results"`

	// System resource snapshots.
	LoadDuration time.Duration    `json:"load_duration_ns"`
	Mode         string           `json:"mode"`          // dense, sparse-70, etc.
	Backend      string           `json:"backend"`       // cpu, vulkan, cuda
	ModelSizeGB  float64          `json:"model_size_gb"`
	SnapshotBefore SystemSnapshot `json:"snapshot_before"`
	SnapshotAfter  SystemSnapshot `json:"snapshot_after"`
	Delta          SystemDelta    `json:"delta"`
	Efficiency     EfficiencyMetrics `json:"efficiency"`
}

// RunAll executes all benchmarks and returns the report.
func (b *Benchmark) RunAll() *BenchmarkReport {
	report := &BenchmarkReport{
		Model:     b.engine.ModelName(),
		Engine:    "iTaK Torch Native",
		Platform:  fmt.Sprintf("%s/%s", runtime.GOOS, runtime.GOARCH),
		Timestamp: time.Now().Format(time.RFC3339),
		Features:  b.engine.FeatureStatus(),
	}

	// Pre-benchmark system snapshot.
	report.SnapshotBefore = TakeSnapshot()

	// Benchmark 1: Short prompt, short generation.
	report.Results = append(report.Results, b.runBench("short_prompt_short_gen", 32, 32))

	// Benchmark 2: Medium prompt, medium generation.
	report.Results = append(report.Results, b.runBench("medium_prompt_medium_gen", 128, 64))

	// Benchmark 3: Long prompt, short generation (prompt processing speed).
	report.Results = append(report.Results, b.runBench("long_prompt_short_gen", 512, 16))

	// Benchmark 4: Short prompt, long generation (generation speed).
	report.Results = append(report.Results, b.runBench("short_prompt_long_gen", 16, 256))

	// Benchmark 5: Batch prefill stress test.
	report.Results = append(report.Results, b.runBench("batch_prefill_stress", 1024, 8))

	// Post-benchmark system snapshot.
	report.SnapshotAfter = TakeSnapshot()
	report.Delta = ComputeDelta(report.SnapshotBefore, report.SnapshotAfter)

	// Compute efficiency from the best benchmark result.
	bestTPS := 0.0
	for _, r := range report.Results {
		if r.TokensPerSec > bestTPS {
			bestTPS = r.TokensPerSec
		}
	}
	report.Efficiency = ComputeEfficiency(bestTPS, report.Delta)

	return report
}

// runBench executes a single benchmark with given prompt and generation lengths.
func (b *Benchmark) runBench(name string, promptLen, genLen int) BenchmarkResult {
	fmt.Printf("  Running: %s (prompt=%d, gen=%d)...", name, promptLen, genLen)

	// Create synthetic prompt tokens.
	prompt := make([]int, promptLen)
	for i := range prompt {
		prompt[i] = (i * 7) % b.engine.vocabSize
	}

	// Force GC before measurement.
	runtime.GC()
	var memBefore runtime.MemStats
	runtime.ReadMemStats(&memBefore)

	// Run forward pass for prompt (measures TTFT).
	promptStart := time.Now()
	_ = b.engine.forward(prompt)
	promptDur := time.Since(promptStart)

	// Run generation with per-token latency tracking.
	genStart := time.Now()
	context := make([]int, len(prompt))
	copy(context, prompt)

	tokenLatencies := make([]time.Duration, 0, genLen)
	for i := 0; i < genLen; i++ {
		tokenStart := time.Now()
		logits := b.engine.forward(context)

		// Greedy decode.
		best := 0
		for j := 1; j < len(logits); j++ {
			if logits[j] > logits[best] {
				best = j
			}
		}
		context = append(context, best)
		tokenLatencies = append(tokenLatencies, time.Since(tokenStart))
	}
	genDur := time.Since(genStart)
	totalDur := promptDur + genDur

	// Memory measurement.
	var memAfter runtime.MemStats
	runtime.ReadMemStats(&memAfter)
	memDeltaMB := float64(memAfter.HeapAlloc-memBefore.HeapAlloc) / (1024 * 1024)

	// Calculate core metrics.
	promptTPS := float64(promptLen) / promptDur.Seconds()
	genTPS := float64(0)
	if genDur.Seconds() > 0 {
		genTPS = float64(genLen) / genDur.Seconds()
	}

	// Calculate latency percentiles.
	itlProfile := computePercentiles(tokenLatencies)
	itlAvg := time.Duration(0)
	if len(tokenLatencies) > 0 {
		var totalLatency time.Duration
		for _, l := range tokenLatencies {
			totalLatency += l
		}
		itlAvg = totalLatency / time.Duration(len(tokenLatencies))
	}

	result := BenchmarkResult{
		Name:            name,
		TokensGenerated: genLen,
		PromptTokens:    promptLen,
		TotalDuration:   totalDur.Round(time.Millisecond).String(),
		PromptDuration:  promptDur.Round(time.Millisecond).String(),
		GenDuration:     genDur.Round(time.Millisecond).String(),
		TokensPerSec:    genTPS,
		PromptTPS:       promptTPS,
		MemoryMB:        memDeltaMB,
		TTFT:            promptDur,
		ITLAvg:          itlAvg,
		ITLPercentiles:  itlProfile,
		TokenLatencies:  tokenLatencies,
	}

	fmt.Printf(" %.1f tok/s (prompt: %.1f tok/s, TTFT: %s)\n", genTPS, promptTPS, promptDur.Round(time.Microsecond))
	return result
}

// computePercentiles calculates P50/P95/P99 from a list of latencies.
func computePercentiles(latencies []time.Duration) LatencyProfile {
	if len(latencies) == 0 {
		return LatencyProfile{}
	}

	sorted := make([]time.Duration, len(latencies))
	copy(sorted, latencies)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i] < sorted[j] })

	n := len(sorted)
	return LatencyProfile{
		P50: sorted[int(float64(n)*0.50)],
		P95: sorted[int(math.Min(float64(n)*0.95, float64(n-1)))],
		P99: sorted[int(math.Min(float64(n)*0.99, float64(n-1)))],
	}
}

// =============================================
// Output formatters
// =============================================

const (
	boxTL = "╔"
	boxTR = "╗"
	boxBL = "╚"
	boxBR = "╝"
	boxH  = "═"
	boxV  = "║"
	boxML = "╠"
	boxMR = "╣"
)

func boxLine(width int) string {
	return strings.Repeat(boxH, width)
}

func boxRow(content string, width int) string {
	padded := fmt.Sprintf("%-*s", width-4, content)
	if len(padded) > width-4 {
		padded = padded[:width-4]
	}
	return fmt.Sprintf("%s %s %s", boxV, padded, boxV)
}

func boxSection(title string, width int) string {
	titleLen := len(title)
	leftPad := (width - titleLen - 2) / 2
	rightPad := width - titleLen - 2 - leftPad
	return fmt.Sprintf("%s%s %s %s%s", boxML, strings.Repeat(boxH, leftPad), title, strings.Repeat(boxH, rightPad), boxMR)
}

// PrintRich outputs the full benchmark report with box-drawing art.
func (r *BenchmarkReport) PrintRich() {
	w := 68

	fmt.Println(boxTL + boxLine(w) + boxTR)
	fmt.Println(boxRow("iTaK Torch Benchmark Report", w))

	// Model section.
	fmt.Println(boxSection("MODEL", w))
	fmt.Println(boxRow(fmt.Sprintf("Model:     %s", r.Model), w))
	if r.ModelSizeGB > 0 {
		fmt.Println(boxRow(fmt.Sprintf("Size:      %.1f GB", r.ModelSizeGB), w))
	}
	fmt.Println(boxRow(fmt.Sprintf("Platform:  %s", r.Platform), w))
	if r.Mode != "" {
		fmt.Println(boxRow(fmt.Sprintf("Mode:      %s   Backend: %s", r.Mode, r.Backend), w))
	}

	// Load section.
	if r.LoadDuration > 0 {
		fmt.Println(boxSection("LOAD", w))
		fmt.Println(boxRow(fmt.Sprintf("Load Time: %s", r.LoadDuration.Round(time.Millisecond)), w))
	}

	// Generation section.
	fmt.Println(boxSection("GENERATION", w))
	for _, res := range r.Results {
		fmt.Println(boxRow(fmt.Sprintf("%-30s  %8.1f tok/s  prompt: %8.1f tok/s",
			res.Name, res.TokensPerSec, res.PromptTPS), w))
	}

	// Find the best result for the summary line.
	if len(r.Results) > 0 {
		best := r.Results[0]
		for _, res := range r.Results[1:] {
			if res.TokensPerSec > best.TokensPerSec {
				best = res
			}
		}
		fmt.Println(boxRow("", w))
		fmt.Println(boxRow(fmt.Sprintf("Best:      %.1f tok/s (%s)", best.TokensPerSec, best.Name), w))
		fmt.Println(boxRow(fmt.Sprintf("TTFT:      %s   ITL avg: %s",
			best.TTFT.Round(time.Microsecond), best.ITLAvg.Round(time.Microsecond)), w))
		fmt.Println(boxRow(fmt.Sprintf("P50:       %s   P95: %s   P99: %s",
			best.ITLPercentiles.P50.Round(time.Microsecond),
			best.ITLPercentiles.P95.Round(time.Microsecond),
			best.ITLPercentiles.P99.Round(time.Microsecond)), w))
	}

	// System resources section.
	fmt.Println(boxSection("SYSTEM RESOURCES", w))
	fmt.Println(boxRow(fmt.Sprintf("CPU:       %s (%dC/%dT)",
		r.Delta.Before.CPUName, r.Delta.Before.CPUCores, r.Delta.Before.CPUThreads), w))

	if r.Delta.Before.RAMTotal > 0 {
		fmt.Println(boxRow(fmt.Sprintf("RAM:       %.1f -> %.1f GB (delta: %+d MB)",
			float64(r.Delta.Before.RAMUsed)/(1024*1024*1024),
			float64(r.Delta.After.RAMUsed)/(1024*1024*1024),
			r.Delta.RAMDeltaMB), w))
	}

	fmt.Println(boxRow(fmt.Sprintf("Heap:      %.1f -> %.1f MB (delta: %+.1f MB)",
		float64(r.Delta.Before.HeapAllocBytes)/(1024*1024),
		float64(r.Delta.After.HeapAllocBytes)/(1024*1024),
		r.Delta.HeapDeltaMB), w))

	fmt.Println(boxRow(fmt.Sprintf("GC Pause:  %.2fms   Goroutines: %d",
		float64(r.Delta.After.GCPauseNs)/1e6,
		r.Delta.After.Goroutines), w))

	// GPU section (only if available).
	if r.Delta.Before.GPUAvailable {
		fmt.Println(boxSection("GPU", w))
		fmt.Println(boxRow(fmt.Sprintf("Device:    %s", r.Delta.Before.GPUName), w))
		fmt.Println(boxRow(fmt.Sprintf("VRAM:      %d -> %d MB (delta: %+d MB)",
			r.Delta.Before.GPUVRAMUsedMB, r.Delta.After.GPUVRAMUsedMB, r.Delta.VRAMDeltaMB), w))
		fmt.Println(boxRow(fmt.Sprintf("Util:      %d%% -> %d%%   Temp: %d -> %dC",
			r.Delta.Before.GPUUtilPct, r.Delta.After.GPUUtilPct,
			r.Delta.Before.GPUTempC, r.Delta.After.GPUTempC), w))
		fmt.Println(boxRow(fmt.Sprintf("Power:     %.1f -> %.1fW   Clock: %d MHz",
			r.Delta.Before.GPUPowerW, r.Delta.After.GPUPowerW, r.Delta.After.GPUClockMHz), w))
		if r.Delta.After.GPUThrottled > 0 {
			fmt.Println(boxRow(fmt.Sprintf("THROTTLE:  %d events detected!", r.Delta.After.GPUThrottled), w))
		}
	}

	// Efficiency section.
	if r.Efficiency.TokPerSecPerWatt > 0 || r.Efficiency.TokPerSecPerGBVRAM > 0 {
		fmt.Println(boxSection("EFFICIENCY", w))
		if r.Efficiency.TokPerSecPerWatt > 0 {
			fmt.Println(boxRow(fmt.Sprintf("tok/s/W:   %.2f", r.Efficiency.TokPerSecPerWatt), w))
		}
		if r.Efficiency.TokPerSecPerGBVRAM > 0 {
			fmt.Println(boxRow(fmt.Sprintf("tok/s/GB:  %.2f (per GB VRAM)", r.Efficiency.TokPerSecPerGBVRAM), w))
		}
	}

	fmt.Println(boxBL + boxLine(w) + boxBR)
}

// Print displays the benchmark results in the original compact table format.
func (r *BenchmarkReport) Print() {
	fmt.Printf("\n%-30s %8s %8s %10s %10s %8s\n",
		"Benchmark", "Prompt", "Gen", "Prompt/s", "Gen/s", "Mem MB")
	fmt.Println(strings.Repeat("-", 80))

	for _, res := range r.Results {
		fmt.Printf("%-30s %8d %8d %10.1f %10.1f %8.1f\n",
			res.Name, res.PromptTokens, res.TokensGenerated,
			res.PromptTPS, res.TokensPerSec, res.MemoryMB)
	}
}

// SaveJSON writes results to a JSON file.
func (r *BenchmarkReport) SaveJSON(path string) error {
	data, err := json.MarshalIndent(r, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal results: %w", err)
	}
	if err := os.WriteFile(path, data, 0644); err != nil {
		return fmt.Errorf("write %s: %w", path, err)
	}
	fmt.Printf("[Benchmark] Results saved to %s\n", path)
	return nil
}
