// benchmark.go implements automated performance benchmarking for the Torch engine.
//
// WHAT: Runs a series of inference benchmarks (prompt processing, token
// generation, different context lengths) and reports metrics like tokens/sec,
// latency, memory usage, and throughput.
//
// USAGE:
//   bench := NewBenchmark(engine)
//   results := bench.RunAll()
//   results.Print()
//   results.SaveJSON("benchmark_results.json")
package native

import (
	"encoding/json"
	"fmt"
	"os"
	"runtime"
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
	Name           string  `json:"name"`
	TokensGenerated int   `json:"tokens_generated"`
	PromptTokens    int   `json:"prompt_tokens"`
	TotalDuration   string `json:"total_duration"`
	PromptDuration  string `json:"prompt_duration"`
	GenDuration     string `json:"gen_duration"`
	TokensPerSec    float64 `json:"tokens_per_sec"`
	PromptTPS       float64 `json:"prompt_tokens_per_sec"`
	MemoryMB        float64 `json:"memory_mb"`
}

// BenchmarkSuite holds all benchmark results.
type BenchmarkSuite struct {
	Model      string            `json:"model"`
	Engine     string            `json:"engine"`
	Platform   string            `json:"platform"`
	Timestamp  string            `json:"timestamp"`
	Features   map[string]bool   `json:"features"`
	Results    []BenchmarkResult `json:"results"`
}

// RunAll executes all benchmarks and returns the suite.
func (b *Benchmark) RunAll() *BenchmarkSuite {
	suite := &BenchmarkSuite{
		Model:     b.engine.ModelName(),
		Engine:    "iTaK Torch Native",
		Platform:  fmt.Sprintf("%s/%s", runtime.GOOS, runtime.GOARCH),
		Timestamp: time.Now().Format(time.RFC3339),
		Features:  b.engine.FeatureStatus(),
	}

	fmt.Println("=== iTaK Torch Benchmark Suite ===")
	fmt.Printf("Model: %s\n", suite.Model)
	fmt.Printf("Platform: %s\n", suite.Platform)
	fmt.Println()

	// Benchmark 1: Short prompt, short generation.
	suite.Results = append(suite.Results, b.runBench("short_prompt_short_gen", 32, 32))

	// Benchmark 2: Medium prompt, medium generation.
	suite.Results = append(suite.Results, b.runBench("medium_prompt_medium_gen", 128, 64))

	// Benchmark 3: Long prompt, short generation (prompt processing speed).
	suite.Results = append(suite.Results, b.runBench("long_prompt_short_gen", 512, 16))

	// Benchmark 4: Short prompt, long generation (generation speed).
	suite.Results = append(suite.Results, b.runBench("short_prompt_long_gen", 16, 256))

	// Benchmark 5: Batch prefill stress test.
	suite.Results = append(suite.Results, b.runBench("batch_prefill_stress", 1024, 8))

	fmt.Println("\n=== Benchmark Complete ===")
	return suite
}

// runBench executes a single benchmark with given prompt and generation lengths.
func (b *Benchmark) runBench(name string, promptLen, genLen int) BenchmarkResult {
	fmt.Printf("Running: %s (prompt=%d, gen=%d)...", name, promptLen, genLen)

	// Create synthetic prompt tokens.
	prompt := make([]int, promptLen)
	for i := range prompt {
		prompt[i] = (i * 7) % b.engine.vocabSize
	}

	// Force GC before measurement.
	runtime.GC()
	var memBefore runtime.MemStats
	runtime.ReadMemStats(&memBefore)

	// Run forward pass for prompt.
	promptStart := time.Now()
	_ = b.engine.forward(prompt)
	promptDur := time.Since(promptStart)

	// Run generation.
	genStart := time.Now()
	context := make([]int, len(prompt))
	copy(context, prompt)

	for i := 0; i < genLen; i++ {
		logits := b.engine.forward(context)
		// Greedy decode.
		best := 0
		for j := 1; j < len(logits); j++ {
			if logits[j] > logits[best] {
				best = j
			}
		}
		context = append(context, best)
	}
	genDur := time.Since(genStart)
	totalDur := promptDur + genDur

	// Memory measurement.
	var memAfter runtime.MemStats
	runtime.ReadMemStats(&memAfter)
	memDeltaMB := float64(memAfter.HeapAlloc-memBefore.HeapAlloc) / (1024 * 1024)

	// Calculate metrics.
	promptTPS := float64(promptLen) / promptDur.Seconds()
	genTPS := float64(0)
	if genDur.Seconds() > 0 {
		genTPS = float64(genLen) / genDur.Seconds()
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
	}

	fmt.Printf(" %.1f tok/s (prompt: %.1f tok/s)\n", genTPS, promptTPS)
	return result
}

// Print displays the benchmark results in a formatted table.
func (s *BenchmarkSuite) Print() {
	fmt.Printf("\n%-30s %8s %8s %10s %10s %8s\n",
		"Benchmark", "Prompt", "Gen", "Prompt/s", "Gen/s", "Mem MB")
	fmt.Println(dashLine(80))

	for _, r := range s.Results {
		fmt.Printf("%-30s %8d %8d %10.1f %10.1f %8.1f\n",
			r.Name, r.PromptTokens, r.TokensGenerated,
			r.PromptTPS, r.TokensPerSec, r.MemoryMB)
	}
}

// SaveJSON writes results to a JSON file.
func (s *BenchmarkSuite) SaveJSON(path string) error {
	data, err := json.MarshalIndent(s, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal results: %w", err)
	}
	if err := os.WriteFile(path, data, 0644); err != nil {
		return fmt.Errorf("write %s: %w", path, err)
	}
	fmt.Printf("[Benchmark] Results saved to %s\n", path)
	return nil
}

func dashLine(n int) string {
	b := make([]byte, n)
	for i := range b {
		b[i] = '-'
	}
	return string(b)
}
