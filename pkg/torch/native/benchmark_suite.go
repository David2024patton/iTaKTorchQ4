// benchmark_suite.go provides a comprehensive benchmarking framework for
// measuring inference and training performance.
//
// WHAT: Standardized benchmarks that measure:
//   - Token generation throughput (tok/s)
//   - Time to first token (TTFT)
//   - Prefill throughput (prompt tok/s)
//   - Memory usage (VRAM, RAM)
//   - Training step time
//   - Gradient computation time
//   - Optimizer step time
//
// WHY: You can't improve what you don't measure. These benchmarks provide
// the baseline numbers to validate that each optimization actually helps
// and to compare against Ollama/vLLM/llama.cpp.
package native

import (
	"fmt"
	"math"
	"runtime"
	"sort"
	"time"
)

// BenchSuiteResult holds the results of one benchmark run.
type BenchSuiteResult struct {
	Name       string
	Iterations int
	TotalTime  time.Duration
	MinTime    time.Duration
	MaxTime    time.Duration
	AvgTime    time.Duration
	P50Time    time.Duration
	P95Time    time.Duration
	P99Time    time.Duration
	Throughput float64 // Operations per second
	Unit       string  // e.g., "tok/s", "samples/s"
}

// BenchSuite manages a collection of benchmarks.
type BenchSuite struct {
	results []BenchSuiteResult
}

// NewBenchSuite creates a new suite.
func NewBenchSuite() *BenchSuite {
	return &BenchSuite{}
}

// Run executes a benchmark function N times and records statistics.
// fn should return the number of "units" processed (e.g., tokens generated).
func (bs *BenchSuite) Run(name string, iterations int, unit string, fn func() int) BenchSuiteResult {
	// Warmup.
	for i := 0; i < 3; i++ {
		fn()
	}
	runtime.GC()

	latencies := make([]time.Duration, iterations)
	var totalUnits int

	for i := 0; i < iterations; i++ {
		start := time.Now()
		units := fn()
		latencies[i] = time.Since(start)
		totalUnits += units
	}

	// Sort for percentile calculation.
	sort.Slice(latencies, func(i, j int) bool { return latencies[i] < latencies[j] })

	totalTime := time.Duration(0)
	for _, l := range latencies {
		totalTime += l
	}

	result := BenchSuiteResult{
		Name:       name,
		Iterations: iterations,
		TotalTime:  totalTime,
		MinTime:    latencies[0],
		MaxTime:    latencies[len(latencies)-1],
		AvgTime:    totalTime / time.Duration(iterations),
		P50Time:    latencies[int(float64(iterations)*0.50)],
		P95Time:    latencies[int(math.Min(float64(iterations)*0.95, float64(iterations-1)))],
		P99Time:    latencies[int(math.Min(float64(iterations)*0.99, float64(iterations-1)))],
		Unit:       unit,
	}

	// Compute throughput.
	if totalTime > 0 {
		result.Throughput = float64(totalUnits) / totalTime.Seconds()
	}

	bs.results = append(bs.results, result)
	return result
}

// RunInference benchmarks token generation throughput.
func (bs *BenchSuite) RunInference(
	name string,
	iterations int,
	generateFn func(promptLen, maxTokens int) (int, time.Duration), // Returns (tokens_generated, ttft)
	promptLen int,
	maxTokens int,
) BenchSuiteResult {
	var ttftSum time.Duration
	var ttftCount int

	result := bs.Run(name, iterations, "tok/s", func() int {
		tokensGenerated, ttft := generateFn(promptLen, maxTokens)
		ttftSum += ttft
		ttftCount++
		return tokensGenerated
	})

	// Add TTFT to the result name.
	if ttftCount > 0 {
		avgTTFT := ttftSum / time.Duration(ttftCount)
		result.Name = fmt.Sprintf("%s (TTFT: %s)", result.Name, avgTTFT)
	}

	return result
}

// RunTraining benchmarks training step throughput.
func (bs *BenchSuite) RunTraining(
	name string,
	iterations int,
	trainStepFn func(batchSize, seqLen int) float32, // Returns loss
	batchSize int,
	seqLen int,
) BenchSuiteResult {
	var lossSum float64

	result := bs.Run(name, iterations, "samples/s", func() int {
		loss := trainStepFn(batchSize, seqLen)
		lossSum += float64(loss)
		return batchSize
	})

	avgLoss := lossSum / float64(iterations)
	result.Name = fmt.Sprintf("%s (avg loss: %.4f)", result.Name, avgLoss)
	return result
}

// RunMemory benchmarks memory allocation and patterns.
func (bs *BenchSuite) RunMemory(
	name string,
	allocFn func() int64, // Returns bytes allocated
) BenchSuiteResult {
	var memBefore, memAfter runtime.MemStats
	runtime.GC()
	runtime.ReadMemStats(&memBefore)

	totalBytes := allocFn()

	runtime.ReadMemStats(&memAfter)
	heapGrowth := int64(memAfter.HeapAlloc - memBefore.HeapAlloc)

	result := BenchSuiteResult{
		Name: fmt.Sprintf("%s (alloc: %s, heap growth: %s)",
			name,
			formatBytes(totalBytes),
			formatBytes(heapGrowth)),
		Iterations: 1,
		Unit:       "bytes",
	}

	bs.results = append(bs.results, result)
	return result
}

// Report generates a formatted benchmark report.
func (bs *BenchSuite) Report() string {
	var report string
	report += "╔══════════════════════════════════════════════════════════════════╗\n"
	report += "║                    iTaK Torch Benchmark Report                  ║\n"
	report += "╠══════════════════════════════════════════════════════════════════╣\n"

	for _, r := range bs.results {
		report += fmt.Sprintf("║ %-62s ║\n", r.Name)
		report += "╟──────────────────────────────────────────────────────────────────╢\n"

		if r.Iterations > 1 {
			report += fmt.Sprintf("║  Iterations: %-10d Throughput: %-10.1f %-16s ║\n",
				r.Iterations, r.Throughput, r.Unit)
			report += fmt.Sprintf("║  Avg: %-12s Min: %-12s Max: %-17s ║\n",
				r.AvgTime, r.MinTime, r.MaxTime)
			report += fmt.Sprintf("║  P50: %-12s P95: %-12s P99: %-17s ║\n",
				r.P50Time, r.P95Time, r.P99Time)
		}
		report += "╠══════════════════════════════════════════════════════════════════╣\n"
	}

	report += "╚══════════════════════════════════════════════════════════════════╝\n"
	return report
}

// CompareWith generates a comparison table against another suite (e.g., Ollama).
func (bs *BenchSuite) CompareWith(other *BenchSuite, otherName string) string {
	report := fmt.Sprintf("\n%-30s %-15s %-15s %-10s\n", "Benchmark", "Torch", otherName, "Speedup")
	report += fmt.Sprintf("%-30s %-15s %-15s %-10s\n", "─────────", "─────", "─────", "───────")

	for _, r := range bs.results {
		for _, o := range other.results {
			if r.Name == o.Name || r.Unit == o.Unit {
				speedup := r.Throughput / (o.Throughput + 0.001)
				speedupStr := fmt.Sprintf("%.2fx", speedup)
				if speedup > 1 {
					speedupStr = fmt.Sprintf("▲ %.2fx", speedup)
				} else if speedup < 1 {
					speedupStr = fmt.Sprintf("▼ %.2fx", speedup)
				}
				report += fmt.Sprintf("%-30s %-15.1f %-15.1f %-10s\n",
					r.Name, r.Throughput, o.Throughput, speedupStr)
				break
			}
		}
	}

	return report
}

// formatBytes formats byte counts to human-readable.
func formatBytes(b int64) string {
	if b < 1024 {
		return fmt.Sprintf("%d B", b)
	}
	if b < 1024*1024 {
		return fmt.Sprintf("%.1f KB", float64(b)/1024)
	}
	if b < 1024*1024*1024 {
		return fmt.Sprintf("%.1f MB", float64(b)/(1024*1024))
	}
	return fmt.Sprintf("%.1f GB", float64(b)/(1024*1024*1024))
}

// PredefinedBenchmarks returns a list of standard benchmarks to run.
func PredefinedBenchmarks() []string {
	return []string{
		"inference/throughput/128-prompt-128-gen",
		"inference/throughput/512-prompt-256-gen",
		"inference/throughput/2048-prompt-512-gen",
		"inference/ttft/128-prompt",
		"inference/ttft/2048-prompt",
		"inference/batch/4-concurrent",
		"inference/batch/16-concurrent",
		"training/step/batch4-seq512",
		"training/step/batch8-seq1024",
		"training/backward/cross-entropy",
		"training/backward/attention",
		"memory/kv-cache/4k-context",
		"memory/kv-cache/32k-context",
		"memory/quantization/int8-compression",
		"memory/quantization/int4-compression",
	}
}
