// warmup_engine.go pre-warms GPU kernels, KV cache, and compute graphs
// during engine startup to eliminate cold-start latency.
//
// WHAT: The first inference request after startup is always slow because:
//   - GPU kernels need to be compiled/loaded (JIT compilation)
//   - CUDA contexts need initialization
//   - KV cache pages need first-touch allocation
//   - CUDA graphs need capture before replay
//
// HOW: Run dummy inference at startup with representative prompt lengths
// (128, 512, 2048 tokens) to trigger all lazy initializations. This
// moves the "cold start" penalty from the first user request to boot time.
//
// GAIN: Eliminates 2-5 second first-request latency spike. Critical for
// production services behind load balancers with health check timeouts.
package native

import (
	"fmt"
	"time"
)

// WarmupConfig configures engine warmup.
type WarmupConfig struct {
	PromptLengths  []int  // Prompt sizes to warmup with (default: [1, 128, 512, 2048])
	GenerateTokens int    // Tokens to generate per warmup (default: 16)
	WarmBatchSizes []int  // Batch sizes to warmup (default: [1, 4, 8])
	WarmGraphs     bool   // Capture CUDA graphs during warmup (default: true)
	WarmKVCache    bool   // Pre-touch KV cache pages (default: true)
	MaxWarmupTime  time.Duration // Abort if warmup takes too long (default: 30s)
}

// DefaultWarmupConfig returns standard warmup settings.
func DefaultWarmupConfig() WarmupConfig {
	return WarmupConfig{
		PromptLengths:  []int{1, 128, 512, 2048},
		GenerateTokens: 16,
		WarmBatchSizes: []int{1, 4, 8},
		WarmGraphs:     true,
		WarmKVCache:    true,
		MaxWarmupTime:  30 * time.Second,
	}
}

// WarmupEngine manages the warmup process.
type WarmupEngine struct {
	config  WarmupConfig
	results []WarmupResult
}

// WarmupResult records one warmup pass.
type WarmupResult struct {
	Phase       string
	PromptLen   int
	BatchSize   int
	Duration    time.Duration
	TokPerSec   float64
	Success     bool
	Error       string
}

// NewWarmupEngine creates a warmup engine.
func NewWarmupEngine(config WarmupConfig) *WarmupEngine {
	return &WarmupEngine{
		config: config,
	}
}

// RunWarmup executes the full warmup sequence.
// inferFn: function that runs inference (promptLen, batchSize) -> (tokens, error).
// graphCaptureFn: function that captures CUDA graphs for a given batch size.
// kvAllocFn: function that pre-allocates KV cache pages for a given sequence length.
func (we *WarmupEngine) RunWarmup(
	inferFn func(promptLen, batchSize, maxTokens int) (int, error),
	graphCaptureFn func(batchSize int) error,
	kvAllocFn func(seqLen int) error,
) error {
	startTime := time.Now()
	deadline := startTime.Add(we.config.MaxWarmupTime)

	fmt.Println("[Warmup] Starting engine warmup...")

	// Phase 1: KV cache pre-touch.
	if we.config.WarmKVCache {
		fmt.Println("[Warmup] Phase 1: KV cache pre-allocation")
		for _, promptLen := range we.config.PromptLengths {
			if time.Now().After(deadline) {
				return fmt.Errorf("warmup timeout after %s", we.config.MaxWarmupTime)
			}
			phaseStart := time.Now()
			var errStr string
			success := true
			if kvAllocFn != nil {
				if err := kvAllocFn(promptLen); err != nil {
					errStr = err.Error()
					success = false
				}
			}
			we.results = append(we.results, WarmupResult{
				Phase:     "kv_cache",
				PromptLen: promptLen,
				Duration:  time.Since(phaseStart),
				Success:   success,
				Error:     errStr,
			})
		}
	}

	// Phase 2: Inference warmup across batch sizes and prompt lengths.
	fmt.Println("[Warmup] Phase 2: Inference warmup")
	for _, batchSize := range we.config.WarmBatchSizes {
		for _, promptLen := range we.config.PromptLengths {
			if time.Now().After(deadline) {
				return fmt.Errorf("warmup timeout after %s", we.config.MaxWarmupTime)
			}

			phaseStart := time.Now()
			var errStr string
			success := true
			var tokPerSec float64

			if inferFn != nil {
				tokens, err := inferFn(promptLen, batchSize, we.config.GenerateTokens)
				if err != nil {
					errStr = err.Error()
					success = false
				} else {
					elapsed := time.Since(phaseStart).Seconds()
					if elapsed > 0 {
						tokPerSec = float64(tokens) / elapsed
					}
				}
			}

			we.results = append(we.results, WarmupResult{
				Phase:     "inference",
				PromptLen: promptLen,
				BatchSize: batchSize,
				Duration:  time.Since(phaseStart),
				TokPerSec: tokPerSec,
				Success:   success,
				Error:     errStr,
			})

			fmt.Printf("[Warmup]   batch=%d prompt=%d -> %.1f tok/s (%s)\n",
				batchSize, promptLen, tokPerSec, time.Since(phaseStart))
		}
	}

	// Phase 3: CUDA graph capture.
	if we.config.WarmGraphs {
		fmt.Println("[Warmup] Phase 3: CUDA graph capture")
		for _, batchSize := range we.config.WarmBatchSizes {
			if time.Now().After(deadline) {
				return fmt.Errorf("warmup timeout after %s", we.config.MaxWarmupTime)
			}

			phaseStart := time.Now()
			var errStr string
			success := true

			if graphCaptureFn != nil {
				if err := graphCaptureFn(batchSize); err != nil {
					errStr = err.Error()
					success = false
				}
			}

			we.results = append(we.results, WarmupResult{
				Phase:     "cuda_graph",
				BatchSize: batchSize,
				Duration:  time.Since(phaseStart),
				Success:   success,
				Error:     errStr,
			})
		}
	}

	totalTime := time.Since(startTime)
	fmt.Printf("[Warmup] Complete in %s (%d phases)\n", totalTime, len(we.results))

	return nil
}

// Report returns a summary of the warmup results.
func (we *WarmupEngine) Report() map[string]interface{} {
	successes := 0
	failures := 0
	var totalDuration time.Duration
	var maxTokPerSec float64

	for _, r := range we.results {
		totalDuration += r.Duration
		if r.Success {
			successes++
		} else {
			failures++
		}
		if r.TokPerSec > maxTokPerSec {
			maxTokPerSec = r.TokPerSec
		}
	}

	return map[string]interface{}{
		"phases_run":    len(we.results),
		"successes":     successes,
		"failures":      failures,
		"total_time":    totalDuration.String(),
		"peak_tok_sec":  fmt.Sprintf("%.1f", maxTokPerSec),
	}
}
