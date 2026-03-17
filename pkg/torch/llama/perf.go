package llama

import (
	"fmt"

	"github.com/ebitengine/purego"
)

// PerfContextData represents the C struct llama_perf_context_data
type PerfContextData struct {
	TStartMs      float64 // absolute start time
	TLoadMs       float64 // time needed for loading the model
	TPromptEvalMs float64 // time needed for processing the prompt
	TEvalMs       float64 // time needed for generating tokens

	NPEval  int32 // number of prompt tokens
	NEval   int32 // number of generated tokens
	NReused int32 // number of times a ggml compute graph had been reused
}

// String returns a formatted string representation of PerfContextData
func (p PerfContextData) String() string {
	return fmt.Sprintf("PerfContextData{Start: %.2fms, Load: %.2fms, Prompt Eval: %.2fms, Eval: %.2fms, Prompt Tokens: %d, Gen Tokens: %d, Reused: %d}",
		p.TStartMs, p.TLoadMs, p.TPromptEvalMs, p.TEvalMs, p.NPEval, p.NEval, p.NReused)
}

// PerfSamplerData represents the C struct llama_perf_sampler_data
type PerfSamplerData struct {
	TSampleMs float64 // time needed for sampling in ms

	NSample int32 // number of sampled tokens
}

// String returns a formatted string representation of PerfSamplerData
func (p PerfSamplerData) String() string {
	return fmt.Sprintf("PerfSamplerData{Sample Time: %.2fms, Samples: %d}", p.TSampleMs, p.NSample)
}

// Typed Go function variables - filled by platform-specific loaders
var (
	perfContextFn func(ctx Context) PerfContextData
	perfSamplerFn func(chain Sampler) PerfSamplerData
)

// purego direct-call function pointers (no struct args)
var (
	perfContextPrintFn     func(ctx Context)
	perfSamplerPrintFn     func(chain Sampler)
	perfSamplerResetFn     func(chain Sampler)
	memoryBreakdownPrintFn func(ctx Context)
)

func loadPerfPuregoFuncs(lib uintptr) {
	purego.RegisterLibFunc(&perfContextPrintFn, lib, "llama_perf_context_print")
	purego.RegisterLibFunc(&perfSamplerPrintFn, lib, "llama_perf_sampler_print")
	purego.RegisterLibFunc(&perfSamplerResetFn, lib, "llama_perf_sampler_reset")
	purego.RegisterLibFunc(&memoryBreakdownPrintFn, lib, "llama_memory_breakdown_print")
}

// PerfContext returns performance data for the model context.
func PerfContext(ctx Context) PerfContextData {
	if ctx == 0 {
		return PerfContextData{}
	}
	return perfContextFn(ctx)
}

// PerfSampler returns performance data for the sampler.
func PerfSampler(chain Sampler) PerfSamplerData {
	if chain == 0 {
		return PerfSamplerData{}
	}
	return perfSamplerFn(chain)
}

// PerfSamplerReset resets sampler performance metrics.
func PerfSamplerReset(chain Sampler) {
	if chain == 0 {
		return
	}
	perfSamplerResetFn(chain)
}

// PerfContextPrint prints performance data for the model context.
func PerfContextPrint(ctx Context) {
	if ctx == 0 {
		return
	}
	perfContextPrintFn(ctx)
}

// PerfSamplerPrint prints performance data for the sampler.
func PerfSamplerPrint(chain Sampler) {
	if chain == 0 {
		return
	}
	perfSamplerPrintFn(chain)
}

// MemoryBreakdownPrint prints a breakdown of per-device memory use.
func MemoryBreakdownPrint(ctx Context) {
	if ctx == 0 {
		return
	}
	memoryBreakdownPrintFn(ctx)
}
