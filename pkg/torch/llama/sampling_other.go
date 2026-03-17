//go:build !windows

package llama

import (
	"unsafe"

	"github.com/ebitengine/purego"
	"github.com/jupiterrider/ffi"
)

// SamplerChainParams is only 1 byte (single uint8), so purego handles it fine.
// But use ffi for consistency and to handle the return value marshaling correctly.
var ffiSamplerChainParams = ffi.NewType(&ffi.TypePointer)

func loadSamplingFuncs(lib uintptr) error {
	var err error
	ffiLib := ffi.Lib{Addr: lib}

	var samplerChainDefaultParamsFFI ffi.Fun
	if samplerChainDefaultParamsFFI, err = ffiLib.Prep("llama_sampler_chain_default_params", &ffiSamplerChainParams); err != nil {
		return loadError("llama_sampler_chain_default_params", err)
	}
	samplerChainDefaultParamsFn = func() SamplerChainParams {
		var p SamplerChainParams
		samplerChainDefaultParamsFFI.Call(unsafe.Pointer(&p))
		return p
	}

	var samplerChainInitFFI ffi.Fun
	if samplerChainInitFFI, err = ffiLib.Prep("llama_sampler_chain_init", &ffi.TypePointer, &ffiSamplerChainParams); err != nil {
		return loadError("llama_sampler_chain_init", err)
	}
	samplerChainInitFn = func(params SamplerChainParams) Sampler {
		var p Sampler
		samplerChainInitFFI.Call(unsafe.Pointer(&p), unsafe.Pointer(&params))
		return p
	}

	// samplerInitDry: 8 scalar args, fits in registers fine
	var samplerInitDryRaw func(vocab Vocab, nCtxTrain int32, multiplier float32, base float32, allowedLength int32, penaltyLast int32, seqBreakers uintptr, numBreakers uint64) Sampler
	purego.RegisterLibFunc(&samplerInitDryRaw, lib, "llama_sampler_init_dry")
	samplerInitDryFn = func(vocab Vocab, nCtxTrain int32, multiplier float32, base float32, allowedLength int32, penaltyLast int32, seqBreakers unsafe.Pointer, numBreakers uint64) Sampler {
		return samplerInitDryRaw(vocab, nCtxTrain, multiplier, base, allowedLength, penaltyLast, uintptr(seqBreakers), numBreakers)
	}

	// samplerInitGrammarLazyPatterns: 7 scalar args, fits in registers
	var samplerInitGrammarLazyPatternsRaw func(vocab Vocab, grammar *byte, root *byte, triggerPatterns uintptr, numPatterns uint64, triggerTokens uintptr, numTokens uint64) Sampler
	purego.RegisterLibFunc(&samplerInitGrammarLazyPatternsRaw, lib, "llama_sampler_init_grammar_lazy_patterns")
	samplerInitGrammarLazyPatternsFn = func(vocab Vocab, grammar *byte, root *byte, triggerPatterns unsafe.Pointer, numPatterns uint64, triggerTokens unsafe.Pointer, numTokens uint64) Sampler {
		return samplerInitGrammarLazyPatternsRaw(vocab, grammar, root, uintptr(triggerPatterns), numPatterns, uintptr(triggerTokens), numTokens)
	}

	loadSamplingPuregoFuncs(lib)
	return nil
}
