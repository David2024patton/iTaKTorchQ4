//go:build windows

package llama

import (
	"unsafe"

	"github.com/jupiterrider/ffi"
)

var (
	ffiSamplerChainParams = ffi.NewType(&ffi.TypePointer)
)

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

	var samplerInitDryFFI ffi.Fun
	if samplerInitDryFFI, err = ffiLib.Prep("llama_sampler_init_dry", &ffi.TypePointer, &ffi.TypePointer, &ffi.TypeSint32, &ffi.TypeFloat, &ffi.TypeFloat,
		&ffi.TypeSint32, &ffi.TypeSint32, &ffi.TypePointer, &ffiTypeSize); err != nil {
		return loadError("llama_sampler_init_dry", err)
	}
	samplerInitDryFn = func(vocab Vocab, nCtxTrain int32, multiplier float32, base float32, allowedLength int32, penaltyLast int32, seqBreakers unsafe.Pointer, numBreakers uint64) Sampler {
		var p Sampler
		samplerInitDryFFI.Call(unsafe.Pointer(&p), unsafe.Pointer(&vocab), &nCtxTrain, &multiplier, &base, &allowedLength, &penaltyLast,
			&seqBreakers, &numBreakers)
		return p
	}

	var samplerInitGrammarLazyPatternsFFI ffi.Fun
	if samplerInitGrammarLazyPatternsFFI, err = ffiLib.Prep("llama_sampler_init_grammar_lazy_patterns",
		&ffi.TypePointer,
		&ffi.TypePointer,
		&ffi.TypePointer,
		&ffi.TypePointer,
		&ffi.TypePointer,
		&ffiTypeSize,
		&ffi.TypePointer,
		&ffiTypeSize,
	); err != nil {
		return loadError("llama_sampler_init_grammar_lazy_patterns", err)
	}
	samplerInitGrammarLazyPatternsFn = func(vocab Vocab, grammar *byte, root *byte, triggerPatterns unsafe.Pointer, numPatterns uint64, triggerTokens unsafe.Pointer, numTokens uint64) Sampler {
		var s Sampler
		samplerInitGrammarLazyPatternsFFI.Call(
			unsafe.Pointer(&s),
			unsafe.Pointer(&vocab),
			unsafe.Pointer(&grammar),
			unsafe.Pointer(&root),
			&triggerPatterns,
			&numPatterns,
			&triggerTokens,
			&numTokens,
		)
		return s
	}

	loadSamplingPuregoFuncs(lib)
	return nil
}
