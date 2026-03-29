package llama

import (
	"math"
	"unsafe"

	"github.com/David2024patton/iTaKTorchQ4/pkg/torch/utils"
	"github.com/ebitengine/purego"
)

type SamplerType int32

const (
	SamplerTypeNone        SamplerType = iota
	SamplerTypeDry                     = 1
	SamplerTypeTopK                    = 2
	SamplerTypeTopP                    = 3
	SamplerTypeMinP                    = 4
	SamplerTypeTypicalP                = 6
	SamplerTypeTemperature             = 7
	SamplerTypeXTC                     = 8
	SamplerTypeInfill                  = 9
	SamplerTypePenalties               = 10
	SamplerTypeTopNSigma               = 11
	SamplerTypeAdaptiveP               = 12
	SamplerTypeLogitBias               = 13
)

// Typed Go function variables for struct-by-value calls
// Filled by platform-specific loaders (sampling_other.go / sampling_windows.go)
var (
	samplerChainDefaultParamsFn         func() SamplerChainParams
	samplerChainInitFn                  func(params SamplerChainParams) Sampler
	samplerInitDryFn                    func(vocab Vocab, nCtxTrain int32, multiplier float32, base float32, allowedLength int32, penaltyLast int32, seqBreakers unsafe.Pointer, numBreakers uint64) Sampler
	samplerInitGrammarLazyPatternsFn    func(vocab Vocab, grammar *byte, root *byte, triggerPatterns unsafe.Pointer, numPatterns uint64, triggerTokens unsafe.Pointer, numTokens uint64) Sampler
)

// purego direct-call function pointers
var (
	samplerNameFn           func(smpl Sampler) *byte
	samplerChainAddFn       func(chain Sampler, smpl Sampler)
	samplerChainGetFn       func(chain Sampler, i int32) Sampler
	samplerChainNFn         func(chain Sampler) int32
	samplerChainRemoveFn    func(chain Sampler, i int32) Sampler
	samplerInitGreedyFn     func() Sampler
	samplerInitDistFn       func(seed uint32) Sampler
	samplerInitLogitBiasFn  func(nVocab int32, nLogitBias int32, logitBias *LogitBias) Sampler
	samplerInitPenaltiesFn  func(lastN int32, repeat float32, freq float32, present float32) Sampler
	samplerInitTopNSigmaFn  func(n float32) Sampler
	samplerInitTopKFn       func(k int32) Sampler
	samplerInitTypicalFn    func(p float32, minKeep uintptr) Sampler
	samplerInitTopPFn       func(p float32, minKeep uintptr) Sampler
	samplerInitMinPFn       func(p float32, minKeep uintptr) Sampler
	samplerInitXTCFn        func(p float32, t float32, minKeep uintptr, seed uint32) Sampler
	samplerInitTempExtFn    func(t float32, delta float32, exponent float32) Sampler
	samplerInitGrammarFn    func(vocab Vocab, grammarStr *byte, grammarRoot *byte) Sampler
	samplerInitAdaptivePFn  func(target float32, decay float32, seed uint32) Sampler
	samplerInitInfillFn     func(vocab Vocab) Sampler
	samplerSampleFast       func(smpl Sampler, ctx Context, idx int32) int32
	samplerAcceptFn         func(smpl Sampler, token int32)
	samplerApplyFn          func(smpl Sampler, curP *TokenDataArray)
	samplerFreeFn           func(smpl Sampler)
	samplerResetFn          func(smpl Sampler)
	samplerCloneFn          func(smpl Sampler) Sampler
)

func loadSamplingPuregoFuncs(lib uintptr) {
	purego.RegisterLibFunc(&samplerNameFn, lib, "llama_sampler_name")
	purego.RegisterLibFunc(&samplerChainAddFn, lib, "llama_sampler_chain_add")
	purego.RegisterLibFunc(&samplerChainGetFn, lib, "llama_sampler_chain_get")
	purego.RegisterLibFunc(&samplerChainNFn, lib, "llama_sampler_chain_n")
	purego.RegisterLibFunc(&samplerChainRemoveFn, lib, "llama_sampler_chain_remove")
	purego.RegisterLibFunc(&samplerInitGreedyFn, lib, "llama_sampler_init_greedy")
	purego.RegisterLibFunc(&samplerInitDistFn, lib, "llama_sampler_init_dist")
	purego.RegisterLibFunc(&samplerInitLogitBiasFn, lib, "llama_sampler_init_logit_bias")
	purego.RegisterLibFunc(&samplerInitPenaltiesFn, lib, "llama_sampler_init_penalties")
	purego.RegisterLibFunc(&samplerInitTopNSigmaFn, lib, "llama_sampler_init_top_n_sigma")
	purego.RegisterLibFunc(&samplerInitTopKFn, lib, "llama_sampler_init_top_k")
	purego.RegisterLibFunc(&samplerInitTypicalFn, lib, "llama_sampler_init_typical")
	purego.RegisterLibFunc(&samplerInitTopPFn, lib, "llama_sampler_init_top_p")
	purego.RegisterLibFunc(&samplerInitMinPFn, lib, "llama_sampler_init_min_p")
	purego.RegisterLibFunc(&samplerInitXTCFn, lib, "llama_sampler_init_xtc")
	purego.RegisterLibFunc(&samplerInitTempExtFn, lib, "llama_sampler_init_temp_ext")
	purego.RegisterLibFunc(&samplerInitGrammarFn, lib, "llama_sampler_init_grammar")
	purego.RegisterLibFunc(&samplerInitAdaptivePFn, lib, "llama_sampler_init_adaptive_p")
	purego.RegisterLibFunc(&samplerInitInfillFn, lib, "llama_sampler_init_infill")
	purego.RegisterLibFunc(&samplerSampleFast, lib, "llama_sampler_sample")
	purego.RegisterLibFunc(&samplerAcceptFn, lib, "llama_sampler_accept")
	purego.RegisterLibFunc(&samplerApplyFn, lib, "llama_sampler_apply")
	purego.RegisterLibFunc(&samplerFreeFn, lib, "llama_sampler_free")
	purego.RegisterLibFunc(&samplerResetFn, lib, "llama_sampler_reset")
	purego.RegisterLibFunc(&samplerCloneFn, lib, "llama_sampler_clone")
}

// SamplerChainDefaultParams returns the default parameters to create a new sampling chain.
func SamplerChainDefaultParams() SamplerChainParams {
	return samplerChainDefaultParamsFn()
}

// SamplerChainInit initializes a new sampling chain.
func SamplerChainInit(params SamplerChainParams) Sampler {
	return samplerChainInitFn(params)
}

// SamplerName returns the name of the sampler as a string.
func SamplerName(smpl Sampler) string {
	if smpl == 0 {
		return ""
	}
	ptr := samplerNameFn(smpl)
	if ptr == nil {
		return ""
	}
	return utils.BytePtrToString(ptr)
}

// SamplerChainAdd adds a sampler to a sampling chain.
func SamplerChainAdd(chain Sampler, smpl Sampler) {
	if chain == 0 || smpl == 0 {
		return
	}
	samplerChainAddFn(chain, smpl)
}

// SamplerChainGet returns the i-th sampler from a sampler chain.
func SamplerChainGet(chain Sampler, i int32) Sampler {
	if chain == 0 {
		return 0
	}
	return samplerChainGetFn(chain, i)
}

// SamplerChainN returns the total number of samplers in the chain.
func SamplerChainN(chain Sampler) int {
	if chain == 0 {
		return 0
	}
	return int(samplerChainNFn(chain))
}

// SamplerChainRemove removes the i-th sampler from the chain and returns it.
func SamplerChainRemove(chain Sampler, i int32) Sampler {
	if chain == 0 {
		return 0
	}
	return samplerChainRemoveFn(chain, i)
}

// SamplerInitGreedy initializes a new greedy sampler.
func SamplerInitGreedy() Sampler {
	return samplerInitGreedyFn()
}

// SamplerInitDist initializes a new distribution sampler with the specified seed.
func SamplerInitDist(seed uint32) Sampler {
	return samplerInitDistFn(seed)
}

// SamplerInitLogitBias initializes a new logit bias sampler.
func SamplerInitLogitBias(nVocab int32, nLogitBias int32, logitBias *LogitBias) Sampler {
	return samplerInitLogitBiasFn(nVocab, nLogitBias, logitBias)
}

// SamplerInitPenalties initializes a new penalties sampler.
func SamplerInitPenalties(lastN int32, repeat float32, freq float32, present float32) Sampler {
	return samplerInitPenaltiesFn(lastN, repeat, freq, present)
}

// SamplerInitDry initializes a new DRY sampler.
func SamplerInitDry(vocab Vocab, nCtxTrain int32, multiplier float32, base float32, allowedLength int32, penaltyLast int32,
	seqBreakers []string) Sampler {
	var sp unsafe.Pointer
	numBreakers := uint64(len(seqBreakers))
	if numBreakers > 0 {
		seq := make([]*byte, 0)
		for _, s := range seqBreakers {
			ptr, err := utils.BytePtrFromString(s)
			if err != nil {
				return Sampler(0)
			}
			seq = append(seq, ptr)
		}
		sp = unsafe.Pointer(&seq[0])
	}

	return samplerInitDryFn(vocab, nCtxTrain, multiplier, base, allowedLength, penaltyLast, sp, numBreakers)
}

// SamplerInitTopNSigma initializes a new Top-N Sigma sampler.
func SamplerInitTopNSigma(n float32) Sampler {
	return samplerInitTopNSigmaFn(n)
}

// SamplerInitTopK initializes a new Top-K sampler.
func SamplerInitTopK(k int32) Sampler {
	return samplerInitTopKFn(k)
}

// SamplerInitTypical initializes a new Typical-P sampler.
func SamplerInitTypical(p float32, keep uint32) Sampler {
	return samplerInitTypicalFn(p, uintptr(keep))
}

// SamplerInitTopP initializes a new Top-P sampler.
func SamplerInitTopP(p float32, keep uint32) Sampler {
	return samplerInitTopPFn(p, uintptr(keep))
}

// SamplerInitMinP initializes a new Min-P sampler.
func SamplerInitMinP(p float32, keep uint32) Sampler {
	return samplerInitMinPFn(p, uintptr(keep))
}

// SamplerInitXTC initializes a new XTC sampler.
func SamplerInitXTC(p float32, t float32, minKeep uint32, seed uint32) Sampler {
	return samplerInitXTCFn(p, t, uintptr(minKeep), seed)
}

// SamplerInitTempExt initializes a new Temperature Extended sampler.
func SamplerInitTempExt(t float32, delta float32, exponent float32) Sampler {
	return samplerInitTempExtFn(t, delta, exponent)
}

// SamplerInitGrammar initializes a new Grammar sampler.
func SamplerInitGrammar(vocab Vocab, grammar, root string) Sampler {
	if vocab == 0 {
		return 0
	}
	grmr, _ := utils.BytePtrFromString(grammar)
	r, _ := utils.BytePtrFromString(root)
	return samplerInitGrammarFn(vocab, grmr, r)
}

// SamplerInitGrammarLazyPatterns initializes a lazy grammar sampler with trigger patterns and tokens.
func SamplerInitGrammarLazyPatterns(
	vocab Vocab,
	grammar, root string,
	triggerPatterns []string,
	triggerTokens []Token,
) Sampler {
	if vocab == 0 {
		return 0
	}
	grmr, _ := utils.BytePtrFromString(grammar)
	r, _ := utils.BytePtrFromString(root)

	var tp unsafe.Pointer
	numPatterns := uint64(len(triggerPatterns))
	if numPatterns > 0 {
		ptrs := make([]*byte, 0, numPatterns)
		for _, pat := range triggerPatterns {
			ptr, err := utils.BytePtrFromString(pat)
			if err != nil {
				return 0
			}
			ptrs = append(ptrs, ptr)
		}
		tp = unsafe.Pointer(&ptrs[0])
	}

	var tt unsafe.Pointer
	numTokens := uint64(len(triggerTokens))
	if numTokens > 0 {
		tt = unsafe.Pointer(&triggerTokens[0])
	}

	return samplerInitGrammarLazyPatternsFn(vocab, grmr, r, tp, numPatterns, tt, numTokens)
}

// SamplerInitAdaptiveP initializes a new Adaptive-P sampler.
func SamplerInitAdaptiveP(target float32, decay float32, seed uint32) Sampler {
	return samplerInitAdaptivePFn(target, decay, seed)
}

// SamplerInitInfill initializes a new infill sampler for fill-in-the-middle infilling.
func SamplerInitInfill(vocab Vocab) Sampler {
	return samplerInitInfillFn(vocab)
}

// SamplerSample samples a token from the sampler given the context and index.
func SamplerSample(smpl Sampler, ctx Context, idx int32) Token {
	if smpl == 0 || ctx == 0 {
		return TokenNull
	}
	return Token(samplerSampleFast(smpl, ctx, idx))
}

// SamplerAccept informs the sampler of the accepted token.
func SamplerAccept(smpl Sampler, token Token) {
	if smpl == 0 {
		return
	}
	samplerAcceptFn(smpl, int32(token))
}

// SamplerApply applies the sampler to the current token data array.
func SamplerApply(smpl Sampler, curP *TokenDataArray) {
	if smpl == 0 || curP == nil {
		return
	}
	samplerApplyFn(smpl, curP)
}

// SamplerFree frees the sampler.
func SamplerFree(smpl Sampler) {
	if smpl == 0 {
		return
	}
	samplerFreeFn(smpl)
}

// SamplerReset resets the sampler state.
func SamplerReset(smpl Sampler) {
	if smpl == 0 {
		return
	}
	samplerResetFn(smpl)
}

// SamplerClone creates a clone of the given sampler.
func SamplerClone(smpl Sampler) Sampler {
	if smpl == 0 {
		return 0
	}
	return samplerCloneFn(smpl)
}

var (
	// DefaultSamplers is the list of default samplers to use in a sampling chain.
	DefaultSamplers = []SamplerType{
		SamplerTypeLogitBias,
		SamplerTypePenalties,
		SamplerTypeDry,
		SamplerTypeTopNSigma,
		SamplerTypeTopK,
		SamplerTypeTypicalP,
		SamplerTypeTopP,
		SamplerTypeMinP,
		SamplerTypeXTC,
		SamplerTypeTemperature,
	}
)

// NewSampler creates a new sampling chain.
func NewSampler(model Model, samplers []SamplerType, params *SamplerParams) Sampler {
	var sampler Sampler
	if model == 0 || len(samplers) == 0 {
		return sampler
	}
	vocab := ModelGetVocab(model)
	nTokens := VocabNTokens(vocab)

	sampler = SamplerChainInit(SamplerChainDefaultParams())

	logitBiasEOG := make([]LogitBias, 0)
	for i := int32(0); i < nTokens; i++ {
		token := Token(i)
		if VocabIsEOG(vocab, token) {
			logitBiasEOG = append(logitBiasEOG, LogitBias{Token: token, Bias: math.SmallestNonzeroFloat32})
		}
	}

	for _, samplerType := range samplers {
		switch samplerType {
		case SamplerTypeLogitBias:
			bias := SamplerInitLogitBias(nTokens, int32(len(logitBiasEOG)), unsafe.SliceData(logitBiasEOG))
			SamplerChainAdd(sampler, bias)

		case SamplerTypeDry:
			dry := SamplerInitDry(vocab, ModelNCtxTrain(model), params.DryMultiplier, params.DryBase, params.DryAllowedLength, params.DryPenaltyLastN, params.DrySequenceBreakers)
			SamplerChainAdd(sampler, dry)

		case SamplerTypeTopK:
			topK := SamplerInitTopK(params.TopK)
			SamplerChainAdd(sampler, topK)

		case SamplerTypeTopP:
			topP := SamplerInitTopP(params.TopP, 0)
			SamplerChainAdd(sampler, topP)

		case SamplerTypeMinP:
			minP := SamplerInitMinP(params.MinP, 0)
			SamplerChainAdd(sampler, minP)

		case SamplerTypeTypicalP:
			typical := SamplerInitTypical(params.TypP, 0)
			SamplerChainAdd(sampler, typical)

		case SamplerTypeTemperature:
			temp := SamplerInitTempExt(params.Temp, 0, 1.0)
			SamplerChainAdd(sampler, temp)

		case SamplerTypeXTC:
			xtc := SamplerInitXTC(params.XTCProbability, params.XTCThreshold, 0, params.Seed)
			SamplerChainAdd(sampler, xtc)

		case SamplerTypeInfill:
			infill := SamplerInitInfill(vocab)
			SamplerChainAdd(sampler, infill)

		case SamplerTypePenalties:
			penalties := SamplerInitPenalties(params.PenaltyLastN, params.PenaltyRepeat, params.PenaltyFreq, params.PenaltyPresent)
			SamplerChainAdd(sampler, penalties)

		case SamplerTypeTopNSigma:
			topNSigma := SamplerInitTopNSigma(params.TopNSigma)
			SamplerChainAdd(sampler, topNSigma)
		}
	}

	dist := SamplerInitDist(params.Seed)
	SamplerChainAdd(sampler, dist)

	return sampler
}

// SamplerParams holds the parameters for creating samplers.
type SamplerParams struct {
	Seed                uint32
	NPrev               int32
	NProbs              int32
	MinKeep             int32
	TopK                int32
	TopP                float32
	MinP                float32
	XTCProbability      float32
	XTCThreshold        float32
	TypP                float32
	Temp                float32
	DynatempRange       float32
	DynatempExponent    float32
	PenaltyLastN        int32
	PenaltyRepeat       float32
	PenaltyFreq         float32
	PenaltyPresent      float32
	DryMultiplier       float32
	DryBase             float32
	DryAllowedLength    int32
	DryPenaltyLastN     int32
	Mirostat            int32
	TopNSigma           float32
	MirostatTau         float32
	MirostatEta         float32
	IgnoreEos           bool
	NoPerf              bool
	TimingPerToken      bool
	DrySequenceBreakers []string
}

// DefaultSamplerParams returns the default sampler parameters.
func DefaultSamplerParams() *SamplerParams {
	return &SamplerParams{
		Seed:                DefaultSeed,
		NPrev:               64,
		NProbs:              0,
		MinKeep:             0,
		TopK:                40,
		TopP:                0.95,
		MinP:                0.05,
		XTCProbability:      0.0,
		XTCThreshold:        0.1,
		TypP:                1.0,
		Temp:                0.8,
		DynatempRange:       0.0,
		DynatempExponent:    1.0,
		PenaltyLastN:        64,
		PenaltyRepeat:       1.0,
		PenaltyFreq:         0.0,
		PenaltyPresent:      0.0,
		DryMultiplier:       0.0,
		DryBase:             1.75,
		DryAllowedLength:    2,
		DryPenaltyLastN:     -1,
		Mirostat:            0,
		TopNSigma:           -1.0,
		MirostatTau:         5.0,
		MirostatEta:         0.1,
		IgnoreEos:           false,
		NoPerf:              false,
		TimingPerToken:      false,
		DrySequenceBreakers: []string{"\n", ":", "\"", "*"},
	}
}
