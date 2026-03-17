package llama

import (
	"errors"
	"unsafe"

	"github.com/ebitengine/purego"
)

// Typed Go function variables for struct-by-value calls
// Filled by platform-specific loaders (context_other.go / context_windows.go)
var (
	contextDefaultParamsFn func() ContextParams
	encodeFn               func(ctx Context, batch Batch) int32
	decodeFn               func(ctx Context, batch Batch) int32
)

// purego direct-call function pointers
var (
	freeFn                          func(ctx Context)
	setWarmupFn                     func(ctx Context, warmup uint8)
	perfContextResetFn              func(ctx Context)
	getMemoryFn                     func(ctx Context) Memory
	synchronizeFn                   func(ctx Context)
	poolingTypeFn                   func(ctx Context) int32
	getEmbeddingsIthFn              func(ctx Context, i int32) *float32
	getEmbeddingsSeqFn              func(ctx Context, seqID int32) *float32
	getEmbeddingsFn                 func(ctx Context) *float32
	getLogitsIthFast                func(ctx Context, i int32) *float32
	getLogitsFast                   func(ctx Context) *float32
	nCtxFn                          func(ctx Context) uint32
	nBatchFn                        func(ctx Context) uint32
	nUBatchFn                       func(ctx Context) uint32
	nSeqMaxFn                       func(ctx Context) uint32
	getModelFn                      func(ctx Context) Model
	setEmbeddingsFn                 func(ctx Context, embeddings uint8)
	setCausalAttnFn                 func(ctx Context, causalAttn uint8)
	setAdapterCvecFn                func(ctx Context, data *float32, length uintptr, nEmbd, ilStart, ilEnd int32) int32
	getSampledTokenIthFn            func(ctx Context, i int32) int32
	getSampledProbsIthFn            func(ctx Context, i int32) *float32
	getSampledProbsCountIthFn       func(ctx Context, i int32) uint32
	getSampledLogitsIthFn           func(ctx Context, i int32) *float32
	getSampledLogitsCountIthFn      func(ctx Context, i int32) uint32
	getSampledCandidatesIthFn       func(ctx Context, i int32) *Token
	getSampledCandidatesCountIthFn  func(ctx Context, i int32) uint32
	setAbortCallbackFn              func(ctx Context, callback uintptr, data uintptr)
)

func loadContextPuregoFuncs(lib uintptr) {
	purego.RegisterLibFunc(&freeFn, lib, "llama_free")
	purego.RegisterLibFunc(&setWarmupFn, lib, "llama_set_warmup")
	purego.RegisterLibFunc(&perfContextResetFn, lib, "llama_perf_context_reset")
	purego.RegisterLibFunc(&getMemoryFn, lib, "llama_get_memory")
	purego.RegisterLibFunc(&synchronizeFn, lib, "llama_synchronize")
	purego.RegisterLibFunc(&poolingTypeFn, lib, "llama_pooling_type")
	purego.RegisterLibFunc(&getEmbeddingsIthFn, lib, "llama_get_embeddings_ith")
	purego.RegisterLibFunc(&getEmbeddingsSeqFn, lib, "llama_get_embeddings_seq")
	purego.RegisterLibFunc(&getEmbeddingsFn, lib, "llama_get_embeddings")
	purego.RegisterLibFunc(&getLogitsIthFast, lib, "llama_get_logits_ith")
	purego.RegisterLibFunc(&getLogitsFast, lib, "llama_get_logits")
	purego.RegisterLibFunc(&nCtxFn, lib, "llama_n_ctx")
	purego.RegisterLibFunc(&nBatchFn, lib, "llama_n_batch")
	purego.RegisterLibFunc(&nUBatchFn, lib, "llama_n_ubatch")
	purego.RegisterLibFunc(&nSeqMaxFn, lib, "llama_n_seq_max")
	purego.RegisterLibFunc(&getModelFn, lib, "llama_get_model")
	purego.RegisterLibFunc(&setEmbeddingsFn, lib, "llama_set_embeddings")
	purego.RegisterLibFunc(&setCausalAttnFn, lib, "llama_set_causal_attn")
	purego.RegisterLibFunc(&setAdapterCvecFn, lib, "llama_set_adapter_cvec")
	purego.RegisterLibFunc(&getSampledTokenIthFn, lib, "llama_get_sampled_token_ith")
	purego.RegisterLibFunc(&getSampledProbsIthFn, lib, "llama_get_sampled_probs_ith")
	purego.RegisterLibFunc(&getSampledProbsCountIthFn, lib, "llama_get_sampled_probs_count_ith")
	purego.RegisterLibFunc(&getSampledLogitsIthFn, lib, "llama_get_sampled_logits_ith")
	purego.RegisterLibFunc(&getSampledLogitsCountIthFn, lib, "llama_get_sampled_logits_count_ith")
	purego.RegisterLibFunc(&getSampledCandidatesIthFn, lib, "llama_get_sampled_candidates_ith")
	purego.RegisterLibFunc(&getSampledCandidatesCountIthFn, lib, "llama_get_sampled_candidates_count_ith")
	purego.RegisterLibFunc(&setAbortCallbackFn, lib, "llama_set_abort_callback")
}

var (
	errInvalidContext = errors.New("invalid context")
)

// ContextDefaultParams returns the default params to initialize a model context.
func ContextDefaultParams() ContextParams {
	return contextDefaultParamsFn()
}

// Free frees the resources for a model context.
func Free(ctx Context) error {
	if ctx == 0 {
		return errInvalidContext
	}
	freeFn(ctx)
	return nil
}

// SetWarmup sets the model context warmup mode on or off.
func SetWarmup(ctx Context, warmup bool) error {
	if ctx == 0 {
		return errInvalidContext
	}
	var w uint8
	if warmup {
		w = 1
	}
	setWarmupFn(ctx, w)
	return nil
}

// Encode encodes a batch of Token.
func Encode(ctx Context, batch Batch) (int32, error) {
	if ctx == 0 {
		return 0, errInvalidContext
	}
	return encodeFn(ctx, batch), nil
}

// Decode decodes a batch of Token.
func Decode(ctx Context, batch Batch) (int32, error) {
	if ctx == 0 {
		return 0, errInvalidContext
	}
	return decodeFn(ctx, batch), nil
}

// PerfContextReset resets the performance metrics for the model context.
func PerfContextReset(ctx Context) error {
	if ctx == 0 {
		return errInvalidContext
	}
	perfContextResetFn(ctx)
	return nil
}

// GetMemory returns the current Memory for the Context.
func GetMemory(ctx Context) (Memory, error) {
	if ctx == 0 {
		return 0, errInvalidContext
	}
	return getMemoryFn(ctx), nil
}

// Synchronize waits until all computations are finished.
func Synchronize(ctx Context) error {
	if ctx == 0 {
		return errInvalidContext
	}
	synchronizeFn(ctx)
	return nil
}

// GetPoolingType returns the PoolingType for this context.
func GetPoolingType(ctx Context) PoolingType {
	if ctx == 0 {
		return PoolingTypeNone
	}
	return PoolingType(poolingTypeFn(ctx))
}

// GetEmbeddingsIth gets the embeddings for the ith token.
func GetEmbeddingsIth(ctx Context, i int32, nVocab int32) ([]float32, error) {
	if ctx == 0 {
		return nil, errInvalidContext
	}
	result := getEmbeddingsIthFn(ctx, i)
	if result == nil {
		return nil, nil
	}
	return unsafe.Slice(result, nVocab), nil
}

// GetEmbeddingsSeq gets the embeddings for this sequence ID.
func GetEmbeddingsSeq(ctx Context, seqID SeqId, nVocab int32) ([]float32, error) {
	if ctx == 0 {
		return nil, errInvalidContext
	}
	result := getEmbeddingsSeqFn(ctx, int32(seqID))
	if result == nil {
		return nil, nil
	}
	return unsafe.Slice(result, nVocab), nil
}

// GetEmbeddings retrieves all output token embeddings.
func GetEmbeddings(ctx Context, nOutputs, nEmbeddings int) ([]float32, error) {
	if ctx == 0 {
		return nil, errInvalidContext
	}
	result := getEmbeddingsFn(ctx)
	if result == nil || nOutputs <= 0 || nEmbeddings <= 0 {
		return nil, nil
	}
	return unsafe.Slice(result, nOutputs*nEmbeddings), nil
}

// GetLogitsIth retrieves the logits for the ith token.
func GetLogitsIth(ctx Context, i int32, nVocab int) ([]float32, error) {
	if ctx == 0 {
		return nil, errInvalidContext
	}
	logitsPtr := getLogitsIthFast(ctx, i)
	if logitsPtr == nil {
		return nil, nil
	}
	return unsafe.Slice(logitsPtr, nVocab), nil
}

// GetLogits retrieves all token logits from the last call to llama_decode.
func GetLogits(ctx Context, nTokens, nVocab int) ([]float32, error) {
	if ctx == 0 {
		return nil, errInvalidContext
	}
	result := getLogitsFast(ctx)
	if result == nil || nTokens <= 0 || nVocab <= 0 {
		return nil, nil
	}
	return unsafe.Slice(result, nTokens*nVocab), nil
}

// NCtx returns the number of context tokens.
func NCtx(ctx Context) uint32 {
	if ctx == 0 {
		return 0
	}
	return nCtxFn(ctx)
}

// NBatch returns the number of batch tokens.
func NBatch(ctx Context) uint32 {
	if ctx == 0 {
		return 0
	}
	return nBatchFn(ctx)
}

// NUBatch returns the number of micro-batch tokens.
func NUBatch(ctx Context) uint32 {
	if ctx == 0 {
		return 0
	}
	return nUBatchFn(ctx)
}

// NSeqMax returns the maximum number of sequences.
func NSeqMax(ctx Context) uint32 {
	if ctx == 0 {
		return 0
	}
	return nSeqMaxFn(ctx)
}

// GetModel retrieves the model associated with the given context.
func GetModel(ctx Context) Model {
	if ctx == 0 {
		return 0
	}
	return getModelFn(ctx)
}

// SetEmbeddings sets whether the context outputs embeddings or not.
func SetEmbeddings(ctx Context, embeddings bool) {
	if ctx == 0 {
		return
	}
	var e uint8
	if embeddings {
		e = 1
	}
	setEmbeddingsFn(ctx, e)
}

// SetCausalAttn sets whether to use causal attention or not.
func SetCausalAttn(ctx Context, causalAttn bool) {
	if ctx == 0 {
		return
	}
	var c uint8
	if causalAttn {
		c = 1
	}
	setCausalAttnFn(ctx, c)
}

// SetAdapterCvec sets a loaded control vector to a llama_context.
func SetAdapterCvec(ctx Context, data []float32, nEmbd, ilStart, ilEnd int32) int32 {
	if ctx == 0 {
		return -1
	}

	var dataPtr *float32
	var length uintptr
	if data != nil {
		dataPtr = unsafe.SliceData(data)
		length = uintptr(len(data))
	}

	return setAdapterCvecFn(ctx, dataPtr, length, nEmbd, ilStart, ilEnd)
}

// GetSampledTokenIth retrieves the sampled token for the ith output.
func GetSampledTokenIth(ctx Context, i int32) (Token, error) {
	if ctx == 0 {
		return TokenNull, errInvalidContext
	}
	return Token(getSampledTokenIthFn(ctx, i)), nil
}

// GetSampledProbsIth retrieves the sampled probabilities for the ith output.
func GetSampledProbsIth(ctx Context, i int32, nVocab int) ([]float32, error) {
	if ctx == 0 {
		return nil, errInvalidContext
	}
	result := getSampledProbsIthFn(ctx, i)
	if result == nil {
		return nil, nil
	}
	return unsafe.Slice(result, nVocab), nil
}

// GetSampledProbsCountIth retrieves the count of sampled probabilities for the ith output.
func GetSampledProbsCountIth(ctx Context, i int32) (uint32, error) {
	if ctx == 0 {
		return 0, errInvalidContext
	}
	return getSampledProbsCountIthFn(ctx, i), nil
}

// GetSampledLogitsIth retrieves the sampled logits for the ith output.
func GetSampledLogitsIth(ctx Context, i int32, nVocab int) ([]float32, error) {
	if ctx == 0 {
		return nil, errInvalidContext
	}
	result := getSampledLogitsIthFn(ctx, i)
	if result == nil {
		return nil, nil
	}
	return unsafe.Slice(result, nVocab), nil
}

// GetSampledLogitsCountIth retrieves the count of sampled logits for the ith output.
func GetSampledLogitsCountIth(ctx Context, i int32) (uint32, error) {
	if ctx == 0 {
		return 0, errInvalidContext
	}
	return getSampledLogitsCountIthFn(ctx, i), nil
}

// GetSampledCandidatesIth retrieves the sampled candidates for the ith output.
func GetSampledCandidatesIth(ctx Context, i int32, nVocab int) ([]Token, error) {
	if ctx == 0 {
		return nil, errInvalidContext
	}
	result := getSampledCandidatesIthFn(ctx, i)
	if result == nil {
		return nil, nil
	}
	return unsafe.Slice(result, nVocab), nil
}

// GetSampledCandidatesCountIth retrieves the count of sampled candidates for the ith output.
func GetSampledCandidatesCountIth(ctx Context, i int32) (uint32, error) {
	if ctx == 0 {
		return 0, errInvalidContext
	}
	return getSampledCandidatesCountIthFn(ctx, i), nil
}

// AbortFunc is a callback function that can be used to abort computation.
type AbortFunc func() bool

// SetAbortCallback sets a callback function that can be used to abort computation.
func SetAbortCallback(ctx Context, fn AbortFunc) {
	callback := newAbortCallback(fn)
	setAbortCallbackFn(ctx, callback, 0)
}

// newAbortCallback creates a C-compatible callback from a Go AbortFunc.
func newAbortCallback(fn AbortFunc) uintptr {
	return purego.NewCallback(func(data uintptr) uintptr {
		if fn() {
			return 1
		}
		return 0
	})
}
