package llama

import (
	"errors"
	"os"
	"unsafe"

	"github.com/David2024patton/iTaKTorch/pkg/torch/utils"
	"github.com/ebitengine/purego"
)

// Typed Go function variables for struct-by-value calls
// Filled by platform-specific loaders (model_other.go / model_windows.go)
var (
	modelDefaultParamsFn         func() ModelParams
	modelLoadFromFileFn          func(pathModel *byte, params ModelParams) Model
	modelLoadFromSplitsFn        func(paths unsafe.Pointer, nPaths uint64, params ModelParams) Model
	initFromModelFn              func(model Model, params ContextParams) Context
	modelQuantizeDefaultParamsFn func() ModelQuantizeParams
	modelQuantizeFn              func(fnameInp *byte, fnameOut *byte, params *ModelQuantizeParams) uint32
	modelParamsFitFn             func(pathModel *byte, mparams *ModelParams, cparams *ContextParams, tensorSplit *float32, tensorBuftOverrides *TensorBuftOverride, margins *uint64, nCtxMin uint32, logLevel LogLevel) int32
)

// purego direct-call function pointers
var (
	modelFreeFn                func(model Model)
	modelChatTemplateFn        func(model Model, name *byte) *byte
	modelHasEncoderFn          func(model Model) uint8
	modelHasDecoderFn          func(model Model) uint8
	modelDecoderStartTokenFn   func(model Model) int32
	modelNCtxTrainFn           func(model Model) int32
	modelNEmbdFn               func(model Model) int32
	modelNEmbdInpFn            func(model Model) int32
	modelNEmbdOutFn            func(model Model) int32
	modelNLayerFn              func(model Model) int32
	modelNHeadFn               func(model Model) int32
	modelNHeadKVFn             func(model Model) int32
	modelNSWAFn                func(model Model) int32
	modelNClsOutFn             func(model Model) uint32
	modelClsLabelFn            func(model Model, i uint32) *byte
	modelDescFn                func(model Model, buf *byte, bufSize uintptr) int32
	modelSizeFn                func(model Model) uint64
	modelIsRecurrentFn         func(model Model) uint8
	modelIsHybridFn            func(model Model) uint8
	modelIsDiffusionFn         func(model Model) uint8
	modelRopeFreqScaleTrainFn  func(model Model) float32
	modelRopeTypeFn            func(model Model) int32
	modelMetaValStrFn          func(model Model, key *byte, buf *byte, bufSize uintptr) int32
	modelMetaCountFn           func(model Model) int32
	modelMetaKeyByIndexFn      func(model Model, i int32, buf *byte, bufSize uintptr) int32
	modelMetaValStrByIndexFn   func(model Model, i int32, buf *byte, bufSize uintptr) int32
	modelMetaKeyStrFn          func(key int32) *byte
)

func loadModelPuregoFuncs(lib uintptr) {
	purego.RegisterLibFunc(&modelFreeFn, lib, "llama_model_free")
	purego.RegisterLibFunc(&modelChatTemplateFn, lib, "llama_model_chat_template")
	purego.RegisterLibFunc(&modelHasEncoderFn, lib, "llama_model_has_encoder")
	purego.RegisterLibFunc(&modelHasDecoderFn, lib, "llama_model_has_decoder")
	purego.RegisterLibFunc(&modelDecoderStartTokenFn, lib, "llama_model_decoder_start_token")
	purego.RegisterLibFunc(&modelNCtxTrainFn, lib, "llama_model_n_ctx_train")
	purego.RegisterLibFunc(&modelNEmbdFn, lib, "llama_model_n_embd")
	purego.RegisterLibFunc(&modelNEmbdInpFn, lib, "llama_model_n_embd_inp")
	purego.RegisterLibFunc(&modelNEmbdOutFn, lib, "llama_model_n_embd_out")
	purego.RegisterLibFunc(&modelNLayerFn, lib, "llama_model_n_layer")
	purego.RegisterLibFunc(&modelNHeadFn, lib, "llama_model_n_head")
	purego.RegisterLibFunc(&modelNHeadKVFn, lib, "llama_model_n_head_kv")
	purego.RegisterLibFunc(&modelNSWAFn, lib, "llama_model_n_swa")
	purego.RegisterLibFunc(&modelNClsOutFn, lib, "llama_model_n_cls_out")
	purego.RegisterLibFunc(&modelClsLabelFn, lib, "llama_model_cls_label")
	purego.RegisterLibFunc(&modelDescFn, lib, "llama_model_desc")
	purego.RegisterLibFunc(&modelSizeFn, lib, "llama_model_size")
	purego.RegisterLibFunc(&modelIsRecurrentFn, lib, "llama_model_is_recurrent")
	purego.RegisterLibFunc(&modelIsHybridFn, lib, "llama_model_is_hybrid")
	purego.RegisterLibFunc(&modelIsDiffusionFn, lib, "llama_model_is_diffusion")
	purego.RegisterLibFunc(&modelRopeFreqScaleTrainFn, lib, "llama_model_rope_freq_scale_train")
	purego.RegisterLibFunc(&modelRopeTypeFn, lib, "llama_model_rope_type")
	purego.RegisterLibFunc(&modelMetaValStrFn, lib, "llama_model_meta_val_str")
	purego.RegisterLibFunc(&modelMetaCountFn, lib, "llama_model_meta_count")
	purego.RegisterLibFunc(&modelMetaKeyByIndexFn, lib, "llama_model_meta_key_by_index")
	purego.RegisterLibFunc(&modelMetaValStrByIndexFn, lib, "llama_model_meta_val_str_by_index")
	purego.RegisterLibFunc(&modelMetaKeyStrFn, lib, "llama_model_meta_key_str")
}

// ModelDefaultParams returns default parameters for loading a Model.
func ModelDefaultParams() ModelParams {
	return modelDefaultParamsFn()
}

// ModelLoadFromFile loads a Model from a GGUF file.
func ModelLoadFromFile(pathModel string, params ModelParams) (Model, error) {
	if _, err := os.Stat(pathModel); os.IsNotExist(err) {
		return 0, err
	}

	file := &[]byte(pathModel + "\x00")[0]
	model := modelLoadFromFileFn(file, params)
	if model == 0 {
		return model, errors.New("failed to load model")
	}

	return model, nil
}

// ModelLoadFromSplits loads a Model from multiple split files.
func ModelLoadFromSplits(paths []string, params ModelParams) (Model, error) {
	if len(paths) == 0 {
		return 0, errors.New("no paths provided")
	}

	cStrs := make([]*byte, len(paths))
	for i := range paths {
		cStrs[i] = &[]byte(paths[i] + "\x00")[0]
	}
	cPaths := unsafe.Pointer(&cStrs[0])
	nPaths := uint64(len(paths))

	model := modelLoadFromSplitsFn(cPaths, nPaths, params)
	if model == 0 {
		return model, errors.New("failed to load model from splits")
	}

	return model, nil
}

// ModelFree frees a previously opened model.
func ModelFree(model Model) error {
	if model == 0 {
		return errors.New("invalid model")
	}
	modelFreeFn(model)
	return nil
}

// InitFromModel initializes a previously loaded Model, and then returns a new Context.
func InitFromModel(model Model, params ContextParams) (Context, error) {
	if model == 0 {
		return 0, errors.New("invalid model")
	}
	ctx := initFromModelFn(model, params)

	if ctx == 0 {
		return ctx, errors.New("failed to initialize model")
	}
	return ctx, nil
}

// ModelChatTemplate returns a named chat template for the Model.
func ModelChatTemplate(model Model, name string) string {
	if model == 0 {
		return ""
	}
	var n *byte
	if len(name) > 0 {
		n = &[]byte(name + "\x00")[0]
	}
	template := modelChatTemplateFn(model, n)
	return utils.BytePtrToString(template)
}

// ModelHasEncoder returns if the Model has an encoder.
func ModelHasEncoder(model Model) bool {
	if model == 0 {
		return false
	}
	return modelHasEncoderFn(model) != 0
}

// ModelHasDecoder returns if the Model has a decoder.
func ModelHasDecoder(model Model) bool {
	if model == 0 {
		return false
	}
	return modelHasDecoderFn(model) != 0
}

// ModelDecoderStartToken returns the start Token for the Model's decoder.
func ModelDecoderStartToken(model Model) Token {
	if model == 0 {
		return TokenNull
	}
	return Token(modelDecoderStartTokenFn(model))
}

// ModelNCtxTrain returns the number of context tokens used during training.
func ModelNCtxTrain(model Model) int32 {
	if model == 0 {
		return 0
	}
	return modelNCtxTrainFn(model)
}

// ModelNEmbd returns the embedding size of the Model.
func ModelNEmbd(model Model) int32 {
	if model == 0 {
		return 0
	}
	return modelNEmbdFn(model)
}

// ModelNEmbdInp returns the input embedding size of the Model.
func ModelNEmbdInp(model Model) int32 {
	if model == 0 {
		return 0
	}
	return modelNEmbdInpFn(model)
}

// ModelNEmbdOut returns the output embedding size of the Model.
func ModelNEmbdOut(model Model) int32 {
	if model == 0 {
		return 0
	}
	return modelNEmbdOutFn(model)
}

// ModelNLayer returns the number of layers in the Model.
func ModelNLayer(model Model) int32 {
	if model == 0 {
		return 0
	}
	return modelNLayerFn(model)
}

// ModelNHead returns the number of attention heads in the Model.
func ModelNHead(model Model) int32 {
	if model == 0 {
		return 0
	}
	return modelNHeadFn(model)
}

// ModelNHeadKV returns the number of key/value attention heads in the Model.
func ModelNHeadKV(model Model) int32 {
	if model == 0 {
		return 0
	}
	return modelNHeadKVFn(model)
}

// ModelNSWA returns the number of SWA layers in the Model.
func ModelNSWA(model Model) int32 {
	if model == 0 {
		return 0
	}
	return modelNSWAFn(model)
}

// ModelNClsOut returns the number of classifier outputs (only valid for classifier models).
func ModelNClsOut(model Model) uint32 {
	if model == 0 {
		return 0
	}
	return modelNClsOutFn(model)
}

// ModelClsLabel returns the label of a classifier output by index.
func ModelClsLabel(model Model, index uint32) string {
	if model == 0 {
		return ""
	}
	labelPtr := modelClsLabelFn(model, index)
	if labelPtr == nil {
		return ""
	}
	return utils.BytePtrToString(labelPtr)
}

// ModelDesc retrieves a string describing the model type.
func ModelDesc(model Model) string {
	if model == 0 {
		return ""
	}
	buf := make([]byte, 128)
	b := unsafe.SliceData(buf)

	result := modelDescFn(model, b, uintptr(len(buf)))
	if result < 0 {
		return ""
	}
	return string(buf[:result])
}

// ModelSize returns the total size of all tensors in the model in bytes.
func ModelSize(model Model) uint64 {
	if model == 0 {
		return 0
	}
	return modelSizeFn(model)
}

// ModelIsRecurrent returns true if the model is recurrent.
func ModelIsRecurrent(model Model) bool {
	if model == 0 {
		return false
	}
	return modelIsRecurrentFn(model) != 0
}

// ModelIsHybrid returns true if the model is hybrid.
func ModelIsHybrid(model Model) bool {
	if model == 0 {
		return false
	}
	return modelIsHybridFn(model) != 0
}

// ModelIsDiffusion returns true if the model is diffusion-based.
func ModelIsDiffusion(model Model) bool {
	if model == 0 {
		return false
	}
	return modelIsDiffusionFn(model) != 0
}

// ModelRopeFreqScaleTrain retrieves the model's RoPE frequency scaling factor.
func ModelRopeFreqScaleTrain(model Model) float32 {
	if model == 0 {
		return 0.0
	}
	return modelRopeFreqScaleTrainFn(model)
}

// ModelRopeType retrieves the RoPE type of the model.
func ModelRopeType(model Model) RopeScalingType {
	if model == 0 {
		return RopeScalingTypeNone
	}
	return RopeScalingType(modelRopeTypeFn(model))
}

// Warmup warms up a model by processing a representative batch of tokens.
func Warmup(lctx Context, model Model) error {
	if lctx == 0 || model == 0 {
		return errors.New("invalid context or model")
	}

	vocab := ModelGetVocab(model)

	SetWarmup(lctx, true)

	bos := VocabBOS(vocab)
	eos := VocabEOS(vocab)

	const warmupBatchSize = 32
	tokens := make([]Token, 0, warmupBatchSize)

	if bos != TokenNull {
		tokens = append(tokens, bos)
	}

	vocabSize := VocabNTokens(vocab)
	if vocabSize <= 0 {
		vocabSize = 32000
	}

	for len(tokens) < warmupBatchSize-1 {
		tokenID := Token((len(tokens) + 100) % int(vocabSize))
		tokens = append(tokens, tokenID)
	}

	if eos != TokenNull {
		tokens = append(tokens, eos)
	} else if len(tokens) < warmupBatchSize {
		tokens = append(tokens, 0)
	}

	if ModelHasEncoder(model) {
		batch := BatchGetOne(tokens)
		Encode(lctx, batch)

		start := ModelDecoderStartToken(model)
		if start == TokenNull {
			start = bos
		}
		tokens = []Token{start}
	}

	if ModelHasDecoder(model) {
		batch := BatchGetOne(tokens)
		Decode(lctx, batch)
	}

	mem, err := GetMemory(lctx)
	if err != nil {
		return err
	}
	if err := MemoryClear(mem, true); err != nil {
		return err
	}

	Synchronize(lctx)
	SetWarmup(lctx, false)

	return nil
}

// ModelMetaValStr gets metadata value as a string by key name.
func ModelMetaValStr(model Model, key string) (string, bool) {
	if model == 0 {
		return "", false
	}
	buf := make([]byte, 32768)
	b := unsafe.SliceData(buf)

	keyPtr, _ := utils.BytePtrFromString(key)
	result := modelMetaValStrFn(model, keyPtr, b, uintptr(len(buf)))
	if result < 0 {
		return "", false
	}

	value := make([]byte, result)
	copy(value, buf[:result])
	return string(value), true
}

// ModelMetaCount gets the number of metadata key/value pairs.
func ModelMetaCount(model Model) int32 {
	if model == 0 {
		return 0
	}
	return modelMetaCountFn(model)
}

// ModelMetaKeyByIndex gets metadata key name by index.
func ModelMetaKeyByIndex(model Model, i int32) (string, bool) {
	if model == 0 {
		return "", false
	}
	buf := make([]byte, 128)
	b := unsafe.SliceData(buf)

	result := modelMetaKeyByIndexFn(model, i, b, uintptr(len(buf)))
	if result < 0 {
		return "", false
	}

	value := make([]byte, result)
	copy(value, buf[:result])
	return string(value), true
}

// ModelMetaValStrByIndex gets metadata value as a string by index.
func ModelMetaValStrByIndex(model Model, i int32) (string, bool) {
	if model == 0 {
		return "", false
	}
	buf := make([]byte, 32768)
	b := unsafe.SliceData(buf)

	result := modelMetaValStrByIndexFn(model, i, b, uintptr(len(buf)))
	if result < 0 {
		return "", false
	}

	value := make([]byte, result)
	copy(value, buf[:result])
	return string(value), true
}

// ModelMetaKeyStr returns the metadata key name for a given enum key.
func ModelMetaKeyStr(key ModelMetaKey) string {
	ptr := modelMetaKeyStrFn(int32(key))
	if ptr == nil {
		return ""
	}
	return utils.BytePtrToString(ptr)
}

// SetTensorBufOverrides sets tensor buffer overrides for Mixture of Experts (MoE) execution.
func (p *ModelParams) SetTensorBufOverrides(overrides []TensorBuftOverride) {
	if len(overrides) == 0 {
		p.TensorBuftOverrides = uintptr(0)
		return
	}

	p.TensorBuftOverrides = uintptr(unsafe.Pointer(&overrides[0]))
}

// SetDevices sets the devices to be used for model execution.
func (p *ModelParams) SetDevices(devices []GGMLBackendDevice) {
	if len(devices) == 0 {
		p.Devices = uintptr(0)
		return
	}

	p.Devices = uintptr(unsafe.Pointer(&devices[0]))
}

// ModelQuantizeDefaultParams returns default parameters for model quantization.
func ModelQuantizeDefaultParams() ModelQuantizeParams {
	return modelQuantizeDefaultParamsFn()
}

// ModelQuantize quantizes a model from an input file to an output file using the specified parameters.
func ModelQuantize(fnameInp, fnameOut string, params *ModelQuantizeParams) uint32 {
	fileInp, err := utils.BytePtrFromString(fnameInp)
	if err != nil {
		return 0
	}

	fileOut, err := utils.BytePtrFromString(fnameOut)
	if err != nil {
		return 0
	}

	return modelQuantizeFn(fileInp, fileOut, params)
}

type ModelParamsFitStatus int32

const (
	ModelParamsFitStatusSuccess ModelParamsFitStatus = 0
	ModelParamsFitStatusFailure ModelParamsFitStatus = 1
	ModelParamsFitStatusError   ModelParamsFitStatus = 2
)

// ModelParamsFit attempts to fit model and context parameters to available device memory.
func ModelParamsFit(pathModel string, mparams *ModelParams, cparams *ContextParams, tensorSplit []float32, tensorBuftOverrides []TensorBuftOverride, margins []uint64, nCtxMin uint32, logLevel LogLevel) ModelParamsFitStatus {
	pathPtr, err := utils.BytePtrFromString(pathModel)
	if err != nil {
		return ModelParamsFitStatusError
	}

	var tensorSplitPtr *float32
	if len(tensorSplit) > 0 {
		tensorSplitPtr = &tensorSplit[0]
	}

	var tensorBuftOverridesPtr *TensorBuftOverride
	if len(tensorBuftOverrides) > 0 {
		tensorBuftOverridesPtr = &tensorBuftOverrides[0]
	}

	var marginsPtr *uint64
	if len(margins) > 0 {
		marginsPtr = &margins[0]
	}

	result := modelParamsFitFn(pathPtr, mparams, cparams, tensorSplitPtr, tensorBuftOverridesPtr, marginsPtr, nCtxMin, logLevel)
	return ModelParamsFitStatus(result)
}
