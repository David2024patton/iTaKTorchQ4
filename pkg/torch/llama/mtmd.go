package llama

import (
	"unsafe"

	"github.com/David2024patton/iTaKTorchQ4/pkg/torch/utils"
	"github.com/ebitengine/purego"
)

// MtmdContext is an opaque handle to a multi-modal context.
type MtmdContext uintptr

// MtmdBitmap is an opaque handle to an image/audio bitmap.
type MtmdBitmap uintptr

// MtmdImageTokens is an opaque handle to image tokens.
type MtmdImageTokens uintptr

// MtmdInputChunk is an opaque handle to an input chunk.
type MtmdInputChunk uintptr

// MtmdInputChunks is an opaque handle to a collection of input chunks.
type MtmdInputChunks uintptr

// MtmdInputChunkType describes whether a chunk is text, image, or audio.
type MtmdInputChunkType int32

const (
	MtmdInputChunkTypeText  MtmdInputChunkType = 0
	MtmdInputChunkTypeImage MtmdInputChunkType = 1
	MtmdInputChunkTypeAudio MtmdInputChunkType = 2
)

// MtmdInputText matches the C struct mtmd_input_text.
type MtmdInputText struct {
	Text         *byte
	AddSpecial   uint8
	ParseSpecial uint8
}

// MtmdContextParams matches the C struct mtmd_context_params.
type MtmdContextParams struct {
	UseGPU         uint8
	PrintTimings   uint8
	NThreads       int32
	ImageMarker    uintptr // deprecated, use MediaMarker
	MediaMarker    uintptr
	FlashAttnType  int32
	Warmup         uint8
	ImageMinTokens int32
	ImageMaxTokens int32
	CbEval         uintptr
	CbEvalUserData uintptr
}

// Typed Go function variables for struct-by-value calls
// Filled by platform-specific loaders (mtmd_other.go / mtmd_windows.go)
var (
	mtmdContextParamsDefaultFn func() MtmdContextParams
	mtmdInitFromFileFn         func(mmprojFname *byte, textModel Model, params MtmdContextParams) MtmdContext
)

// purego direct-call function pointers (no struct args)
var (
	mtmdDefaultMarkerFn            func() *byte
	mtmdFreeFn                     func(ctx MtmdContext)
	mtmdSupportVisionFn            func(ctx MtmdContext) uint8
	mtmdSupportAudioFn             func(ctx MtmdContext) uint8
	mtmdDecodeUseNonCausalFn       func(ctx MtmdContext) uint8
	mtmdDecodeUseMropeFn           func(ctx MtmdContext) uint8
	mtmdBitmapInitFn               func(nx uint32, ny uint32, data *byte) MtmdBitmap
	mtmdBitmapFreeFn               func(bitmap MtmdBitmap)
	mtmdBitmapGetNxFn              func(bitmap MtmdBitmap) uint32
	mtmdBitmapGetNyFn              func(bitmap MtmdBitmap) uint32
	mtmdInputChunksInitFn          func() MtmdInputChunks
	mtmdInputChunksSizeFn          func(chunks MtmdInputChunks) uint64
	mtmdInputChunksGetFn           func(chunks MtmdInputChunks, idx uint64) MtmdInputChunk
	mtmdInputChunksFreeFn          func(chunks MtmdInputChunks)
	mtmdInputChunkGetTypeFn        func(chunk MtmdInputChunk) int32
	mtmdInputChunkGetTokensTextFn  func(chunk MtmdInputChunk, nTokensOut *uint64) *Token
	mtmdInputChunkGetNTokensFn     func(chunk MtmdInputChunk) uint64
	mtmdTokenizeFn                 func(ctx MtmdContext, output MtmdInputChunks, text *MtmdInputText, bitmaps *MtmdBitmap, nBitmaps uint64) int32
	mtmdEncodeChunkFn              func(ctx MtmdContext, chunk MtmdInputChunk) int32
	mtmdGetOutputEmbdFn            func(ctx MtmdContext) *float32
)

// mtmdAvailable tracks whether mtmd.dll was loaded successfully.
var mtmdAvailable bool

// MtmdAvailable returns whether multi-modal support (mtmd.dll) is loaded.
func MtmdAvailable() bool {
	return mtmdAvailable
}

func loadMtmdPuregoFuncs(lib uintptr) {
	purego.RegisterLibFunc(&mtmdDefaultMarkerFn, lib, "mtmd_default_marker")
	purego.RegisterLibFunc(&mtmdFreeFn, lib, "mtmd_free")
	purego.RegisterLibFunc(&mtmdSupportVisionFn, lib, "mtmd_support_vision")
	purego.RegisterLibFunc(&mtmdSupportAudioFn, lib, "mtmd_support_audio")
	purego.RegisterLibFunc(&mtmdDecodeUseNonCausalFn, lib, "mtmd_decode_use_non_causal")
	purego.RegisterLibFunc(&mtmdDecodeUseMropeFn, lib, "mtmd_decode_use_mrope")
	purego.RegisterLibFunc(&mtmdBitmapInitFn, lib, "mtmd_bitmap_init")
	purego.RegisterLibFunc(&mtmdBitmapFreeFn, lib, "mtmd_bitmap_free")
	purego.RegisterLibFunc(&mtmdBitmapGetNxFn, lib, "mtmd_bitmap_get_nx")
	purego.RegisterLibFunc(&mtmdBitmapGetNyFn, lib, "mtmd_bitmap_get_ny")
	purego.RegisterLibFunc(&mtmdInputChunksInitFn, lib, "mtmd_input_chunks_init")
	purego.RegisterLibFunc(&mtmdInputChunksSizeFn, lib, "mtmd_input_chunks_size")
	purego.RegisterLibFunc(&mtmdInputChunksGetFn, lib, "mtmd_input_chunks_get")
	purego.RegisterLibFunc(&mtmdInputChunksFreeFn, lib, "mtmd_input_chunks_free")
	purego.RegisterLibFunc(&mtmdInputChunkGetTypeFn, lib, "mtmd_input_chunk_get_type")
	purego.RegisterLibFunc(&mtmdInputChunkGetTokensTextFn, lib, "mtmd_input_chunk_get_tokens_text")
	purego.RegisterLibFunc(&mtmdInputChunkGetNTokensFn, lib, "mtmd_input_chunk_get_n_tokens")
	purego.RegisterLibFunc(&mtmdTokenizeFn, lib, "mtmd_tokenize")
	purego.RegisterLibFunc(&mtmdEncodeChunkFn, lib, "mtmd_encode_chunk")
	purego.RegisterLibFunc(&mtmdGetOutputEmbdFn, lib, "mtmd_get_output_embd")
}

// -----------------------------------------------------------------------
// Go wrappers for mtmd C API
// -----------------------------------------------------------------------

// MtmdDefaultMarker returns the default media marker string ("<__media__>").
func MtmdDefaultMarker() string {
	ptr := mtmdDefaultMarkerFn()
	if ptr == nil {
		return "<__media__>"
	}
	return utils.BytePtrToString(ptr)
}

// MtmdContextParamsDefault returns default parameters for multi-modal context.
func MtmdContextParamsDefault() MtmdContextParams {
	return mtmdContextParamsDefaultFn()
}

// MtmdInitFromFile creates a multi-modal context from an mmproj GGUF file
// and an already-loaded text model.
func MtmdInitFromFile(mmprojPath string, textModel Model, params MtmdContextParams) (MtmdContext, error) {
	file := &[]byte(mmprojPath + "\x00")[0]
	ctx := mtmdInitFromFileFn(file, textModel, params)
	if ctx == 0 {
		return 0, loadError("mtmd_init_from_file", nil)
	}
	return ctx, nil
}

// MtmdFree releases multi-modal context resources.
func MtmdFree(ctx MtmdContext) {
	if ctx == 0 {
		return
	}
	mtmdFreeFn(ctx)
}

// MtmdSupportVision returns true if the context supports vision input.
func MtmdSupportVision(ctx MtmdContext) bool {
	if ctx == 0 {
		return false
	}
	return mtmdSupportVisionFn(ctx) != 0
}

// MtmdSupportAudio returns true if the context supports audio input.
func MtmdSupportAudio(ctx MtmdContext) bool {
	if ctx == 0 {
		return false
	}
	return mtmdSupportAudioFn(ctx) != 0
}

// MtmdDecodeUseNonCausal returns whether non-causal mask is needed before decode.
func MtmdDecodeUseNonCausal(ctx MtmdContext) bool {
	if ctx == 0 {
		return false
	}
	return mtmdDecodeUseNonCausalFn(ctx) != 0
}

// MtmdDecodeUseMrope returns whether the model uses M-RoPE.
func MtmdDecodeUseMrope(ctx MtmdContext) bool {
	if ctx == 0 {
		return false
	}
	return mtmdDecodeUseMropeFn(ctx) != 0
}

// MtmdBitmapInit creates a bitmap from raw RGB pixel data.
// data must be nx * ny * 3 bytes (RGBRGBRGB...).
func MtmdBitmapInit(nx, ny uint32, data []byte) MtmdBitmap {
	return mtmdBitmapInitFn(nx, ny, &data[0])
}

// MtmdBitmapFree releases bitmap resources.
func MtmdBitmapFree(bitmap MtmdBitmap) {
	if bitmap == 0 {
		return
	}
	mtmdBitmapFreeFn(bitmap)
}

// MtmdBitmapGetNx returns the width of the bitmap.
func MtmdBitmapGetNx(bitmap MtmdBitmap) uint32 {
	if bitmap == 0 {
		return 0
	}
	return mtmdBitmapGetNxFn(bitmap)
}

// MtmdBitmapGetNy returns the height of the bitmap.
func MtmdBitmapGetNy(bitmap MtmdBitmap) uint32 {
	if bitmap == 0 {
		return 0
	}
	return mtmdBitmapGetNyFn(bitmap)
}

// MtmdInputChunksInit creates a new empty input chunks container.
func MtmdInputChunksInit() MtmdInputChunks {
	return mtmdInputChunksInitFn()
}

// MtmdInputChunksSize returns the number of chunks in the container.
func MtmdInputChunksSize(chunks MtmdInputChunks) uint64 {
	if chunks == 0 {
		return 0
	}
	return mtmdInputChunksSizeFn(chunks)
}

// MtmdInputChunksGet returns the chunk at the given index.
func MtmdInputChunksGet(chunks MtmdInputChunks, idx uint64) MtmdInputChunk {
	if chunks == 0 {
		return 0
	}
	return mtmdInputChunksGetFn(chunks, idx)
}

// MtmdInputChunksFree releases all chunks and the container.
func MtmdInputChunksFree(chunks MtmdInputChunks) {
	if chunks == 0 {
		return
	}
	mtmdInputChunksFreeFn(chunks)
}

// MtmdInputChunkGetType returns the type of a chunk (text, image, or audio).
func MtmdInputChunkGetType(chunk MtmdInputChunk) MtmdInputChunkType {
	if chunk == 0 {
		return MtmdInputChunkTypeText
	}
	return MtmdInputChunkType(mtmdInputChunkGetTypeFn(chunk))
}

// MtmdInputChunkGetTokensText returns the text tokens and count for a text chunk.
func MtmdInputChunkGetTokensText(chunk MtmdInputChunk) ([]Token, uint64) {
	if chunk == 0 {
		return nil, 0
	}
	var nTokens uint64
	tokensPtr := mtmdInputChunkGetTokensTextFn(chunk, &nTokens)

	if tokensPtr == nil || nTokens == 0 {
		return nil, 0
	}

	tokens := unsafe.Slice(tokensPtr, nTokens)
	result := make([]Token, nTokens)
	copy(result, tokens)
	return result, nTokens
}

// MtmdInputChunkGetNTokens returns the total token count for a chunk.
func MtmdInputChunkGetNTokens(chunk MtmdInputChunk) uint64 {
	if chunk == 0 {
		return 0
	}
	return mtmdInputChunkGetNTokensFn(chunk)
}

// MtmdTokenize tokenizes input text with embedded media markers and bitmaps.
func MtmdTokenize(ctx MtmdContext, output MtmdInputChunks, text MtmdInputText, bitmaps []MtmdBitmap) int32 {
	if ctx == 0 || output == 0 {
		return -1
	}

	nBitmaps := uint64(len(bitmaps))

	var bitmapsPtr *MtmdBitmap
	if nBitmaps > 0 {
		bitmapsPtr = &bitmaps[0]
	}

	return mtmdTokenizeFn(ctx, output, &text, bitmapsPtr, nBitmaps)
}

// MtmdEncodeChunk encodes a chunk (runs CLIP for image chunks).
// Returns 0 on success.
func MtmdEncodeChunk(ctx MtmdContext, chunk MtmdInputChunk) int32 {
	if ctx == 0 || chunk == 0 {
		return -1
	}
	return mtmdEncodeChunkFn(ctx, chunk)
}

// MtmdGetOutputEmbd returns the output embeddings from the last encode pass.
func MtmdGetOutputEmbd(ctx MtmdContext) unsafe.Pointer {
	if ctx == 0 {
		return nil
	}
	return unsafe.Pointer(mtmdGetOutputEmbdFn(ctx))
}
