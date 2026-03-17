package llama

import (
	"unsafe"

	"github.com/David2024patton/iTaKTorch/pkg/torch/utils"
	"github.com/ebitengine/purego"
)

// --- purego direct-call function pointers ---
var (
	stateSaveFileFn      func(ctx Context, pathSession *byte, tokens *Token, nTokenCount uintptr) uint8
	stateLoadFileFn      func(ctx Context, pathSession *byte, tokensOut *Token, nTokenCapacity uintptr, nTokenCountOut *uint64) uint8
	stateGetSizeFn       func(ctx Context) uintptr
	stateGetDataFn       func(ctx Context, dst *byte, size uintptr) uintptr
	stateSetDataFn       func(ctx Context, src *byte, size uintptr) uintptr
	stateSeqGetSizeFn    func(ctx Context, seqId int32) uintptr
	stateSeqGetDataFn    func(ctx Context, dst *byte, size uintptr, seqId int32) uintptr
	stateSeqSetDataFn    func(ctx Context, src *byte, size uintptr, seqId int32) uintptr
	stateSeqSaveFileFn   func(ctx Context, filepath *byte, seqId int32, tokens *Token, nTokenCount uintptr) uintptr
	stateSeqLoadFileFn   func(ctx Context, filepath *byte, destSeqId int32, tokensOut *Token, nTokenCapacity uintptr, nTokenCountOut *uint64) uintptr
	stateSeqGetSizeExtFn func(ctx Context, seqId int32, flags uint32) uintptr
	stateSeqGetDataExtFn func(ctx Context, dst *byte, size uintptr, seqId int32, flags uint32) uintptr
	stateSeqSetDataExtFn func(ctx Context, src *byte, size uintptr, destSeqId int32, flags uint32) uintptr
)

func loadStateFuncs(lib uintptr) error {
	purego.RegisterLibFunc(&stateSaveFileFn, lib, "llama_state_save_file")
	purego.RegisterLibFunc(&stateLoadFileFn, lib, "llama_state_load_file")
	purego.RegisterLibFunc(&stateGetSizeFn, lib, "llama_state_get_size")
	purego.RegisterLibFunc(&stateGetDataFn, lib, "llama_state_get_data")
	purego.RegisterLibFunc(&stateSetDataFn, lib, "llama_state_set_data")
	purego.RegisterLibFunc(&stateSeqGetSizeFn, lib, "llama_state_seq_get_size")
	purego.RegisterLibFunc(&stateSeqGetDataFn, lib, "llama_state_seq_get_data")
	purego.RegisterLibFunc(&stateSeqSetDataFn, lib, "llama_state_seq_set_data")
	purego.RegisterLibFunc(&stateSeqSaveFileFn, lib, "llama_state_seq_save_file")
	purego.RegisterLibFunc(&stateSeqLoadFileFn, lib, "llama_state_seq_load_file")
	purego.RegisterLibFunc(&stateSeqGetSizeExtFn, lib, "llama_state_seq_get_size_ext")
	purego.RegisterLibFunc(&stateSeqGetDataExtFn, lib, "llama_state_seq_get_data_ext")
	purego.RegisterLibFunc(&stateSeqSetDataExtFn, lib, "llama_state_seq_set_data_ext")
	return nil
}

// StateSaveFile saves the state to a file and returns true on success.
func StateSaveFile(ctx Context, path string, tokens []Token) bool {
	if ctx == 0 {
		return false
	}
	pathPtr, _ := utils.BytePtrFromString(path)
	var toks *Token
	if len(tokens) > 0 {
		toks = unsafe.SliceData(tokens)
	}
	return stateSaveFileFn(ctx, pathPtr, toks, uintptr(len(tokens))) != 0
}

// StateLoadFile loads the state from a file and returns true on success.
func StateLoadFile(ctx Context, path string, tokensOut []Token, nTokenCapacity uint64, nTokenCountOut *uint64) bool {
	if ctx == 0 {
		return false
	}
	pathPtr, _ := utils.BytePtrFromString(path)
	var toks *Token
	if len(tokensOut) > 0 {
		toks = unsafe.SliceData(tokensOut)
	}
	return stateLoadFileFn(ctx, pathPtr, toks, uintptr(nTokenCapacity), nTokenCountOut) != 0
}

// StateGetSize returns the actual size in bytes of the state.
func StateGetSize(ctx Context) uint64 {
	if ctx == 0 {
		return 0
	}
	return uint64(stateGetSizeFn(ctx))
}

// StateGetData copies the state to the specified destination address.
func StateGetData(ctx Context, dst []byte) uint64 {
	if ctx == 0 {
		return 0
	}
	var dstPtr *byte
	if len(dst) > 0 {
		dstPtr = &dst[0]
	}
	return uint64(stateGetDataFn(ctx, dstPtr, uintptr(len(dst))))
}

// StateSetData sets the state by reading from the specified address.
func StateSetData(ctx Context, src []byte) uint64 {
	if ctx == 0 {
		return 0
	}
	var srcPtr *byte
	if len(src) > 0 {
		srcPtr = &src[0]
	}
	return uint64(stateSetDataFn(ctx, srcPtr, uintptr(len(src))))
}

// StateSeqGetSize returns the exact size needed to copy the state of a single sequence.
func StateSeqGetSize(ctx Context, seqId SeqId) uint64 {
	if ctx == 0 {
		return 0
	}
	return uint64(stateSeqGetSizeFn(ctx, int32(seqId)))
}

// StateSeqGetData copies the state of a single sequence into the specified buffer.
func StateSeqGetData(ctx Context, dst []byte, seqId SeqId) uint64 {
	if ctx == 0 {
		return 0
	}
	var dstPtr *byte
	if len(dst) > 0 {
		dstPtr = &dst[0]
	}
	return uint64(stateSeqGetDataFn(ctx, dstPtr, uintptr(len(dst)), int32(seqId)))
}

// StateSeqSetData copies the sequence data into the specified sequence.
func StateSeqSetData(ctx Context, src []byte, destSeqId SeqId) uint64 {
	if ctx == 0 {
		return 0
	}
	var srcPtr *byte
	if len(src) > 0 {
		srcPtr = &src[0]
	}
	return uint64(stateSeqSetDataFn(ctx, srcPtr, uintptr(len(src)), int32(destSeqId)))
}

// StateSeqSaveFile saves the state of a single sequence to a file.
func StateSeqSaveFile(ctx Context, filepath string, seqId SeqId, tokens []Token) uint64 {
	if ctx == 0 {
		return 0
	}
	pathPtr, _ := utils.BytePtrFromString(filepath)
	var toks *Token
	if len(tokens) > 0 {
		toks = unsafe.SliceData(tokens)
	}
	return uint64(stateSeqSaveFileFn(ctx, pathPtr, int32(seqId), toks, uintptr(len(tokens))))
}

// StateSeqLoadFile loads the state of a single sequence from a file.
func StateSeqLoadFile(ctx Context, filepath string, destSeqId SeqId, tokensOut []Token, nTokenCapacity uint64, nTokenCountOut *uint64) uint64 {
	if ctx == 0 {
		return 0
	}
	pathPtr, _ := utils.BytePtrFromString(filepath)
	var toks *Token
	if len(tokensOut) > 0 {
		toks = unsafe.SliceData(tokensOut)
	}
	return uint64(stateSeqLoadFileFn(ctx, pathPtr, int32(destSeqId), toks, uintptr(nTokenCapacity), nTokenCountOut))
}

// StateSeqGetSizeExt returns the size needed for a sequence with flags.
func StateSeqGetSizeExt(ctx Context, seqId SeqId, flags uint32) uint64 {
	if ctx == 0 {
		return 0
	}
	return uint64(stateSeqGetSizeExtFn(ctx, int32(seqId), flags))
}

// StateSeqGetDataExt copies the state of a sequence with flags into the buffer.
func StateSeqGetDataExt(ctx Context, dst []byte, seqId SeqId, flags uint32) uint64 {
	if ctx == 0 {
		return 0
	}
	var dstPtr *byte
	if len(dst) > 0 {
		dstPtr = &dst[0]
	}
	return uint64(stateSeqGetDataExtFn(ctx, dstPtr, uintptr(len(dst)), int32(seqId), flags))
}

// StateSeqSetDataExt sets the state of a sequence with flags from the buffer.
func StateSeqSetDataExt(ctx Context, src []byte, destSeqId SeqId, flags uint32) uint64 {
	if ctx == 0 {
		return 0
	}
	var srcPtr *byte
	if len(src) > 0 {
		srcPtr = &src[0]
	}
	return uint64(stateSeqSetDataExtFn(ctx, srcPtr, uintptr(len(src)), int32(destSeqId), flags))
}
