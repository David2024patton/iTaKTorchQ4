//go:build windows

package llama

import (
	"unsafe"

	"github.com/jupiterrider/ffi"
)

// ffiTypeBatch represents the C struct llama_batch for ffi on Windows.
var ffiTypeBatch = ffi.NewType(&ffi.TypeSint32,
	&ffi.TypePointer, &ffi.TypePointer,
	&ffi.TypePointer, &ffi.TypePointer,
	&ffi.TypePointer, &ffi.TypePointer)

func loadBatchFuncs(lib uintptr) error {
	var err error
	ffiLib := ffi.Lib{Addr: lib}

	var batchInitFFI ffi.Fun
	if batchInitFFI, err = ffiLib.Prep("llama_batch_init", &ffiTypeBatch, &ffi.TypeSint32, &ffi.TypeSint32, &ffi.TypeSint32); err != nil {
		return loadError("llama_batch_init", err)
	}
	batchInitFn = func(nTokens, embd, nSeqMax int32) Batch {
		var batch Batch
		batchInitFFI.Call(unsafe.Pointer(&batch), &nTokens, &embd, &nSeqMax)
		return batch
	}

	var batchFreeFFI ffi.Fun
	if batchFreeFFI, err = ffiLib.Prep("llama_batch_free", &ffi.TypeVoid, &ffiTypeBatch); err != nil {
		return loadError("llama_batch_free", err)
	}
	batchFreeFn = func(batch Batch) {
		batchFreeFFI.Call(nil, unsafe.Pointer(&batch))
	}

	var batchGetOneFFI ffi.Fun
	if batchGetOneFFI, err = ffiLib.Prep("llama_batch_get_one", &ffiTypeBatch, &ffi.TypePointer, &ffi.TypeSint32); err != nil {
		return loadError("llama_batch_get_one", err)
	}
	batchGetOneFn = func(tokens *Token, nTokens int32) Batch {
		var batch Batch
		batchGetOneFFI.Call(unsafe.Pointer(&batch), unsafe.Pointer(&tokens), &nTokens)
		return batch
	}

	return nil
}
