//go:build !windows

package llama

import (
	"unsafe"

	"github.com/jupiterrider/ffi"
)

var ffiTypeContextParams = ffi.NewType(
	&ffi.TypeUint32, &ffi.TypeUint32,
	&ffi.TypeUint32, &ffi.TypeUint32,
	&ffi.TypeSint32, &ffi.TypeSint32,
	&ffi.TypeSint32, &ffi.TypeSint32,
	&ffi.TypeSint32, &ffi.TypeSint32,
	&ffi.TypeFloat, &ffi.TypeFloat,
	&ffi.TypeFloat, &ffi.TypeFloat,
	&ffi.TypeFloat, &ffi.TypeFloat,
	&ffi.TypeUint32, &ffi.TypeFloat,
	&ffi.TypePointer, &ffi.TypePointer,
	&ffi.TypeSint32, &ffi.TypeSint32,
	&ffi.TypePointer, &ffi.TypePointer,
	&ffi.TypeUint8, &ffi.TypeUint8,
	&ffi.TypeUint8, &ffi.TypeUint8,
	&ffi.TypeUint8, &ffi.TypeUint8,
	&ffi.TypePointer, &ffi.TypeUint64)

func loadContextFuncs(lib uintptr) error {
	var err error
	ffiLib := ffi.Lib{Addr: lib}

	var contextDefaultParamsFFI ffi.Fun
	if contextDefaultParamsFFI, err = ffiLib.Prep("llama_context_default_params", &ffiTypeContextParams); err != nil {
		return loadError("llama_context_default_params", err)
	}
	contextDefaultParamsFn = func() ContextParams {
		var p ContextParams
		contextDefaultParamsFFI.Call(unsafe.Pointer(&p))
		return p
	}

	var encodeFFI ffi.Fun
	if encodeFFI, err = ffiLib.Prep("llama_encode", &ffi.TypeSint32, &ffi.TypePointer, &ffiTypeBatch); err != nil {
		return loadError("llama_encode", err)
	}
	encodeFn = func(ctx Context, batch Batch) int32 {
		var result ffi.Arg
		encodeFFI.Call(unsafe.Pointer(&result), unsafe.Pointer(&ctx), unsafe.Pointer(&batch))
		return int32(result)
	}

	var decodeFFI ffi.Fun
	if decodeFFI, err = ffiLib.Prep("llama_decode", &ffi.TypeSint32, &ffi.TypePointer, &ffiTypeBatch); err != nil {
		return loadError("llama_decode", err)
	}
	decodeFn = func(ctx Context, batch Batch) int32 {
		var result ffi.Arg
		decodeFFI.Call(unsafe.Pointer(&result), unsafe.Pointer(&ctx), unsafe.Pointer(&batch))
		return int32(result)
	}

	loadContextPuregoFuncs(lib)
	return nil
}
