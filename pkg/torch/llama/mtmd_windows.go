//go:build windows

package llama

import (
	"unsafe"

	"github.com/jupiterrider/ffi"
)

var (
	ffiTypeMtmdContextParams = ffi.NewType(
		&ffi.TypeUint8,   // use_gpu
		&ffi.TypeUint8,   // print_timings
		&ffi.TypeSint32,  // n_threads
		&ffi.TypePointer, // image_marker
		&ffi.TypePointer, // media_marker
		&ffi.TypeSint32,  // flash_attn_type
		&ffi.TypeUint8,   // warmup
		&ffi.TypeSint32,  // image_min_tokens
		&ffi.TypeSint32,  // image_max_tokens
		&ffi.TypePointer, // cb_eval
		&ffi.TypePointer, // cb_eval_user_data
	)
)

func loadMtmdFuncs(lib uintptr) error {
	var err error
	ffiLib := ffi.Lib{Addr: lib}

	var mtmdContextParamsDefaultFFI ffi.Fun
	if mtmdContextParamsDefaultFFI, err = ffiLib.Prep("mtmd_context_params_default", &ffiTypeMtmdContextParams); err != nil {
		return loadError("mtmd_context_params_default", err)
	}
	mtmdContextParamsDefaultFn = func() MtmdContextParams {
		var p MtmdContextParams
		mtmdContextParamsDefaultFFI.Call(unsafe.Pointer(&p))
		return p
	}

	var mtmdInitFromFileFFI ffi.Fun
	if mtmdInitFromFileFFI, err = ffiLib.Prep("mtmd_init_from_file", &ffi.TypePointer, &ffi.TypePointer, &ffi.TypePointer, &ffiTypeMtmdContextParams); err != nil {
		return loadError("mtmd_init_from_file", err)
	}
	mtmdInitFromFileFn = func(mmprojFname *byte, textModel Model, params MtmdContextParams) MtmdContext {
		var ctx MtmdContext
		mtmdInitFromFileFFI.Call(unsafe.Pointer(&ctx), unsafe.Pointer(&mmprojFname), unsafe.Pointer(&textModel), unsafe.Pointer(&params))
		return ctx
	}

	loadMtmdPuregoFuncs(lib)
	mtmdAvailable = true
	return nil
}
