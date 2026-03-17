//go:build !windows

package llama

import (
	"unsafe"

	"github.com/ebitengine/purego"
	"github.com/jupiterrider/ffi"
)

var (
	ffiTypeSize = ffi.TypeUint64

	ffiTypeModelParams = ffi.NewType(&ffi.TypePointer, &ffi.TypePointer, &ffi.TypeSint32,
		&ffi.TypeSint32, &ffi.TypeSint32,
		&ffi.TypePointer, &ffi.TypePointer, &ffi.TypePointer, &ffi.TypePointer,
		&ffi.TypeUint8, &ffi.TypeUint8, &ffi.TypeUint8, &ffi.TypeUint8, &ffi.TypeUint8,
		&ffi.TypeUint8, &ffi.TypeUint8, &ffi.TypeUint8)

	ffiTypeModelQuantizeParams = ffi.NewType(&ffi.TypeSint32, &ffi.TypeSint32,
		&ffi.TypeSint32, &ffi.TypeSint32, &ffi.TypeUint8, &ffi.TypeUint8, &ffi.TypeUint8, &ffi.TypeUint8, &ffi.TypeUint8, &ffi.TypeUint8,
		&ffi.TypePointer, &ffi.TypePointer, &ffi.TypePointer, &ffi.TypePointer)
)

func loadModelFuncs(lib uintptr) error {
	var err error
	ffiLib := ffi.Lib{Addr: lib}

	// Large-struct functions must use ffi because:
	// - purego.RegisterLibFunc panics on structs >16 bytes (flattens to too many args)
	// - purego.SyscallN can't pass struct data by value on the stack
	// ffi handles the ABI correctly on all platforms.

	var modelDefaultParamsFFI ffi.Fun
	if modelDefaultParamsFFI, err = ffiLib.Prep("llama_model_default_params", &ffiTypeModelParams); err != nil {
		return loadError("llama_model_default_params", err)
	}
	modelDefaultParamsFn = func() ModelParams {
		var p ModelParams
		modelDefaultParamsFFI.Call(unsafe.Pointer(&p))
		return p
	}

	var modelLoadFromFileFFI ffi.Fun
	if modelLoadFromFileFFI, err = ffiLib.Prep("llama_model_load_from_file", &ffi.TypePointer, &ffi.TypePointer, &ffiTypeModelParams); err != nil {
		return loadError("llama_model_load_from_file", err)
	}
	modelLoadFromFileFn = func(pathModel *byte, params ModelParams) Model {
		var model Model
		modelLoadFromFileFFI.Call(unsafe.Pointer(&model), unsafe.Pointer(&pathModel), unsafe.Pointer(&params))
		return model
	}

	var modelLoadFromSplitsFFI ffi.Fun
	if modelLoadFromSplitsFFI, err = ffiLib.Prep("llama_model_load_from_splits", &ffi.TypePointer, &ffi.TypePointer, &ffiTypeSize, &ffiTypeModelParams); err != nil {
		return loadError("llama_model_load_from_splits", err)
	}
	modelLoadFromSplitsFn = func(paths unsafe.Pointer, nPaths uint64, params ModelParams) Model {
		var model Model
		modelLoadFromSplitsFFI.Call(unsafe.Pointer(&model), &paths, &nPaths, unsafe.Pointer(&params))
		return model
	}

	var initFromModelFFI ffi.Fun
	if initFromModelFFI, err = ffiLib.Prep("llama_init_from_model", &ffi.TypePointer, &ffi.TypePointer, &ffiTypeContextParams); err != nil {
		return loadError("llama_init_from_model", err)
	}
	initFromModelFn = func(model Model, params ContextParams) Context {
		var ctx Context
		initFromModelFFI.Call(unsafe.Pointer(&ctx), unsafe.Pointer(&model), unsafe.Pointer(&params))
		return ctx
	}

	var modelQuantizeDefaultParamsFFI ffi.Fun
	if modelQuantizeDefaultParamsFFI, err = ffiLib.Prep("llama_model_quantize_default_params", &ffiTypeModelQuantizeParams); err != nil {
		return loadError("llama_model_quantize_default_params", err)
	}
	modelQuantizeDefaultParamsFn = func() ModelQuantizeParams {
		var p ModelQuantizeParams
		modelQuantizeDefaultParamsFFI.Call(unsafe.Pointer(&p))
		return p
	}

	var modelQuantizeFFI ffi.Fun
	if modelQuantizeFFI, err = ffiLib.Prep("llama_model_quantize", &ffi.TypeUint32, &ffi.TypePointer, &ffi.TypePointer, &ffi.TypePointer); err != nil {
		return loadError("llama_model_quantize", err)
	}
	modelQuantizeFn = func(fnameInp *byte, fnameOut *byte, params *ModelQuantizeParams) uint32 {
		var result ffi.Arg
		modelQuantizeFFI.Call(unsafe.Pointer(&result), unsafe.Pointer(&fnameInp), unsafe.Pointer(&fnameOut), unsafe.Pointer(&params))
		return uint32(result)
	}

	var modelParamsFitFFI ffi.Fun
	if modelParamsFitFFI, err = ffiLib.Prep("llama_params_fit", &ffi.TypeSint32, &ffi.TypePointer, &ffi.TypePointer, &ffi.TypePointer, &ffi.TypePointer, &ffi.TypePointer, &ffi.TypePointer, &ffi.TypeUint32, &ffi.TypeSint32); err != nil {
		return loadError("llama_params_fit", err)
	}
	modelParamsFitFn = func(pathModel *byte, mparams *ModelParams, cparams *ContextParams, tensorSplit *float32, tensorBuftOverrides *TensorBuftOverride, margins *uint64, nCtxMin uint32, logLevel LogLevel) int32 {
		var result ffi.Arg
		modelParamsFitFFI.Call(
			unsafe.Pointer(&result),
			unsafe.Pointer(&pathModel),
			unsafe.Pointer(&mparams),
			unsafe.Pointer(&cparams),
			unsafe.Pointer(&tensorSplit),
			unsafe.Pointer(&tensorBuftOverrides),
			unsafe.Pointer(&margins),
			&nCtxMin,
			&logLevel,
		)
		return int32(result)
	}

	loadModelPuregoFuncs(lib)
	return nil
}

// SetProgressCallback sets a progress callback for model loading.
// On Linux, use purego.NewCallback for the progress callback.
func (p *ModelParams) SetProgressCallback(cb ProgressCallback) {
	if cb == nil {
		p.ProgressCallback = uintptr(0)
		return
	}

	callback := purego.NewCallback(func(progress float32, userData uintptr) uintptr {
		result := cb(progress, userData)
		return uintptr(result)
	})

	p.ProgressCallback = callback
}
