//go:build windows

package llama

import (
	"unsafe"

	"github.com/jupiterrider/ffi"
)

var (
	// Shared FFI type definitions for Windows
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

var progressCallback unsafe.Pointer
var sizeOfClosure = unsafe.Sizeof(ffi.Closure{})

func loadModelFuncs(lib uintptr) error {
	var err error
	ffiLib := ffi.Lib{Addr: lib}

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
// On Windows, uses ffi.ClosureAlloc to create a C-compatible closure.
func (p *ModelParams) SetProgressCallback(cb ProgressCallback) {
	if cb == nil {
		p.ProgressCallback = uintptr(0)
		return
	}

	closure := ffi.ClosureAlloc(sizeOfClosure, &progressCallback)

	fn := ffi.NewCallback(func(cif *ffi.Cif, ret unsafe.Pointer, args *unsafe.Pointer, userData unsafe.Pointer) uintptr {
		if args == nil || ret == nil {
			return 1
		}

		arg := unsafe.Slice(args, cif.NArgs)
		progress := *(*float32)(arg[0])
		userDataPtr := *(*uintptr)(arg[1])
		result := cb(progress, userDataPtr)
		*(*uint8)(ret) = result
		return 0
	})

	var cifCallback ffi.Cif
	if status := ffi.PrepCif(&cifCallback, ffi.DefaultAbi, 2, &ffi.TypeUint8, &ffi.TypeFloat, &ffi.TypePointer); status != ffi.OK {
		panic(status)
	}

	if closure != nil {
		if status := ffi.PrepClosureLoc(closure, &cifCallback, fn, nil, progressCallback); status != ffi.OK {
			panic(status)
		}
	}

	p.ProgressCallback = uintptr(progressCallback)
}
