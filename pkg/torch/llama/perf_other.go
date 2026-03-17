//go:build !windows

package llama

import (
	"unsafe"

	"github.com/jupiterrider/ffi"
)

var (
	ffiPerfContextData = ffi.NewType(&ffi.TypeDouble, &ffi.TypeDouble, &ffi.TypeDouble, &ffi.TypeDouble, &ffi.TypeSint32, &ffi.TypeSint32, &ffi.TypeSint32)
	ffiPerfSamplerData = ffi.NewType(&ffi.TypeDouble, &ffi.TypeSint32)
)

func loadPerfFuncs(lib uintptr) error {
	var err error
	ffiLib := ffi.Lib{Addr: lib}

	// PerfContextData is 44 bytes (>16 bytes) - must use ffi.
	var perfContextFFI ffi.Fun
	if perfContextFFI, err = ffiLib.Prep("llama_perf_context", &ffiPerfContextData, &ffi.TypePointer); err != nil {
		return loadError("llama_perf_context", err)
	}
	perfContextFn = func(ctx Context) PerfContextData {
		var data PerfContextData
		perfContextFFI.Call(unsafe.Pointer(&data), unsafe.Pointer(&ctx))
		return data
	}

	// PerfSamplerData is 12 bytes (<= 16 bytes) - works with ffi too.
	var perfSamplerFFI ffi.Fun
	if perfSamplerFFI, err = ffiLib.Prep("llama_perf_sampler", &ffiPerfSamplerData, &ffi.TypePointer); err != nil {
		return loadError("llama_perf_sampler", err)
	}
	perfSamplerFn = func(chain Sampler) PerfSamplerData {
		var data PerfSamplerData
		perfSamplerFFI.Call(unsafe.Pointer(&data), unsafe.Pointer(&chain))
		return data
	}

	loadPerfPuregoFuncs(lib)
	return nil
}
