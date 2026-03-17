//go:build windows

package llama

import (
	"github.com/jupiterrider/ffi"
)

func loadBackendFuncs(lib uintptr) error {
	loadBackendPuregoFuncs(lib)

	// Optional: ggml_backend_load_all may not exist in older builds.
	ffiLib := ffi.Lib{Addr: lib}
	var backendLoadAllFFI ffi.Fun
	var err error
	if backendLoadAllFFI, err = ffiLib.Prep("ggml_backend_load_all", &ffi.TypeVoid); err == nil {
		hasBackendLoadAll = true
		backendLoadAllFn = func() {
			backendLoadAllFFI.Call(nil)
		}
	}

	return nil
}
