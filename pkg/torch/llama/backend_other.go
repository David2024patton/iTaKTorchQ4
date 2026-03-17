//go:build !windows

package llama

import (
	"github.com/ebitengine/purego"
)

func loadBackendFuncs(lib uintptr) error {
	loadBackendPuregoFuncs(lib)

	// Optional: ggml_backend_load_all may not exist in older builds.
	// Use Dlsym to check if symbol exists before registering.
	sym, err := purego.Dlsym(lib, "ggml_backend_load_all")
	if err == nil && sym != 0 {
		purego.RegisterLibFunc(&backendLoadAllFn, lib, "ggml_backend_load_all")
		hasBackendLoadAll = true
	}

	return nil
}
