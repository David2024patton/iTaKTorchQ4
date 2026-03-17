//go:build !windows

package loader

import (
	"fmt"

	"github.com/ebitengine/purego"
)

// openLibrary loads a shared library on Unix platforms using dlopen.
func openLibrary(name string) (uintptr, error) {
	handle, err := purego.Dlopen(name, purego.RTLD_NOW|purego.RTLD_GLOBAL)
	if err != nil {
		return 0, fmt.Errorf("%s: error loading library: %w", name, err)
	}
	return handle, nil
}

// setDllSearchPath is a no-op on non-Windows platforms.
// On Linux/macOS, dlopen resolves dependencies from RPATH, LD_LIBRARY_PATH,
// or already-loaded shared objects in the process.
func setDllSearchPath(dir string) {
	// No-op on Linux/macOS.
}
