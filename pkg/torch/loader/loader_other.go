//go:build !windows

package loader

import (
	"fmt"
	"os"
	"sync"

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

var ldPathOnce sync.Once

// setDllSearchPath on Linux/macOS prepends the lib directory to LD_LIBRARY_PATH
// so that dlopen and llama.cpp's BackendLoadAll can resolve CPU-optimized
// backend variants (libggml-cpu-haswell.so, libggml-cpu-alderlake.so, etc.)
// that live alongside the base libraries.
func setDllSearchPath(dir string) {
	ldPathOnce.Do(func() {
		currentPath := os.Getenv("LD_LIBRARY_PATH")
		if currentPath == "" {
			os.Setenv("LD_LIBRARY_PATH", dir)
		} else {
			os.Setenv("LD_LIBRARY_PATH", dir+":"+currentPath)
		}
	})
}

