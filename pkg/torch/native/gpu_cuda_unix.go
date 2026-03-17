//go:build cuda && !windows

package native

import "github.com/ebitengine/purego"

// loadLib loads a shared library by trying each name in order (Unix: dlopen).
func loadLib(names []string) (uintptr, error) {
	var lastErr error
	for _, name := range names {
		lib, err := purego.Dlopen(name, purego.RTLD_LAZY)
		if err == nil {
			return lib, nil
		}
		lastErr = err
	}
	return 0, lastErr
}
