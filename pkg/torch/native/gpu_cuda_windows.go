//go:build cuda && windows

package native

import (
	"fmt"
	"syscall"
)

// loadLib loads a DLL by trying each name in order (Windows: LoadDLL).
func loadLib(names []string) (uintptr, error) {
	var lastErr error
	for _, name := range names {
		dll, err := syscall.LoadDLL(name)
		if err == nil {
			return uintptr(dll.Handle), nil
		}
		lastErr = fmt.Errorf("LoadDLL %s: %w", name, err)
	}
	return 0, lastErr
}
