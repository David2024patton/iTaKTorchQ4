//go:build windows

package loader

import (
	"fmt"
	"os"
	"sync"
	"syscall"
	"unsafe"
)

// openLibrary loads a DLL on Windows and returns the handle as uintptr.
func openLibrary(name string) (uintptr, error) {
	handle, err := syscall.LoadLibrary(name)
	if err != nil {
		return 0, fmt.Errorf("%s: error loading library: %w", name, err)
	}
	return uintptr(handle), nil
}

var dllDirOnce sync.Once

// setDllSearchPath configures Windows to find dependent DLLs (like llama.dll,
// ggml.dll) when loading plugin DLLs (like mtmd.dll) that import them.
//
// Uses three strategies:
// 1. SetDllDirectoryW - adds to standard DLL search path
// 2. AddDllDirectory  - adds to the "safe DLL search" list (Win8+)
// 3. PATH prepend     - fallback for LoadLibraryExW LOAD_WITH_ALTERED_SEARCH_PATH
func setDllSearchPath(dir string) {
	dllDirOnce.Do(func() {
		kernel32 := syscall.NewLazyDLL("kernel32.dll")

		// Strategy 1: SetDllDirectoryW
		setDllDir := kernel32.NewProc("SetDllDirectoryW")
		dirUTF16, err := syscall.UTF16PtrFromString(dir)
		if err != nil {
			return
		}
		setDllDir.Call(uintptr(unsafe.Pointer(dirUTF16)))

		// Strategy 2: AddDllDirectory (Windows 8+)
		addDllDir := kernel32.NewProc("AddDllDirectory")
		if addDllDir.Find() == nil {
			addDllDir.Call(uintptr(unsafe.Pointer(dirUTF16)))
		}

		// Strategy 3: Prepend to PATH environment variable
		currentPath := os.Getenv("PATH")
		os.Setenv("PATH", dir+";"+currentPath)
	})
}
