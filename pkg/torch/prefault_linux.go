//go:build linux

package torch

import (
	"os"
	"syscall"
)

// prefaultPlatform uses posix_fadvise(POSIX_FADV_WILLNEED) to tell the kernel
// to read the entire file into the page cache asynchronously.
func prefaultPlatform(f *os.File, size int64) error {
	// POSIX_FADV_WILLNEED = 3
	_, _, errno := syscall.Syscall6(
		syscall.SYS_FADVISE64,
		f.Fd(),
		0,
		uintptr(size),
		3, // POSIX_FADV_WILLNEED
		0, 0,
	)
	if errno != 0 {
		return errno
	}
	return nil
}
