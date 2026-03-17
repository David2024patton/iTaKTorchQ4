//go:build !windows

package native

import (
	"fmt"
	"os"
	"syscall"
)

// mmapFile maps a file into memory using Unix mmap.
func mmapFile(f *os.File, size int64) ([]byte, error) {
	data, err := syscall.Mmap(int(f.Fd()), 0, int(size),
		syscall.PROT_READ, syscall.MAP_SHARED)
	if err != nil {
		return nil, fmt.Errorf("mmap: %w", err)
	}
	return data, nil
}

// munmapFile unmaps a previously mapped file.
func munmapFile(data []byte) {
	syscall.Munmap(data)
}
