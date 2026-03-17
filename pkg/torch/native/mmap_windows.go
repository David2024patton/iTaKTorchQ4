//go:build windows

package native

import (
	"fmt"
	"os"
	"syscall"
	"unsafe"
)

// mmapFile maps a file into memory using Windows CreateFileMapping + MapViewOfFile.
func mmapFile(f *os.File, size int64) ([]byte, error) {
	// Create a file mapping object.
	sizeHigh := uint32(size >> 32)
	sizeLow := uint32(size & 0xFFFFFFFF)

	handle, err := syscall.CreateFileMapping(
		syscall.Handle(f.Fd()),
		nil,
		syscall.PAGE_READONLY,
		sizeHigh,
		sizeLow,
		nil,
	)
	if err != nil {
		return nil, fmt.Errorf("CreateFileMapping: %w", err)
	}

	// Map the file into the process address space.
	addr, err := syscall.MapViewOfFile(
		handle,
		syscall.FILE_MAP_READ,
		0, 0,
		uintptr(size),
	)
	if err != nil {
		syscall.CloseHandle(handle)
		return nil, fmt.Errorf("MapViewOfFile: %w", err)
	}

	// Convert the mapped address to a byte slice.
	// This unsafe.Pointer conversion is correct: addr is a valid memory
	// address returned by Windows MapViewOfFile (Win32 kernel API).
	data := unsafe.Slice((*byte)(unsafe.Pointer(addr)), int(size))
	return data, nil
}

// munmapFile unmaps a previously mapped file on Windows.
func munmapFile(data []byte) {
	if len(data) > 0 {
		syscall.UnmapViewOfFile(uintptr(unsafe.Pointer(&data[0])))
	}
}
