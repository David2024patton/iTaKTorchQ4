//go:build windows

package torch

import (
	"fmt"
	"os"
	"syscall"
	"unsafe"
)

var (
	prefaultKernel32          = syscall.NewLazyDLL("kernel32.dll")
	procPrefetchVirtualMemory = prefaultKernel32.NewProc("PrefetchVirtualMemory")
	procCreateFileMappingW    = prefaultKernel32.NewProc("CreateFileMappingW")
	procMapViewOfFile         = prefaultKernel32.NewProc("MapViewOfFile")
	procUnmapViewOfFile       = prefaultKernel32.NewProc("UnmapViewOfFile")
)

// prefaultPlatform uses PrefetchVirtualMemory (Win8+) to pre-fault mmap'd pages.
func prefaultPlatform(f *os.File, size int64) error {
	if procPrefetchVirtualMemory.Find() == nil {
		return prefetchViaMapping(f, size)
	}
	return fmt.Errorf("PrefetchVirtualMemory not available")
}

func prefetchViaMapping(f *os.File, size int64) error {
	handle := syscall.Handle(f.Fd())

	mapHandle, _, err := procCreateFileMappingW.Call(
		uintptr(handle),
		0,
		0x02, // PAGE_READONLY
		uintptr(uint32(size>>32)),
		uintptr(uint32(size)),
		0,
	)
	if mapHandle == 0 {
		return fmt.Errorf("CreateFileMappingW: %w", err)
	}
	defer syscall.CloseHandle(syscall.Handle(mapHandle))

	viewAddr, _, err := procMapViewOfFile.Call(
		mapHandle,
		0x04, // FILE_MAP_READ
		0, 0,
		0,
	)
	if viewAddr == 0 {
		return fmt.Errorf("MapViewOfFile: %w", err)
	}
	defer procUnmapViewOfFile.Call(viewAddr)

	type memoryRangeEntry struct {
		VirtualAddress uintptr
		NumberOfBytes  uintptr
	}
	entry := memoryRangeEntry{
		VirtualAddress: viewAddr,
		NumberOfBytes:  uintptr(size),
	}

	currentProcess := uintptr(^uintptr(0))
	ret, _, _ := procPrefetchVirtualMemory.Call(
		currentProcess,
		1,
		uintptr(unsafe.Pointer(&entry)),
		0,
	)
	if ret == 0 {
		return fmt.Errorf("PrefetchVirtualMemory failed")
	}

	return nil
}
