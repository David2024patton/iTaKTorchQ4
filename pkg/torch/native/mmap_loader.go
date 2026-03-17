// mmap_loader.go implements memory-mapped model file loading.
//
// WHY: Reading a 4GB model file with os.ReadFile copies it entirely into
// Go-managed heap memory, causing a massive GC spike. Memory-mapping
// (mmap) maps the file directly into the process address space. The OS
// pages data in on demand, giving us:
//   - Instant "load" time (only reads pages as accessed)
//   - No GC pressure (data lives outside the Go heap)
//   - Shared memory (multiple processes can share the same pages)
//   - Automatic eviction under memory pressure
//
// FALLBACK: If mmap fails (e.g. on some WSL configurations), falls back
// to standard file reading.
package native

import (
	"fmt"
	"os"
)

// MappedFile represents a memory-mapped file.
type MappedFile struct {
	Data     []byte // mmap'd data (or read data as fallback)
	Size     int64
	isMapped bool // true if using mmap, false if using read fallback
	file     *os.File
}

// MMapFile opens a file and memory-maps it into the process address space.
// On failure, falls back to reading the entire file into memory.
func MMapFile(path string) (*MappedFile, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open %s: %w", path, err)
	}

	stat, err := f.Stat()
	if err != nil {
		f.Close()
		return nil, fmt.Errorf("stat %s: %w", path, err)
	}

	size := stat.Size()

	// Try platform-specific mmap.
	data, err := mmapFile(f, size)
	if err == nil {
		fmt.Printf("[MMap] Memory-mapped %s (%.1f MB)\n", path, float64(size)/(1024*1024))
		return &MappedFile{
			Data:     data,
			Size:     size,
			isMapped: true,
			file:     f,
		}, nil
	}

	// Fallback: read the entire file.
	fmt.Printf("[MMap] mmap failed (%v), falling back to read()\n", err)
	data = make([]byte, size)
	_, err = f.ReadAt(data, 0)
	f.Close()
	if err != nil {
		return nil, fmt.Errorf("read %s: %w", path, err)
	}

	return &MappedFile{
		Data:     data,
		Size:     size,
		isMapped: false,
	}, nil
}

// Close unmaps and closes the file.
func (m *MappedFile) Close() error {
	if m.isMapped && m.Data != nil {
		munmapFile(m.Data)
		m.Data = nil
	}
	if m.file != nil {
		return m.file.Close()
	}
	return nil
}
