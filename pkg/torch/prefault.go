// prefault.go provides cross-platform memory pre-faulting for mmap'd model files.
//
// WHY: When llama.cpp loads a model with mmap, pages are loaded lazily on first
// access. This causes page fault stalls during the first inference pass, adding
// 200-500ms of latency. Pre-faulting forces the OS to load all pages into RAM
// upfront, eliminating those stalls.
package torch

import (
	"fmt"
	"io"
	"os"
	"time"
)

// PrefaultModelFile pre-faults the model file into the OS page cache.
// This ensures mmap'd pages are resident in RAM before first inference,
// eliminating page fault stalls that add 200-500ms to first-token latency.
func PrefaultModelFile(path string) error {
	start := time.Now()

	f, err := os.Open(path)
	if err != nil {
		return fmt.Errorf("prefault open: %w", err)
	}
	defer f.Close()

	fi, err := f.Stat()
	if err != nil {
		return fmt.Errorf("prefault stat: %w", err)
	}
	sizeMB := fi.Size() / (1024 * 1024)

	// Try platform-specific optimized path first.
	if err := prefaultPlatform(f, fi.Size()); err == nil {
		fmt.Printf("[iTaK Torch] Pre-faulted %d MB in %s (optimized)\n",
			sizeMB, time.Since(start).Round(time.Millisecond))
		return nil
	}

	// Fallback: sequential read to fault pages into page cache.
	buf := make([]byte, 1024*1024) // 1MB chunks
	for {
		_, err := f.Read(buf)
		if err == io.EOF {
			break
		}
		if err != nil {
			return fmt.Errorf("prefault read: %w", err)
		}
	}

	fmt.Printf("[iTaK Torch] Pre-faulted %d MB in %s (sequential read)\n",
		sizeMB, time.Since(start).Round(time.Millisecond))
	return nil
}
