//go:build !linux && !windows

package torch

import (
	"fmt"
	"os"
)

// prefaultPlatform is a no-op on unsupported platforms.
// The fallback sequential read in prefault.go handles these.
func prefaultPlatform(f *os.File, size int64) error {
	return fmt.Errorf("no platform-specific prefault for this OS")
}
