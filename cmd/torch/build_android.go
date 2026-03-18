//go:build android

// build_android.go contains Android-specific defaults for iTaK Torch.
//
// When compiled with GOOS=android, this file adjusts defaults:
//   - Reduces default context window (phones have less RAM)
//   - Enables memory-conservative settings
//   - Disables mmap by default (Android storage is different)
//   - Sets reasonable thread counts for mobile SoCs
//
// The GOTensor engine (pure Go) works on Android with zero changes.
// The llama.cpp FFI engine requires a pre-compiled ARM64 .so,
// which can be built separately using the Android NDK.
//
// Build: GOOS=android GOARCH=arm64 go build ./cmd/itaktorch/
// Or:    make android-arm64
package main

import "runtime"

func init() {
	// Android-specific defaults applied before flag parsing.
	// These can still be overridden by CLI flags.

	// Limit Go's thread pool on mobile to avoid battery drain.
	// Most phone SoCs have 4 performance + 4 efficiency cores.
	// Default to using performance cores only.
	if runtime.NumCPU() > 4 {
		runtime.GOMAXPROCS(4)
	}
}

// AndroidDefaults returns recommended settings for Android devices.
// Called by the serve command to adjust defaults when running on mobile.
func AndroidDefaults() map[string]interface{} {
	return map[string]interface{}{
		"ctx_size":   1024,  // Smaller context to fit in mobile RAM
		"batch_size": 512,   // Smaller batches for less memory pressure
		"threads":    4,     // Performance cores only
		"gpu_layers": 0,     // CPU-only by default (GPU support varies)
		"no_mmap":    true,  // Android storage doesn't mmap well
		"mlock":      false, // Can't mlock on Android without root
	}
}
