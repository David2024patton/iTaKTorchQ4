// model_integrity.go provides SHA256 integrity verification for GGUF model files.
//
// WHY: Corrupted model files silently produce garbage output. Checking the hash
// on load catches corruption from interrupted downloads, disk errors, or
// incomplete copies before wasting GPU VRAM on a broken model.
package torch

import (
	"crypto/sha256"
	"fmt"
	"io"
	"os"
	"strings"
)

// VerifyModelIntegrity checks if a model file matches its .sha256 sidecar.
// Returns nil if verified, error on mismatch, or a warning message if no
// sidecar file exists (not treated as an error).
//
// Sidecar format: "<sha256hex>  <filename>\n" or just "<sha256hex>\n"
func VerifyModelIntegrity(modelPath string) error {
	sha256Path := modelPath + ".sha256"
	expected, err := readSHA256File(sha256Path)
	if err != nil {
		// No sidecar file is not a hard error, just log a warning.
		fmt.Printf("[iTaK Torch] No .sha256 sidecar found for model (skipping integrity check)\n")
		return nil
	}

	// Compute actual hash.
	fmt.Printf("[iTaK Torch] Verifying model integrity (SHA256)...\n")
	actual, err := computeSHA256(modelPath)
	if err != nil {
		return fmt.Errorf("integrity check failed: %w", err)
	}

	if actual != expected {
		return fmt.Errorf("model integrity check FAILED: expected %s, got %s", expected, actual)
	}

	fmt.Printf("[iTaK Torch] Model integrity verified: %s\n", actual[:16]+"...")
	return nil
}

// readSHA256File reads a .sha256 sidecar file and extracts the hash.
func readSHA256File(path string) (string, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return "", err
	}
	content := strings.TrimSpace(string(data))
	// Handle "hash  filename" format.
	if idx := strings.Index(content, " "); idx > 0 {
		content = content[:idx]
	}
	content = strings.ToLower(content)
	if len(content) != 64 {
		return "", fmt.Errorf("invalid SHA256 format in sidecar: %q", content)
	}
	return content, nil
}

// computeSHA256 computes the SHA256 hash of a file.
func computeSHA256(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()

	h := sha256.New()
	buf := make([]byte, 32*1024*1024) // 32MB buffer for fast hashing
	if _, err := io.CopyBuffer(h, f, buf); err != nil {
		return "", err
	}
	return fmt.Sprintf("%x", h.Sum(nil)), nil
}
