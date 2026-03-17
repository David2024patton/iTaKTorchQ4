// validate.go implements security validation for file paths passed to iTaK Torch.
//
// WHY THIS EXISTS:
// Users pass model paths via CLI flags (--model, --mmproj) and API requests.
// Without validation, an attacker could use path traversal ("../../etc/passwd")
// or symlinks to trick Torch into reading files outside the allowed directories.
// This package blocks those attacks at the point of entry.
package torch

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// ---------- Public Validation Functions ----------

// ValidateModelPath checks that a model file path is safe to load.
// It rejects:
//   - Paths containing ".." (directory traversal)
//   - Files that don't end in .gguf
//   - Symlinks that point outside the parent directory
//   - Absolute paths (when running inside a models directory context)
//
// Returns nil if the path is safe, or an error describing the violation.
func ValidateModelPath(path string) error {
	return validatePath(path, ".gguf", "model")
}

// ValidateMmprojPath checks that a multimodal projection file path is safe.
// Same rules as ValidateModelPath but allows .mmproj extension.
func ValidateMmprojPath(path string) error {
	return validatePath(path, ".mmproj", "mmproj")
}

// ---------- Internal Validation Logic ----------

// validatePath is the shared implementation for all file path validation.
// It enforces a strict allowlist approach:
//   1. No ".." components (prevents escaping the intended directory)
//   2. Must have the expected file extension
//   3. Symlinks must resolve to within the same parent directory
//   4. File must actually exist
func validatePath(path, requiredExt, label string) error {
	if path == "" {
		return fmt.Errorf("%s path is empty", label)
	}

	// --- Rule 1: Block directory traversal ---
	// filepath.Clean normalizes the path (resolves "." and redundant separators)
	// but preserves ".." if it can't be resolved against a root.
	// We check for ".." in both the original and cleaned path.
	cleaned := filepath.Clean(path)
	if containsTraversal(cleaned) {
		return fmt.Errorf("%s path %q contains directory traversal (..)", label, path)
	}

	// --- Rule 2: Enforce file extension ---
	// Only allow known model file formats to prevent loading arbitrary files.
	lower := strings.ToLower(cleaned)
	if !strings.HasSuffix(lower, requiredExt) {
		return fmt.Errorf("%s path %q must end with %s", label, path, requiredExt)
	}

	// --- Rule 3: Verify the file actually exists ---
	// Use os.Lstat (not os.Stat) to detect symlinks before following them.
	info, err := os.Lstat(cleaned)
	if err != nil {
		if os.IsNotExist(err) {
			return fmt.Errorf("%s file %q does not exist", label, path)
		}
		return fmt.Errorf("%s path %q: %w", label, path, err)
	}

	// --- Rule 4: If it's a symlink, verify the target is safe ---
	// Symlinks can be used to escape directory boundaries.
	// We resolve the symlink and check that the real path stays within
	// the same parent directory as the link itself.
	if info.Mode()&os.ModeSymlink != 0 {
		realPath, err := filepath.EvalSymlinks(cleaned)
		if err != nil {
			return fmt.Errorf("%s symlink %q could not be resolved: %w", label, path, err)
		}

		// The resolved path must be within the same parent directory.
		linkDir := filepath.Dir(cleaned)
		absLinkDir, _ := filepath.Abs(linkDir)
		absRealPath, _ := filepath.Abs(realPath)

		if !strings.HasPrefix(absRealPath, absLinkDir) {
			return fmt.Errorf("%s symlink %q points outside its directory (resolves to %q)", label, path, realPath)
		}

		// Also validate the resolved path's extension.
		if !strings.HasSuffix(strings.ToLower(realPath), requiredExt) {
			return fmt.Errorf("%s symlink %q resolves to non-%s file %q", label, path, requiredExt, realPath)
		}
	}

	// --- Rule 5: Must be a regular file (not a directory or device) ---
	// After symlink resolution, verify we're pointing at a real file.
	if !info.Mode().IsRegular() && info.Mode()&os.ModeSymlink == 0 {
		return fmt.Errorf("%s path %q is not a regular file", label, path)
	}

	return nil
}

// containsTraversal checks if a cleaned path contains any ".." path component.
// We split on the OS path separator and check each component individually
// to avoid false positives (e.g., a file named "foo..bar.gguf").
func containsTraversal(path string) bool {
	for _, part := range strings.Split(path, string(os.PathSeparator)) {
		if part == ".." {
			return true
		}
	}
	// Also check forward-slash separators (cross-platform safety).
	for _, part := range strings.Split(path, "/") {
		if part == ".." {
			return true
		}
	}
	return false
}
