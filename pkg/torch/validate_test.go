package torch

import (
	"os"
	"path/filepath"
	"testing"
)

// TestValidateModelPath_ValidPaths verifies that legitimate model paths pass validation.
func TestValidateModelPath_ValidPaths(t *testing.T) {
	// Create a temp directory with a valid .gguf file.
	dir := t.TempDir()
	validModel := filepath.Join(dir, "test-model.gguf")
	if err := os.WriteFile(validModel, []byte("fake-gguf"), 0644); err != nil {
		t.Fatal(err)
	}

	if err := ValidateModelPath(validModel); err != nil {
		t.Errorf("expected valid path to pass, got: %v", err)
	}
}

// TestValidateModelPath_Traversal verifies that directory traversal is blocked.
func TestValidateModelPath_Traversal(t *testing.T) {
	tests := []string{
		"../../../etc/passwd.gguf",
		"models/../../../secret.gguf",
		"..\\..\\windows\\system32\\config.gguf",
	}

	for _, path := range tests {
		err := ValidateModelPath(path)
		if err == nil {
			t.Errorf("expected traversal path %q to be rejected", path)
		}
	}
}

// TestValidateModelPath_WrongExtension verifies that non-.gguf files are rejected.
func TestValidateModelPath_WrongExtension(t *testing.T) {
	dir := t.TempDir()
	badFile := filepath.Join(dir, "model.bin")
	if err := os.WriteFile(badFile, []byte("data"), 0644); err != nil {
		t.Fatal(err)
	}

	err := ValidateModelPath(badFile)
	if err == nil {
		t.Error("expected non-.gguf path to be rejected")
	}
}

// TestValidateModelPath_EmptyPath verifies that empty paths are rejected.
func TestValidateModelPath_EmptyPath(t *testing.T) {
	err := ValidateModelPath("")
	if err == nil {
		t.Error("expected empty path to be rejected")
	}
}

// TestValidateModelPath_NonexistentFile verifies that missing files are caught.
func TestValidateModelPath_NonexistentFile(t *testing.T) {
	err := ValidateModelPath("/nonexistent/model.gguf")
	if err == nil {
		t.Error("expected nonexistent path to be rejected")
	}
}

// TestValidateMmprojPath_Valid verifies that .mmproj files pass validation.
func TestValidateMmprojPath_Valid(t *testing.T) {
	dir := t.TempDir()
	validFile := filepath.Join(dir, "clip-vit.mmproj")
	if err := os.WriteFile(validFile, []byte("fake-mmproj"), 0644); err != nil {
		t.Fatal(err)
	}

	if err := ValidateMmprojPath(validFile); err != nil {
		t.Errorf("expected valid mmproj path to pass, got: %v", err)
	}
}

// TestValidateMmprojPath_WrongExtension verifies that .gguf files don't pass as mmproj.
func TestValidateMmprojPath_WrongExtension(t *testing.T) {
	dir := t.TempDir()
	ggufFile := filepath.Join(dir, "model.gguf")
	if err := os.WriteFile(ggufFile, []byte("data"), 0644); err != nil {
		t.Fatal(err)
	}

	err := ValidateMmprojPath(ggufFile)
	if err == nil {
		t.Error("expected .gguf file to be rejected as mmproj")
	}
}

// TestContainsTraversal checks the internal helper for edge cases.
func TestContainsTraversal(t *testing.T) {
	tests := []struct {
		path   string
		expect bool
	}{
		{"model.gguf", false},
		{"models/qwen3.gguf", false},
		{"../model.gguf", true},
		{"models/../secret.gguf", true},
		{"foo..bar.gguf", false}, // ".." in filename is NOT traversal
		{"..", true},
	}

	for _, tt := range tests {
		got := containsTraversal(tt.path)
		if got != tt.expect {
			t.Errorf("containsTraversal(%q) = %v, want %v", tt.path, got, tt.expect)
		}
	}
}
