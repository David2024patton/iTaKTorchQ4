package torch

import (
	"testing"

	"github.com/google/go-cmp/cmp"
)

// TestParseOCIRef validates parsing OCI image references into components.
func TestParseOCIRef(t *testing.T) {
	tests := []struct {
		input    string
		wantReg  string
		wantRepo string
		wantTag  string
	}{
		{
			input:    "ghcr.io/org/model:v1",
			wantReg:  "ghcr.io",
			wantRepo: "org/model",
			wantTag:  "v1",
		},
		{
			input:    "registry.ollama.ai/library/qwen3:0.6b",
			wantReg:  "registry.ollama.ai",
			wantRepo: "library/qwen3",
			wantTag:  "0.6b",
		},
		{
			input:    "docker.io/user/model:q4_k_m",
			wantReg:  "index.docker.io", // docker.io resolves to index.docker.io
			wantRepo: "user/model",
			wantTag:  "q4_k_m",
		},
		{
			input:    "localhost:5000/my-models:custom",
			wantReg:  "localhost:5000",
			wantRepo: "my-models",
			wantTag:  "custom",
		},
	}

	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			gotReg, gotRepo, gotTag, err := ParseOCIRef(tt.input)
			if err != nil {
				t.Fatalf("ParseOCIRef(%q) error: %v", tt.input, err)
			}
			if diff := cmp.Diff(tt.wantReg, gotReg); diff != "" {
				t.Errorf("registry mismatch (-want +got):\n%s", diff)
			}
			if diff := cmp.Diff(tt.wantRepo, gotRepo); diff != "" {
				t.Errorf("repository mismatch (-want +got):\n%s", diff)
			}
			if diff := cmp.Diff(tt.wantTag, gotTag); diff != "" {
				t.Errorf("tag mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

// TestIsOCIRef validates OCI reference detection heuristics.
func TestIsOCIRef(t *testing.T) {
	tests := []struct {
		input string
		want  bool
	}{
		{"ghcr.io/org/model:v1", true},
		{"registry.ollama.ai/library/qwen3:0.6b", true},
		{"localhost:5000/models:custom", true},
		{"docker.io/user/model:latest", true},
		{"qwen3:0.6b", false},          // Ollama-style, no registry
		{"llama3", false},               // Plain model name
		{"Qwen/Qwen3-GGUF", false},     // HuggingFace-style, no dots in first segment
	}

	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			got := IsOCIRef(tt.input)
			if got != tt.want {
				t.Errorf("IsOCIRef(%q) = %v, want %v", tt.input, got, tt.want)
			}
		})
	}
}

// TestNewOCIPuller validates constructor logic.
func TestNewOCIPuller(t *testing.T) {
	tmpDir := t.TempDir()

	puller, err := NewOCIPuller(tmpDir, "", "")
	if err != nil {
		t.Fatal(err)
	}
	if puller.CacheDir != tmpDir {
		t.Errorf("CacheDir = %q, want %q", puller.CacheDir, tmpDir)
	}
	if puller.Auth == nil {
		t.Error("Auth should not be nil")
	}

	// With credentials.
	puller2, err := NewOCIPuller(tmpDir, "user", "pass")
	if err != nil {
		t.Fatal(err)
	}
	if puller2.Auth == nil {
		t.Error("Auth should not be nil with credentials")
	}
}

// TestModelMediaTypes verifies the priority list is sensible.
func TestModelMediaTypes(t *testing.T) {
	if len(modelMediaTypes) == 0 {
		t.Fatal("modelMediaTypes should not be empty")
	}

	// Ollama type should be first priority.
	want := "application/vnd.ollama.image.model"
	if diff := cmp.Diff(want, modelMediaTypes[0]); diff != "" {
		t.Errorf("first media type mismatch (-want +got):\n%s", diff)
	}
}

// TestFormatSize_WithCmp demonstrates go-cmp for the existing FormatSize helper.
func TestFormatSize_WithCmp(t *testing.T) {
	tests := []struct {
		input int64
		want  string
	}{
		{0, "0 B"},
		{512, "512 B"},
		{1024, "1.0 KB"},
		{1536, "1.5 KB"},
		{1048576, "1.0 MB"},
		{1073741824, "1.0 GB"},
		{4294967296, "4.0 GB"},
	}

	for _, tt := range tests {
		got := FormatSize(tt.input)
		if diff := cmp.Diff(tt.want, got); diff != "" {
			t.Errorf("FormatSize(%d) mismatch (-want +got):\n%s", tt.input, diff)
		}
	}
}
