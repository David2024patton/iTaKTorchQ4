// model_file.go provides a unified interface for loading model weights from
// different file formats (GGUF, SafeTensors). The GOTensor engine uses this
// abstraction so it doesn't need to know which format the weights came from.
package native

import (
	"fmt"
	"path/filepath"
	"strings"
)

// ModelFile is a format-agnostic interface for accessing model tensors.
type ModelFile interface {
	// Metadata returns model metadata (architecture, context length, etc.).
	Metadata() map[string]interface{}

	// TensorNames returns all tensor names in the file.
	TensorNames() []string

	// ReadTensor loads a tensor by name, dequantizing if needed.
	// All data is returned as float32 regardless of storage format.
	ReadTensor(name string) (*Tensor, error)
}

// ggufModelFile wraps GGUFFile to implement ModelFile.
type ggufModelFile struct {
	gf *GGUFFile
}

func (g *ggufModelFile) Metadata() map[string]interface{} {
	return g.gf.Metadata
}

func (g *ggufModelFile) TensorNames() []string {
	return g.gf.TensorNames()
}

func (g *ggufModelFile) ReadTensor(name string) (*Tensor, error) {
	info := g.gf.FindTensor(name)
	if info == nil {
		return nil, fmt.Errorf("tensor %q not found in GGUF", name)
	}
	return g.gf.ReadTensor(*info)
}

// safetensorsModelFile wraps SafeTensorsFile to implement ModelFile.
type safetensorsModelFile struct {
	sf *SafeTensorsFile
}

func (s *safetensorsModelFile) Metadata() map[string]interface{} {
	return s.sf.Metadata()
}

func (s *safetensorsModelFile) TensorNames() []string {
	return s.sf.TensorNames()
}

func (s *safetensorsModelFile) ReadTensor(name string) (*Tensor, error) {
	return s.sf.ReadTensor(name)
}

// LoadModelFile auto-detects the file format and returns a unified ModelFile.
// Supported formats:
//   - .gguf: GGUF v3 (llama.cpp format)
//   - .safetensors: HuggingFace SafeTensors
//
// For SafeTensors, pass any shard file or the model directory.
func LoadModelFile(path string) (ModelFile, error) {
	ext := strings.ToLower(filepath.Ext(path))

	switch ext {
	case ".gguf":
		gf, err := LoadGGUF(path)
		if err != nil {
			return nil, fmt.Errorf("load GGUF: %w", err)
		}
		return &ggufModelFile{gf: gf}, nil

	case ".safetensors":
		sf, err := LoadSafeTensors(path)
		if err != nil {
			return nil, fmt.Errorf("load SafeTensors: %w", err)
		}
		return &safetensorsModelFile{sf: sf}, nil

	case "":
		// No extension: might be a directory of SafeTensors shards.
		sf, err := LoadSafeTensors(path)
		if err != nil {
			return nil, fmt.Errorf("load model directory: %w", err)
		}
		return &safetensorsModelFile{sf: sf}, nil

	default:
		return nil, fmt.Errorf("unsupported model format %q (supported: .gguf, .safetensors)", ext)
	}
}
