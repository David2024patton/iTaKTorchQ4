// safetensors.go implements a pure Go parser for HuggingFace SafeTensors files.
//
// SafeTensors is a simple, secure format for storing model weights:
//   [8 bytes: header_size (uint64 LE)]
//   [N bytes: JSON header with tensor metadata]
//   [rest: contiguous tensor data]
//
// The JSON header maps tensor names to {dtype, shape, data_offsets: [begin, end]}.
// Data offsets are relative to the start of the data block (after the header).
//
// Supported dtypes: F32, F16, BF16.
// Models can be sharded across multiple .safetensors files.
package native

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// SafeTensorsFile represents a parsed SafeTensors file (or set of sharded files).
type SafeTensorsFile struct {
	tensors  map[string]*safeTensorInfo
	metadata map[string]string
	files    []string // paths to all shard files (1 for single, N for sharded)
}

// safeTensorInfo holds metadata for a single tensor.
type safeTensorInfo struct {
	Dtype       string   `json:"dtype"`
	Shape       []int    `json:"shape"`
	DataOffsets [2]int64 `json:"data_offsets"` // [begin, end] relative to data block
	fileIdx     int      // which file this tensor lives in
	dataStart   int64    // absolute byte offset of the data block in this file
}

// LoadSafeTensors loads a SafeTensors file or set of sharded files.
// For sharded models, pass the path to any shard or the model directory.
// The loader auto-discovers sibling shards by filename pattern.
func LoadSafeTensors(path string) (*SafeTensorsFile, error) {
	// Discover all shard files.
	files, err := discoverShards(path)
	if err != nil {
		return nil, err
	}

	sf := &SafeTensorsFile{
		tensors:  make(map[string]*safeTensorInfo),
		metadata: make(map[string]string),
		files:    files,
	}

	for fileIdx, filePath := range files {
		if err := sf.parseFile(filePath, fileIdx); err != nil {
			return nil, fmt.Errorf("parse %s: %w", filepath.Base(filePath), err)
		}
	}

	return sf, nil
}

// parseFile reads one SafeTensors file and adds its tensors to the index.
func (sf *SafeTensorsFile) parseFile(path string, fileIdx int) error {
	f, err := os.Open(path)
	if err != nil {
		return fmt.Errorf("open: %w", err)
	}
	defer f.Close()

	// Read 8-byte header size.
	var headerSize uint64
	if err := binary.Read(f, binary.LittleEndian, &headerSize); err != nil {
		return fmt.Errorf("read header size: %w", err)
	}

	// Sanity: header must not exceed 100MB (SafeTensors spec limit).
	if headerSize > 100*1024*1024 {
		return fmt.Errorf("header too large: %d bytes (max 100MB)", headerSize)
	}

	// Read JSON header.
	headerBytes := make([]byte, headerSize)
	if _, err := io.ReadFull(f, headerBytes); err != nil {
		return fmt.Errorf("read header: %w", err)
	}

	// Data block starts right after header.
	dataStart := int64(8 + headerSize)

	// Parse JSON: map of tensor names to info, plus optional __metadata__.
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(headerBytes, &raw); err != nil {
		return fmt.Errorf("parse JSON header: %w", err)
	}

	for name, rawVal := range raw {
		if name == "__metadata__" {
			// Parse optional metadata.
			var meta map[string]string
			if json.Unmarshal(rawVal, &meta) == nil {
				for k, v := range meta {
					sf.metadata[k] = v
				}
			}
			continue
		}

		var info safeTensorInfo
		if err := json.Unmarshal(rawVal, &info); err != nil {
			return fmt.Errorf("parse tensor %q: %w", name, err)
		}
		info.fileIdx = fileIdx
		info.dataStart = dataStart
		sf.tensors[name] = &info
	}

	return nil
}

// TensorNames returns all tensor names across all shards.
func (sf *SafeTensorsFile) TensorNames() []string {
	names := make([]string, 0, len(sf.tensors))
	for name := range sf.tensors {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

// Metadata returns the __metadata__ key-value pairs.
func (sf *SafeTensorsFile) Metadata() map[string]interface{} {
	result := make(map[string]interface{}, len(sf.metadata))
	for k, v := range sf.metadata {
		result[k] = v
	}
	return result
}

// ReadTensor loads a tensor by name. F16 and BF16 are converted to float32.
func (sf *SafeTensorsFile) ReadTensor(name string) (*Tensor, error) {
	info, ok := sf.tensors[name]
	if !ok {
		return nil, fmt.Errorf("tensor %q not found", name)
	}

	if info.fileIdx >= len(sf.files) {
		return nil, fmt.Errorf("tensor %q references invalid shard index %d", name, info.fileIdx)
	}

	f, err := os.Open(sf.files[info.fileIdx])
	if err != nil {
		return nil, fmt.Errorf("open shard: %w", err)
	}
	defer f.Close()

	// Seek to tensor data.
	absOffset := info.dataStart + info.DataOffsets[0]
	if _, err := f.Seek(absOffset, io.SeekStart); err != nil {
		return nil, fmt.Errorf("seek tensor data: %w", err)
	}

	dataBytes := info.DataOffsets[1] - info.DataOffsets[0]

	// Calculate total elements.
	totalElements := 1
	for _, dim := range info.Shape {
		totalElements *= dim
	}

	data := make([]float32, totalElements)

	switch info.Dtype {
	case "F32":
		if err := binary.Read(f, binary.LittleEndian, data); err != nil {
			return nil, fmt.Errorf("read F32: %w", err)
		}

	case "F16":
		raw := make([]uint16, totalElements)
		if err := binary.Read(f, binary.LittleEndian, raw); err != nil {
			return nil, fmt.Errorf("read F16: %w", err)
		}
		for i, v := range raw {
			data[i] = float16ToFloat32(v)
		}

	case "BF16":
		raw := make([]uint16, totalElements)
		if err := binary.Read(f, binary.LittleEndian, raw); err != nil {
			return nil, fmt.Errorf("read BF16: %w", err)
		}
		for i, v := range raw {
			data[i] = bfloat16ToFloat32(v)
		}

	default:
		return nil, fmt.Errorf("unsupported dtype %q (supported: F32, F16, BF16), data size: %d bytes",
			info.Dtype, dataBytes)
	}

	shape := make([]int, len(info.Shape))
	copy(shape, info.Shape)

	return &Tensor{Data: data, Shape: shape}, nil
}

// FindTensor returns true if the named tensor exists.
func (sf *SafeTensorsFile) FindTensor(name string) bool {
	_, ok := sf.tensors[name]
	return ok
}

// --- Shard Discovery ---

// discoverShards finds all SafeTensors shard files for a given path.
// Handles patterns like:
//   - model.safetensors (single file)
//   - model-00001-of-00003.safetensors (sharded)
func discoverShards(path string) ([]string, error) {
	info, err := os.Stat(path)
	if err != nil {
		return nil, fmt.Errorf("stat %s: %w", path, err)
	}

	// If it's a directory, find all .safetensors files in it.
	if info.IsDir() {
		entries, err := os.ReadDir(path)
		if err != nil {
			return nil, fmt.Errorf("read dir: %w", err)
		}
		var files []string
		for _, e := range entries {
			if strings.HasSuffix(e.Name(), ".safetensors") {
				files = append(files, filepath.Join(path, e.Name()))
			}
		}
		if len(files) == 0 {
			return nil, fmt.Errorf("no .safetensors files in %s", path)
		}
		sort.Strings(files)
		return files, nil
	}

	// Single file: check for sharded siblings.
	dir := filepath.Dir(path)
	base := filepath.Base(path)

	// Check for sharded pattern: *-00001-of-*.safetensors
	if idx := strings.Index(base, "-00001-of-"); idx >= 0 {
		prefix := base[:idx]
		entries, err := os.ReadDir(dir)
		if err != nil {
			return nil, fmt.Errorf("read dir for shards: %w", err)
		}
		var files []string
		for _, e := range entries {
			if strings.HasPrefix(e.Name(), prefix) && strings.HasSuffix(e.Name(), ".safetensors") {
				files = append(files, filepath.Join(dir, e.Name()))
			}
		}
		sort.Strings(files)
		return files, nil
	}

	// Single, non-sharded file.
	return []string{path}, nil
}

// float16ToFloat32ST converts IEEE 754 half-precision float to float32.
// This is a local alias to avoid depending on the gguf.go version directly,
// but in practice both are in the same package.
var _ = math.Float32frombits // ensure math is used
