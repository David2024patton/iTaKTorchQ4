// gguf.go implements a minimal GGUF v3 file parser for loading model weights
// into the GOTensor pure Go engine.
//
// GGUF (GGML Universal Format) is the standard file format for llama.cpp models.
// This parser reads the binary format and extracts:
//   - Header: magic, version, tensor count, metadata count
//   - Metadata: key-value pairs (model architecture, context length, etc.)
//   - Tensor info: name, dimensions, data type, offset
//   - Tensor data: raw float32/float16 weight arrays
//
// LIMITATIONS:
//   - Only supports GGUF v3 (the current standard)
//   - Only loads F32 and F16 tensor types (no quantized formats like Q4_0)
//   - Designed for tiny models (<1B params) that fit in RAM
//
// For quantized or large models, use TorchEngine (llama.cpp FFI) instead.
package native

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
)

// GGUF magic number: bytes 'G','G','U','F' read as little-endian uint32.
const ggufMagic uint32 = 0x46554747

// GGUF metadata value types.
const (
	ggufTypeUint8   uint32 = 0
	ggufTypeInt8    uint32 = 1
	ggufTypeUint16  uint32 = 2
	ggufTypeInt16   uint32 = 3
	ggufTypeUint32  uint32 = 4
	ggufTypeInt32   uint32 = 5
	ggufTypeFloat32 uint32 = 6
	ggufTypeBool    uint32 = 7
	ggufTypeString  uint32 = 8
	ggufTypeArray   uint32 = 9
	ggufTypeUint64  uint32 = 10
	ggufTypeInt64   uint32 = 11
	ggufTypeFloat64 uint32 = 12
)

// GGUF tensor data types.
const (
	ggmlTypeF32  uint32 = 0
	ggmlTypeF16  uint32 = 1
	ggmlTypeQ4_0 uint32 = 2
	ggmlTypeQ4_1 uint32 = 3
	ggmlTypeQ5_0 uint32 = 6
	ggmlTypeQ5_1 uint32 = 7
	ggmlTypeQ8_0 uint32 = 8
	ggmlTypeQ8_1 uint32 = 9
	ggmlTypeQ2_K uint32 = 10
	ggmlTypeQ3_K uint32 = 11
	ggmlTypeQ4_K uint32 = 12
	ggmlTypeQ5_K uint32 = 13
	ggmlTypeQ6_K uint32 = 14
	ggmlTypeI2_S uint32 = 30 // BitNet 1.58-bit Ternary
	ggmlTypeBF16 uint32 = 31 // Shifted to 31 to avoid collision with BitNet
)

// ggmlBlockSize returns the number of elements per quantization block.
func ggmlBlockSize(dtype uint32) int {
	switch dtype {
	case ggmlTypeQ4_0, ggmlTypeQ4_1, ggmlTypeQ5_0, ggmlTypeQ5_1:
		return 32
	case ggmlTypeQ8_0, ggmlTypeQ8_1:
		return 32
	case ggmlTypeQ2_K, ggmlTypeQ3_K, ggmlTypeQ4_K, ggmlTypeQ5_K, ggmlTypeQ6_K:
		return 256
	default:
		return 1
	}
}

// ggmlBytesPerBlock returns bytes per quantization block.
func ggmlBytesPerBlock(dtype uint32) int {
	switch dtype {
	case ggmlTypeF32:
		return 4
	case ggmlTypeF16, ggmlTypeBF16:
		return 2
	case ggmlTypeQ4_0:
		return 18 // 2 (FP16 scale) + 16 (nibbles)
	case ggmlTypeQ4_1:
		return 20 // 2 (scale) + 2 (min) + 16 (nibbles)
	case ggmlTypeQ5_0:
		return 22 // 2 (scale) + 4 (high-bit mask) + 16 (nibbles)
	case ggmlTypeQ5_1:
		return 24 // 2 (scale) + 2 (min) + 4 (mask) + 16 (nibbles)
	case ggmlTypeQ8_0:
		return 34 // 2 (scale) + 32 (int8 values)
	case ggmlTypeQ8_1:
		return 36 // 2 (scale) + 2 (sum) + 32 (int8 values)
	case ggmlTypeQ2_K:
		return 84
	case ggmlTypeQ3_K:
		return 110
	case ggmlTypeQ4_K:
		return 144
	case ggmlTypeQ5_K:
		return 176
	case ggmlTypeQ6_K:
		return 210
	case ggmlTypeI2_S:
		return 32 // 128 elements in 32 bytes (2 bits/elem)
	default:
		return 0
	}
}

// GGUFFile represents a parsed GGUF model file.
type GGUFFile struct {
	Version     uint32
	TensorCount uint64
	MetaCount   uint64
	Metadata    map[string]interface{}
	Tensors     []GGUFTensorInfo
	DataOffset  int64 // byte offset where tensor data starts
	filePath    string
}

// GGUFTensorInfo describes a single tensor in the file.
type GGUFTensorInfo struct {
	Name       string
	NDims      uint32
	Dimensions []uint64
	Type       uint32 // ggmlTypeF32, ggmlTypeF16, etc.
	Offset     uint64 // offset from start of data section
}

// LoadGGUF parses a GGUF v3 file and returns its header, metadata, and tensor info.
// Does NOT load tensor data into memory until ReadTensor is called.
func LoadGGUF(path string) (*GGUFFile, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open GGUF: %w", err)
	}
	defer f.Close()

	gf := &GGUFFile{
		Metadata: make(map[string]interface{}),
		filePath: path,
	}

	// Read magic.
	var magic uint32
	if err := binary.Read(f, binary.LittleEndian, &magic); err != nil {
		return nil, fmt.Errorf("read magic: %w", err)
	}
	if magic != ggufMagic {
		return nil, fmt.Errorf("not a GGUF file (magic: 0x%08X, want 0x%08X)", magic, ggufMagic)
	}

	// Read version.
	if err := binary.Read(f, binary.LittleEndian, &gf.Version); err != nil {
		return nil, fmt.Errorf("read version: %w", err)
	}
	if gf.Version != 3 {
		return nil, fmt.Errorf("unsupported GGUF version %d (only v3 supported)", gf.Version)
	}

	// Read tensor count and metadata count.
	if err := binary.Read(f, binary.LittleEndian, &gf.TensorCount); err != nil {
		return nil, fmt.Errorf("read tensor count: %w", err)
	}
	if err := binary.Read(f, binary.LittleEndian, &gf.MetaCount); err != nil {
		return nil, fmt.Errorf("read metadata count: %w", err)
	}

	// Read metadata key-value pairs.
	for i := uint64(0); i < gf.MetaCount; i++ {
		key, err := readGGUFString(f)
		if err != nil {
			return nil, fmt.Errorf("read metadata key %d: %w", i, err)
		}

		val, err := readGGUFValue(f)
		if err != nil {
			return nil, fmt.Errorf("read metadata value for %q: %w", key, err)
		}

		gf.Metadata[key] = val
	}

	// Read tensor info entries.
	gf.Tensors = make([]GGUFTensorInfo, gf.TensorCount)
	for i := uint64(0); i < gf.TensorCount; i++ {
		name, err := readGGUFString(f)
		if err != nil {
			return nil, fmt.Errorf("read tensor %d name: %w", i, err)
		}

		var nDims uint32
		if err := binary.Read(f, binary.LittleEndian, &nDims); err != nil {
			return nil, fmt.Errorf("read tensor %d ndims: %w", i, err)
		}

		dims := make([]uint64, nDims)
		for d := uint32(0); d < nDims; d++ {
			if err := binary.Read(f, binary.LittleEndian, &dims[d]); err != nil {
				return nil, fmt.Errorf("read tensor %d dim %d: %w", i, d, err)
			}
		}

		var dtype uint32
		if err := binary.Read(f, binary.LittleEndian, &dtype); err != nil {
			return nil, fmt.Errorf("read tensor %d type: %w", i, err)
		}

		var offset uint64
		if err := binary.Read(f, binary.LittleEndian, &offset); err != nil {
			return nil, fmt.Errorf("read tensor %d offset: %w", i, err)
		}

		gf.Tensors[i] = GGUFTensorInfo{
			Name:       name,
			NDims:      nDims,
			Dimensions: dims,
			Type:       dtype,
			Offset:     offset,
		}
	}

	// Record where tensor data starts (aligned to next 32-byte boundary).
	pos, err := f.Seek(0, io.SeekCurrent)
	if err != nil {
		return nil, fmt.Errorf("seek current pos: %w", err)
	}
	alignment := int64(32)
	gf.DataOffset = ((pos + alignment - 1) / alignment) * alignment

	return gf, nil
}

// ReadTensor loads a specific tensor's data from the GGUF file.
// Supports F32, F16, BF16, and all standard quantized formats (Q4_0, Q4_1,
// Q5_0, Q5_1, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K).
// Quantized data is dequantized to float32 at load time.
func (gf *GGUFFile) ReadTensor(info GGUFTensorInfo) (*Tensor, error) {
	f, err := os.Open(gf.filePath)
	if err != nil {
		return nil, fmt.Errorf("open for tensor read: %w", err)
	}
	defer f.Close()

	// Calculate total element count.
	totalElements := uint64(1)
	for _, d := range info.Dimensions {
		totalElements *= d
	}

	// Seek to tensor data.
	dataPos := gf.DataOffset + int64(info.Offset)
	if _, err := f.Seek(dataPos, io.SeekStart); err != nil {
		return nil, fmt.Errorf("seek tensor data: %w", err)
	}

	data := make([]float32, totalElements)

	switch info.Type {
	case ggmlTypeF32:
		if err := binary.Read(f, binary.LittleEndian, data); err != nil {
			return nil, fmt.Errorf("read F32 tensor: %w", err)
		}

	case ggmlTypeF16:
		f16data := make([]uint16, totalElements)
		if err := binary.Read(f, binary.LittleEndian, f16data); err != nil {
			return nil, fmt.Errorf("read F16 tensor: %w", err)
		}
		for i, v := range f16data {
			data[i] = float16ToFloat32(v)
		}

	case ggmlTypeBF16:
		bf16data := make([]uint16, totalElements)
		if err := binary.Read(f, binary.LittleEndian, bf16data); err != nil {
			return nil, fmt.Errorf("read BF16 tensor: %w", err)
		}
		for i, v := range bf16data {
			data[i] = bfloat16ToFloat32(v)
		}

	case ggmlTypeQ4_0:
		if err := dequantQ4_0(f, data, totalElements); err != nil {
			return nil, fmt.Errorf("dequant Q4_0: %w", err)
		}

	case ggmlTypeQ4_1:
		if err := dequantQ4_1(f, data, totalElements); err != nil {
			return nil, fmt.Errorf("dequant Q4_1: %w", err)
		}

	case ggmlTypeQ5_0:
		if err := dequantQ5_0(f, data, totalElements); err != nil {
			return nil, fmt.Errorf("dequant Q5_0: %w", err)
		}

	case ggmlTypeQ5_1:
		if err := dequantQ5_1(f, data, totalElements); err != nil {
			return nil, fmt.Errorf("dequant Q5_1: %w", err)
		}

	case ggmlTypeQ8_0:
		if err := dequantQ8_0(f, data, totalElements); err != nil {
			return nil, fmt.Errorf("dequant Q8_0: %w", err)
		}

	case ggmlTypeQ2_K:
		if err := dequantQ2_K(f, data, totalElements); err != nil {
			return nil, fmt.Errorf("dequant Q2_K: %w", err)
		}

	case ggmlTypeQ4_K:
		// --- Q4_K INTERCEPTION ---
		// We completely bypass dequantQ4_K here.
		// Instead, we read the raw compressed blocks directly into RAM.
		bytesPerBlock := 144
		numBlocks := totalElements / 256
		if totalElements%256 != 0 {
			return nil, fmt.Errorf("Q4_K total elements %d not divisible by 256", totalElements)
		}
		dataQ4 := make([]byte, numBlocks*uint64(bytesPerBlock))
		if err := binary.Read(f, binary.LittleEndian, dataQ4); err != nil {
			return nil, fmt.Errorf("read Q4_K raw bytes: %w", err)
		}

		fmt.Printf("[GOTensor-DEBUG] ⚡ Intercepted Q4_K Tensor '%s': Bypassing decompression, buffering %d bytes natively.\n", info.Name, len(dataQ4))
		
		// Build and return the Tensor immediately, skipping everything else!
		shape := make([]int, len(info.Dimensions))
		for i, d := range info.Dimensions {
			shape[i] = int(d)
		}
		return &Tensor{Type: ggmlTypeQ4_K, DataQ4: dataQ4, Shape: shape, Strides: calculateStrides(shape)}, nil

	case ggmlTypeQ6_K:
		if err := dequantQ6_K(f, data, totalElements); err != nil {
			return nil, fmt.Errorf("dequant Q6_K: %w", err)
		}

	case ggmlTypeI2_S:
		// --- BitNet I2_S INTERCEPTION ---
		// We read the raw ternary blocks directly for our SIMD kernels.
		bytesPerBlock := 32
		numBlocks := totalElements / 128
		if totalElements%128 != 0 {
			return nil, fmt.Errorf("I2_S total elements %d not divisible by 128", totalElements)
		}
		dataI2S := make([]byte, numBlocks*uint64(bytesPerBlock))
		if _, err := io.ReadFull(f, dataI2S); err != nil {
			return nil, fmt.Errorf("read I2_S raw bytes: %w", err)
		}

		fmt.Printf("[GOTensor] ⚡ Loaded BitNet Tensor '%s' (ternary i2_s)\n", info.Name)
		
		shape := make([]int, len(info.Dimensions))
		for i, d := range info.Dimensions {
			shape[i] = int(d)
		}
		return &Tensor{Type: ggmlTypeI2_S, DataQ4: dataI2S, Shape: shape, Strides: calculateStrides(shape)}, nil

	default:
		return nil, fmt.Errorf("unsupported tensor type %d", info.Type)
	}

	// Build tensor shape.
	shape := make([]int, len(info.Dimensions))
	for i, d := range info.Dimensions {
		shape[i] = int(d)
	}

	return &Tensor{Type: ggmlTypeF32, Data: data, Shape: shape, Strides: calculateStrides(shape)}, nil
}

// calculateStrides computes row-major strides from a shape array.
func calculateStrides(shape []int) []int {
	strides := make([]int, len(shape))
	if len(shape) == 0 {
		return strides
	}
	strides[len(shape)-1] = 1
	for i := len(shape) - 2; i >= 0; i-- {
		strides[i] = strides[i+1] * shape[i+1]
	}
	return strides
}

// GetMetadataString returns a string metadata value, or empty string if not found.
func (gf *GGUFFile) GetMetadataString(key string) string {
	if v, ok := gf.Metadata[key].(string); ok {
		return v
	}
	return ""
}

// GetMetadataUint32 returns a uint32 metadata value, or 0 if not found.
func (gf *GGUFFile) GetMetadataUint32(key string) uint32 {
	if v, ok := gf.Metadata[key].(uint32); ok {
		return v
	}
	return 0
}

// FindTensor returns the tensor info for a given name, or nil if not found.
func (gf *GGUFFile) FindTensor(name string) *GGUFTensorInfo {
	for i := range gf.Tensors {
		if gf.Tensors[i].Name == name {
			return &gf.Tensors[i]
		}
	}
	return nil
}

// TensorNames returns all tensor names in the file.
func (gf *GGUFFile) TensorNames() []string {
	names := make([]string, len(gf.Tensors))
	for i, t := range gf.Tensors {
		names[i] = t.Name
	}
	return names
}

// --- Internal helpers ---

// readGGUFString reads a GGUF string (uint64 length + bytes).
func readGGUFString(r io.Reader) (string, error) {
	var length uint64
	if err := binary.Read(r, binary.LittleEndian, &length); err != nil {
		return "", err
	}
	if length > 1<<20 { // 1MB sanity limit
		return "", fmt.Errorf("string too long: %d bytes", length)
	}
	buf := make([]byte, length)
	if _, err := io.ReadFull(r, buf); err != nil {
		return "", err
	}
	return string(buf), nil
}

// readGGUFValue reads a typed GGUF metadata value.
func readGGUFValue(r io.Reader) (interface{}, error) {
	var valType uint32
	if err := binary.Read(r, binary.LittleEndian, &valType); err != nil {
		return nil, err
	}

	switch valType {
	case ggufTypeUint8:
		var v uint8
		err := binary.Read(r, binary.LittleEndian, &v)
		return v, err
	case ggufTypeInt8:
		var v int8
		err := binary.Read(r, binary.LittleEndian, &v)
		return v, err
	case ggufTypeUint16:
		var v uint16
		err := binary.Read(r, binary.LittleEndian, &v)
		return v, err
	case ggufTypeInt16:
		var v int16
		err := binary.Read(r, binary.LittleEndian, &v)
		return v, err
	case ggufTypeUint32:
		var v uint32
		err := binary.Read(r, binary.LittleEndian, &v)
		return v, err
	case ggufTypeInt32:
		var v int32
		err := binary.Read(r, binary.LittleEndian, &v)
		return v, err
	case ggufTypeFloat32:
		var v float32
		err := binary.Read(r, binary.LittleEndian, &v)
		return v, err
	case ggufTypeBool:
		var v uint8
		if err := binary.Read(r, binary.LittleEndian, &v); err != nil {
			return nil, err
		}
		return v != 0, nil
	case ggufTypeString:
		return readGGUFString(r)
	case ggufTypeArray:
		var elemType uint32
		if err := binary.Read(r, binary.LittleEndian, &elemType); err != nil {
			return nil, err
		}
		var count uint64
		if err := binary.Read(r, binary.LittleEndian, &count); err != nil {
			return nil, err
		}
		if count > 1<<20 { // sanity limit
			return nil, fmt.Errorf("array too large: %d elements", count)
		}
		arr := make([]interface{}, count)
		for i := uint64(0); i < count; i++ {
			// For arrays, read values of the element type directly.
			val, err := readGGUFValueOfType(r, elemType)
			if err != nil {
				return nil, fmt.Errorf("array element %d: %w", i, err)
			}
			arr[i] = val
		}
		return arr, nil
	case ggufTypeUint64:
		var v uint64
		err := binary.Read(r, binary.LittleEndian, &v)
		return v, err
	case ggufTypeInt64:
		var v int64
		err := binary.Read(r, binary.LittleEndian, &v)
		return v, err
	case ggufTypeFloat64:
		var v float64
		err := binary.Read(r, binary.LittleEndian, &v)
		return v, err
	default:
		return nil, fmt.Errorf("unknown metadata type: %d", valType)
	}
}

// readGGUFValueOfType reads a value of a known type (used for array elements).
func readGGUFValueOfType(r io.Reader, valType uint32) (interface{}, error) {
	switch valType {
	case ggufTypeUint32:
		var v uint32
		err := binary.Read(r, binary.LittleEndian, &v)
		return v, err
	case ggufTypeFloat32:
		var v float32
		err := binary.Read(r, binary.LittleEndian, &v)
		return v, err
	case ggufTypeString:
		return readGGUFString(r)
	case ggufTypeInt32:
		var v int32
		err := binary.Read(r, binary.LittleEndian, &v)
		return v, err
	case ggufTypeUint64:
		var v uint64
		err := binary.Read(r, binary.LittleEndian, &v)
		return v, err
	default:
		return nil, fmt.Errorf("unsupported array element type: %d", valType)
	}
}

// float16ToFloat32 converts an IEEE 754 half-precision float to float32.
func float16ToFloat32(h uint16) float32 {
	sign := uint32(h>>15) & 1
	exp := uint32(h>>10) & 0x1F
	mant := uint32(h) & 0x3FF

	if exp == 0 {
		if mant == 0 {
			// Zero.
			return math.Float32frombits(sign << 31)
		}
		// Denormalized: normalize it.
		for mant&0x400 == 0 {
			mant <<= 1
			exp--
		}
		exp++
		mant &= 0x3FF
	} else if exp == 0x1F {
		// Inf/NaN.
		return math.Float32frombits((sign << 31) | 0x7F800000 | (mant << 13))
	}

	exp = exp + (127 - 15) // rebias exponent
	f32bits := (sign << 31) | (exp << 23) | (mant << 13)
	return math.Float32frombits(f32bits)
}
