// gguf_writer.go exports trained models to GGUF format for portable deployment.
//
// WHAT: GGUF (GGML Universal Format) is the standard format for local LLM
// deployment, used by llama.cpp, Ollama, and our own GGUF reader. This writer
// enables exporting models trained in Torch directly to GGUF without
// any intermediate format conversion.
//
// WHY: This is the key benefit of a combined training + inference engine.
// Train a model -> export to GGUF -> serve instantly. No ONNX, no safetensors
// round-trip, no Python dependency for conversion.
//
// GGUF STRUCTURE:
//   Header:     magic, version, tensor count, metadata count
//   Metadata:   key-value pairs (model name, architecture, hyperparameters)
//   Tensor Info: name, dimensions, type, offset for each tensor
//   Tensor Data: raw tensor data (aligned to 32 bytes)
package native

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
)

// GGUFWriter creates GGUF v3 files.
// Reuses type constants from gguf.go (ggufTypeString, ggufTypeUint32, etc.).
type GGUFWriter struct {
	metadata   []ggufWriterKV
	tensorInfo []ggufWriterTensorInfo
	tensorData [][]byte

	// Alignment.
	alignment int
}

// ggufWriterKV holds one metadata key-value pair for writing.
type ggufWriterKV struct {
	key    string
	kvType uint32
	value  interface{}
}

// ggufWriterTensorInfo extends tensor info with offset tracking for writing.
type ggufWriterTensorInfo struct {
	name   string
	dims   []uint64
	ggType uint32
	offset uint64
}

// NewGGUFWriter creates a new GGUF writer.
func NewGGUFWriter() *GGUFWriter {
	return &GGUFWriter{
		alignment: 32,
	}
}

// AddString adds a string metadata key-value pair.
func (w *GGUFWriter) AddString(key, value string) {
	w.metadata = append(w.metadata, ggufWriterKV{key: key, kvType: ggufTypeString, value: value})
}

// AddUint32 adds a uint32 metadata value.
func (w *GGUFWriter) AddUint32(key string, value uint32) {
	w.metadata = append(w.metadata, ggufWriterKV{key: key, kvType: ggufTypeUint32, value: value})
}

// AddInt32 adds an int32 metadata value.
func (w *GGUFWriter) AddInt32(key string, value int32) {
	w.metadata = append(w.metadata, ggufWriterKV{key: key, kvType: ggufTypeInt32, value: value})
}

// AddFloat32 adds a float32 metadata value.
func (w *GGUFWriter) AddFloat32(key string, value float32) {
	w.metadata = append(w.metadata, ggufWriterKV{key: key, kvType: ggufTypeFloat32, value: value})
}

// AddFloat64 adds a float64 metadata value.
func (w *GGUFWriter) AddFloat64(key string, value float64) {
	w.metadata = append(w.metadata, ggufWriterKV{key: key, kvType: ggufTypeFloat64, value: value})
}

// AddBool adds a boolean metadata value.
func (w *GGUFWriter) AddBool(key string, value bool) {
	w.metadata = append(w.metadata, ggufWriterKV{key: key, kvType: ggufTypeBool, value: value})
}

// AddTensorF32 adds a float32 tensor to the file.
func (w *GGUFWriter) AddTensorF32(name string, data []float32, shape []int) {
	// Convert float32 to bytes.
	buf := make([]byte, len(data)*4)
	for i, v := range data {
		binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(v))
	}

	dims := make([]uint64, len(shape))
	for i, s := range shape {
		dims[i] = uint64(s)
	}

	w.tensorInfo = append(w.tensorInfo, ggufWriterTensorInfo{
		name:   name,
		dims:   dims,
		ggType: ggmlTypeF32,
	})
	w.tensorData = append(w.tensorData, buf)
}

// AddTensorQ8 adds a quantized INT8 tensor to the file.
func (w *GGUFWriter) AddTensorQ8(name string, data []int8, scales []float32, shape []int) {
	// Q8_0 format: blocks of 32 values, each block has 1 float16 scale + 32 int8 values.
	blockSize := 32
	numBlocks := (len(data) + blockSize - 1) / blockSize
	bytesPerBlock := 2 + blockSize // 2 bytes scale (f16) + 32 bytes data
	buf := make([]byte, numBlocks*bytesPerBlock)

	for b := 0; b < numBlocks; b++ {
		start := b * blockSize
		end := start + blockSize
		if end > len(data) {
			end = len(data)
		}

		// Scale (stored as float16).
		scaleIdx := b
		if scaleIdx >= len(scales) {
			scaleIdx = len(scales) - 1
		}
		f16Scale := f32ToF16(scales[scaleIdx])
		blockOff := b * bytesPerBlock
		binary.LittleEndian.PutUint16(buf[blockOff:], f16Scale)

		// Data.
		for i := start; i < end; i++ {
			buf[blockOff+2+(i-start)] = byte(data[i])
		}
	}

	dims := make([]uint64, len(shape))
	for i, s := range shape {
		dims[i] = uint64(s)
	}

	w.tensorInfo = append(w.tensorInfo, ggufWriterTensorInfo{
		name:   name,
		dims:   dims,
		ggType: ggmlTypeQ8_0,
	})
	w.tensorData = append(w.tensorData, buf)
}

// WriteModelConfig writes standard model architecture metadata.
func (w *GGUFWriter) WriteModelConfig(config ModelConfig) {
	prefix := string(config.Arch)
	w.AddString("general.architecture", string(config.Arch))
	w.AddString("general.name", config.Name)
	w.AddUint32(prefix+".block_count", uint32(config.NumLayers))
	w.AddUint32(prefix+".embedding_length", uint32(config.HiddenDim))
	w.AddUint32(prefix+".feed_forward_length", uint32(config.IntermediateDim))
	w.AddUint32(prefix+".attention.head_count", uint32(config.NumHeads))
	w.AddUint32(prefix+".attention.head_count_kv", uint32(config.NumKVHeads))
	w.AddUint32(prefix+".context_length", uint32(config.MaxSeqLen))
	w.AddFloat64(prefix+".rope.freq_base", config.RoPEBase)
	w.AddUint32(prefix+".vocab_size", uint32(config.VocabSize))
}

// WriteTo serializes the GGUF file to a writer.
func (w *GGUFWriter) WriteTo(out io.Writer) (int64, error) {
	var totalWritten int64

	// Calculate tensor data offsets.
	var dataOffset uint64
	for i := range w.tensorInfo {
		w.tensorInfo[i].offset = dataOffset
		dataSize := uint64(len(w.tensorData[i]))
		// Align to 32 bytes.
		padding := (w.alignment - int(dataSize%uint64(w.alignment))) % w.alignment
		dataOffset += dataSize + uint64(padding)
	}

	// Write header.
	n, err := w.writeHeader(out)
	if err != nil {
		return totalWritten, err
	}
	totalWritten += n

	// Write metadata.
	for _, kv := range w.metadata {
		n, err := w.writeKV(out, kv)
		if err != nil {
			return totalWritten, err
		}
		totalWritten += n
	}

	// Write tensor info.
	for _, ti := range w.tensorInfo {
		n, err := w.writeTensorInfo(out, ti)
		if err != nil {
			return totalWritten, err
		}
		totalWritten += n
	}

	// Align to 32 bytes before tensor data.
	padN, err := w.writePadding(out, totalWritten)
	if err != nil {
		return totalWritten, err
	}
	totalWritten += padN

	// Write tensor data.
	for _, data := range w.tensorData {
		nn, err := out.Write(data)
		if err != nil {
			return totalWritten, err
		}
		totalWritten += int64(nn)

		// Pad to alignment.
		padN, err := w.writePadding(out, totalWritten)
		if err != nil {
			return totalWritten, err
		}
		totalWritten += padN
	}

	return totalWritten, nil
}

func (w *GGUFWriter) writeHeader(out io.Writer) (int64, error) {
	var buf [24]byte
	// Magic: "GGUF"
	copy(buf[0:4], []byte{0x47, 0x47, 0x55, 0x46})
	// Version: 3
	binary.LittleEndian.PutUint32(buf[4:8], 3)
	// Tensor count.
	binary.LittleEndian.PutUint64(buf[8:16], uint64(len(w.tensorInfo)))
	// Metadata KV count.
	binary.LittleEndian.PutUint64(buf[16:24], uint64(len(w.metadata)))

	n, err := out.Write(buf[:])
	return int64(n), err
}

func (w *GGUFWriter) writeKV(out io.Writer, kv ggufWriterKV) (int64, error) {
	var total int64

	// Write key string.
	n, err := w.writeString(out, kv.key)
	if err != nil {
		return total, err
	}
	total += n

	// Write value type.
	var typeBuf [4]byte
	binary.LittleEndian.PutUint32(typeBuf[:], kv.kvType)
	nn, err := out.Write(typeBuf[:])
	total += int64(nn)
	if err != nil {
		return total, err
	}

	// Write value.
	switch kv.kvType {
	case ggufTypeString:
		n, err = w.writeString(out, kv.value.(string))
		total += n
	case ggufTypeUint32:
		var vBuf [4]byte
		binary.LittleEndian.PutUint32(vBuf[:], kv.value.(uint32))
		nn, err = out.Write(vBuf[:])
		total += int64(nn)
	case ggufTypeInt32:
		var vBuf [4]byte
		binary.LittleEndian.PutUint32(vBuf[:], uint32(kv.value.(int32)))
		nn, err = out.Write(vBuf[:])
		total += int64(nn)
	case ggufTypeFloat32:
		var vBuf [4]byte
		binary.LittleEndian.PutUint32(vBuf[:], math.Float32bits(kv.value.(float32)))
		nn, err = out.Write(vBuf[:])
		total += int64(nn)
	case ggufTypeFloat64:
		var vBuf [8]byte
		binary.LittleEndian.PutUint64(vBuf[:], math.Float64bits(kv.value.(float64)))
		nn, err = out.Write(vBuf[:])
		total += int64(nn)
	case ggufTypeBool:
		var vBuf [1]byte
		if kv.value.(bool) {
			vBuf[0] = 1
		}
		nn, err = out.Write(vBuf[:])
		total += int64(nn)
	}

	return total, err
}

func (w *GGUFWriter) writeTensorInfo(out io.Writer, ti ggufWriterTensorInfo) (int64, error) {
	var total int64

	// Name.
	n, err := w.writeString(out, ti.name)
	if err != nil {
		return total, err
	}
	total += n

	// Number of dimensions.
	var ndimBuf [4]byte
	binary.LittleEndian.PutUint32(ndimBuf[:], uint32(len(ti.dims)))
	nn, err := out.Write(ndimBuf[:])
	total += int64(nn)
	if err != nil {
		return total, err
	}

	// Dimensions.
	for _, d := range ti.dims {
		var dBuf [8]byte
		binary.LittleEndian.PutUint64(dBuf[:], d)
		nn, err = out.Write(dBuf[:])
		total += int64(nn)
		if err != nil {
			return total, err
		}
	}

	// Type.
	var tBuf [4]byte
	binary.LittleEndian.PutUint32(tBuf[:], ti.ggType)
	nn, err = out.Write(tBuf[:])
	total += int64(nn)
	if err != nil {
		return total, err
	}

	// Offset.
	var oBuf [8]byte
	binary.LittleEndian.PutUint64(oBuf[:], ti.offset)
	nn, err = out.Write(oBuf[:])
	total += int64(nn)

	return total, err
}

func (w *GGUFWriter) writeString(out io.Writer, s string) (int64, error) {
	var lenBuf [8]byte
	binary.LittleEndian.PutUint64(lenBuf[:], uint64(len(s)))
	n, err := out.Write(lenBuf[:])
	if err != nil {
		return int64(n), err
	}
	nn, err := out.Write([]byte(s))
	return int64(n + nn), err
}

func (w *GGUFWriter) writePadding(out io.Writer, currentPos int64) (int64, error) {
	padding := (int64(w.alignment) - currentPos%int64(w.alignment)) % int64(w.alignment)
	if padding == 0 {
		return 0, nil
	}
	pad := make([]byte, padding)
	n, err := out.Write(pad)
	return int64(n), err
}

// f32ToF16 converts float32 to IEEE 754 float16 (binary16).
func f32ToF16(f float32) uint16 {
	bits := math.Float32bits(f)
	sign := (bits >> 31) & 1
	exp := int((bits >> 23) & 0xFF) - 127
	frac := bits & 0x7FFFFF

	if exp > 15 {
		// Overflow: clamp to max f16.
		return uint16(sign<<15 | 0x7C00)
	}
	if exp < -14 {
		// Underflow: denormalized or zero.
		return uint16(sign << 15)
	}

	return uint16(sign<<15 | uint32(exp+15)<<10 | (frac >> 13))
}

// Stats returns info about the file being constructed.
func (w *GGUFWriter) Stats() map[string]interface{} {
	var totalDataBytes int64
	for _, d := range w.tensorData {
		totalDataBytes += int64(len(d))
	}
	return map[string]interface{}{
		"metadata_count": len(w.metadata),
		"tensor_count":   len(w.tensorInfo),
		"data_bytes":     fmt.Sprintf("%.1f MB", float64(totalDataBytes)/(1024*1024)),
	}
}
