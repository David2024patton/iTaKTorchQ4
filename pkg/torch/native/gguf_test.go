package native

import (
	"bytes"
	"encoding/binary"
	"math"
	"os"
	"testing"
)

// buildSyntheticGGUF creates a minimal GGUF v3 file in memory for testing.
// Contains 1 metadata entry ("general.architecture" = "test") and 1 tensor
// ("test.weight" with shape [2,3] of F32).
func buildSyntheticGGUF(t *testing.T) []byte {
	t.Helper()
	var buf bytes.Buffer

	// Magic: "GGUF"
	binary.Write(&buf, binary.LittleEndian, uint32(0x46554747))
	// Version: 3
	binary.Write(&buf, binary.LittleEndian, uint32(3))
	// Tensor count: 1
	binary.Write(&buf, binary.LittleEndian, uint64(1))
	// Metadata count: 1
	binary.Write(&buf, binary.LittleEndian, uint64(1))

	// Metadata: "general.architecture" = "test"
	writeGGUFString(&buf, "general.architecture")
	binary.Write(&buf, binary.LittleEndian, uint32(8)) // string type
	writeGGUFString(&buf, "test")

	// Tensor info: "test.weight" shape [2,3] F32
	writeGGUFString(&buf, "test.weight")
	binary.Write(&buf, binary.LittleEndian, uint32(2))  // ndims
	binary.Write(&buf, binary.LittleEndian, uint64(2))  // dim0
	binary.Write(&buf, binary.LittleEndian, uint64(3))  // dim1
	binary.Write(&buf, binary.LittleEndian, uint32(0))  // F32 type
	binary.Write(&buf, binary.LittleEndian, uint64(0))  // offset from data section

	// Pad to 32-byte alignment for data section.
	pos := buf.Len()
	alignment := 32
	padded := ((pos + alignment - 1) / alignment) * alignment
	for buf.Len() < padded {
		buf.WriteByte(0)
	}

	// Tensor data: 6 floats [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
	testData := []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}
	binary.Write(&buf, binary.LittleEndian, testData)

	return buf.Bytes()
}

func writeGGUFString(buf *bytes.Buffer, s string) {
	binary.Write(buf, binary.LittleEndian, uint64(len(s)))
	buf.WriteString(s)
}

// TestGGUF_ParseHeader verifies GGUF header parsing (magic, version, counts).
func TestGGUF_ParseHeader(t *testing.T) {
	data := buildSyntheticGGUF(t)

	tmpFile, err := os.CreateTemp("", "test-*.gguf")
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(tmpFile.Name())
	tmpFile.Write(data)
	tmpFile.Close()

	gf, err := LoadGGUF(tmpFile.Name())
	if err != nil {
		t.Fatalf("LoadGGUF failed: %v", err)
	}

	if gf.Version != 3 {
		t.Errorf("Version = %d, want 3", gf.Version)
	}
	if gf.TensorCount != 1 {
		t.Errorf("TensorCount = %d, want 1", gf.TensorCount)
	}
	if gf.MetaCount != 1 {
		t.Errorf("MetaCount = %d, want 1", gf.MetaCount)
	}
}

// TestGGUF_Metadata verifies metadata key-value parsing.
func TestGGUF_Metadata(t *testing.T) {
	data := buildSyntheticGGUF(t)

	tmpFile, err := os.CreateTemp("", "test-*.gguf")
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(tmpFile.Name())
	tmpFile.Write(data)
	tmpFile.Close()

	gf, err := LoadGGUF(tmpFile.Name())
	if err != nil {
		t.Fatal(err)
	}

	arch := gf.GetMetadataString("general.architecture")
	if arch != "test" {
		t.Errorf("architecture = %q, want %q", arch, "test")
	}

	// Non-existent key should return empty.
	missing := gf.GetMetadataString("nonexistent.key")
	if missing != "" {
		t.Errorf("missing key returned %q, want empty", missing)
	}
}

// TestGGUF_TensorInfo verifies tensor info extraction.
func TestGGUF_TensorInfo(t *testing.T) {
	data := buildSyntheticGGUF(t)

	tmpFile, err := os.CreateTemp("", "test-*.gguf")
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(tmpFile.Name())
	tmpFile.Write(data)
	tmpFile.Close()

	gf, err := LoadGGUF(tmpFile.Name())
	if err != nil {
		t.Fatal(err)
	}

	if len(gf.Tensors) != 1 {
		t.Fatalf("tensor count = %d, want 1", len(gf.Tensors))
	}

	ti := gf.Tensors[0]
	if ti.Name != "test.weight" {
		t.Errorf("name = %q, want %q", ti.Name, "test.weight")
	}
	if ti.NDims != 2 {
		t.Errorf("ndims = %d, want 2", ti.NDims)
	}
	if ti.Dimensions[0] != 2 || ti.Dimensions[1] != 3 {
		t.Errorf("dims = %v, want [2, 3]", ti.Dimensions)
	}
	if ti.Type != ggmlTypeF32 {
		t.Errorf("type = %d, want F32 (%d)", ti.Type, ggmlTypeF32)
	}
}

// TestGGUF_ReadTensor verifies loading F32 tensor data.
func TestGGUF_ReadTensor(t *testing.T) {
	data := buildSyntheticGGUF(t)

	tmpFile, err := os.CreateTemp("", "test-*.gguf")
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(tmpFile.Name())
	tmpFile.Write(data)
	tmpFile.Close()

	gf, err := LoadGGUF(tmpFile.Name())
	if err != nil {
		t.Fatal(err)
	}

	tensor, err := gf.ReadTensor(gf.Tensors[0])
	if err != nil {
		t.Fatalf("ReadTensor: %v", err)
	}

	expected := []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}
	if len(tensor.Data) != len(expected) {
		t.Fatalf("data length = %d, want %d", len(tensor.Data), len(expected))
	}
	for i, v := range expected {
		if tensor.Data[i] != v {
			t.Errorf("data[%d] = %f, want %f", i, tensor.Data[i], v)
		}
	}
}

// TestGGUF_FindTensor verifies tensor lookup by name.
func TestGGUF_FindTensor(t *testing.T) {
	data := buildSyntheticGGUF(t)

	tmpFile, err := os.CreateTemp("", "test-*.gguf")
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(tmpFile.Name())
	tmpFile.Write(data)
	tmpFile.Close()

	gf, err := LoadGGUF(tmpFile.Name())
	if err != nil {
		t.Fatal(err)
	}

	found := gf.FindTensor("test.weight")
	if found == nil {
		t.Fatal("FindTensor returned nil for existing tensor")
	}

	missing := gf.FindTensor("nonexistent")
	if missing != nil {
		t.Error("FindTensor should return nil for missing tensor")
	}
}

// TestGGUF_BadMagic verifies rejection of non-GGUF files.
func TestGGUF_BadMagic(t *testing.T) {
	tmpFile, err := os.CreateTemp("", "bad-*.gguf")
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(tmpFile.Name())
	tmpFile.Write([]byte("NOT A GGUF FILE"))
	tmpFile.Close()

	_, err = LoadGGUF(tmpFile.Name())
	if err == nil {
		t.Error("expected error for non-GGUF file")
	}
}

// TestFloat16Conversion verifies IEEE 754 half-precision conversion.
func TestFloat16Conversion(t *testing.T) {
	tests := []struct {
		f16  uint16
		want float32
	}{
		{0x3C00, 1.0},           // 1.0 in FP16
		{0xBC00, -1.0},          // -1.0 in FP16
		{0x0000, 0.0},           // +0
		{0x8000, math.Float32frombits(0x80000000)}, // -0 (sign bit set)
		{0x4000, 2.0},           // 2.0
		{0x3800, 0.5},           // 0.5
		{0x7C00, float32(math.Inf(1))},  // +Inf
		{0xFC00, float32(math.Inf(-1))}, // -Inf
	}

	for _, tt := range tests {
		got := float16ToFloat32(tt.f16)
		if got != tt.want {
			t.Errorf("float16ToFloat32(0x%04X) = %v, want %v", tt.f16, got, tt.want)
		}
	}
}
