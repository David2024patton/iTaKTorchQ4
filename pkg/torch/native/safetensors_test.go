package native

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"
)

// ----- Dequantization Tests -----

func TestDequantQ4_0(t *testing.T) {
	// Build a single Q4_0 block: 18 bytes.
	// Scale = 1.0 (FP16 0x3C00), nibbles all 8 (producing 0.0).
	var buf bytes.Buffer
	binary.Write(&buf, binary.LittleEndian, uint16(0x3C00)) // scale = 1.0
	nib := make([]byte, 16)
	for i := range nib {
		// lo = 8 (0.0 after -8), hi = 9 (1.0 after -8)
		nib[i] = 0x98 // hi=9, lo=8
	}
	buf.Write(nib)

	out := make([]float32, 32)
	if err := dequantQ4_0(&buf, out, 32); err != nil {
		t.Fatal(err)
	}

	// First 16: (8-8)*1.0 = 0.0
	for i := 0; i < 16; i++ {
		if out[i] != 0.0 {
			t.Errorf("Q4_0[%d] = %f, want 0.0", i, out[i])
		}
	}
	// Last 16: (9-8)*1.0 = 1.0
	for i := 16; i < 32; i++ {
		if out[i] != 1.0 {
			t.Errorf("Q4_0[%d] = %f, want 1.0", i, out[i])
		}
	}
}

func TestDequantQ8_0(t *testing.T) {
	// Build a single Q8_0 block: 34 bytes.
	// Scale = 2.0 (FP16 0x4000), values = 1 to 32.
	var buf bytes.Buffer
	binary.Write(&buf, binary.LittleEndian, uint16(0x4000)) // scale = 2.0
	vals := make([]byte, 32)
	for i := range vals {
		vals[i] = byte(int8(i + 1))
	}
	buf.Write(vals)

	out := make([]float32, 32)
	if err := dequantQ8_0(&buf, out, 32); err != nil {
		t.Fatal(err)
	}

	for i := 0; i < 32; i++ {
		want := float32(i+1) * 2.0
		if out[i] != want {
			t.Errorf("Q8_0[%d] = %f, want %f", i, out[i], want)
		}
	}
}

func TestDequantQ4_1(t *testing.T) {
	// Q4_1 block: 20 bytes.
	// Scale = 1.0, min = 0.5, nibbles = 0 -> value = 0*1.0 + 0.5 = 0.5
	var buf bytes.Buffer
	binary.Write(&buf, binary.LittleEndian, uint16(0x3C00)) // scale = 1.0
	binary.Write(&buf, binary.LittleEndian, uint16(0x3800)) // min = 0.5
	nib := make([]byte, 16)
	// All zeros: lo = 0, hi = 0
	buf.Write(nib)

	out := make([]float32, 32)
	if err := dequantQ4_1(&buf, out, 32); err != nil {
		t.Fatal(err)
	}

	for i := 0; i < 32; i++ {
		if math.Abs(float64(out[i]-0.5)) > 0.01 {
			t.Errorf("Q4_1[%d] = %f, want 0.5", i, out[i])
		}
	}
}

func TestBFloat16ToFloat32(t *testing.T) {
	tests := []struct {
		bf16 uint16
		want float32
	}{
		{0x3F80, 1.0},  // 1.0 in BF16
		{0xBF80, -1.0}, // -1.0 in BF16
		{0x0000, 0.0},  // zero
		{0x4000, 2.0},  // 2.0 in BF16
		{0x4040, 3.0},  // 3.0 in BF16
	}

	for _, tt := range tests {
		got := bfloat16ToFloat32(tt.bf16)
		if got != tt.want {
			t.Errorf("bfloat16ToFloat32(0x%04X) = %v, want %v", tt.bf16, got, tt.want)
		}
	}
}

// ----- SafeTensors Tests -----

// buildSyntheticSafeTensors creates a minimal valid .safetensors file.
func buildSyntheticSafeTensors(t *testing.T, tensors map[string][]float32) []byte {
	t.Helper()
	var dataBuf bytes.Buffer

	header := make(map[string]interface{})

	for name, values := range tensors {
		begin := dataBuf.Len()
		binary.Write(&dataBuf, binary.LittleEndian, values)
		end := dataBuf.Len()

		header[name] = map[string]interface{}{
			"dtype":        "F32",
			"shape":        []int{len(values)},
			"data_offsets": []int{begin, end},
		}
	}

	headerJSON, err := json.Marshal(header)
	if err != nil {
		t.Fatal(err)
	}

	var out bytes.Buffer
	binary.Write(&out, binary.LittleEndian, uint64(len(headerJSON)))
	out.Write(headerJSON)
	out.Write(dataBuf.Bytes())

	return out.Bytes()
}

func TestSafeTensors_ParseAndRead(t *testing.T) {
	data := buildSyntheticSafeTensors(t, map[string][]float32{
		"weight": {1.0, 2.0, 3.0, 4.0},
		"bias":   {0.5, -0.5},
	})

	tmpFile, err := os.CreateTemp("", "test-*.safetensors")
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(tmpFile.Name())
	tmpFile.Write(data)
	tmpFile.Close()

	sf, err := LoadSafeTensors(tmpFile.Name())
	if err != nil {
		t.Fatalf("LoadSafeTensors: %v", err)
	}

	names := sf.TensorNames()
	if len(names) != 2 {
		t.Fatalf("tensor count = %d, want 2", len(names))
	}

	// Read weight tensor.
	weight, err := sf.ReadTensor("weight")
	if err != nil {
		t.Fatalf("ReadTensor(weight): %v", err)
	}
	expected := []float32{1.0, 2.0, 3.0, 4.0}
	if len(weight.Data) != len(expected) {
		t.Fatalf("weight length = %d, want %d", len(weight.Data), len(expected))
	}
	for i, v := range expected {
		if weight.Data[i] != v {
			t.Errorf("weight[%d] = %f, want %f", i, weight.Data[i], v)
		}
	}

	// Read bias tensor.
	bias, err := sf.ReadTensor("bias")
	if err != nil {
		t.Fatalf("ReadTensor(bias): %v", err)
	}
	if len(bias.Data) != 2 {
		t.Fatalf("bias length = %d, want 2", len(bias.Data))
	}
	if bias.Data[0] != 0.5 || bias.Data[1] != -0.5 {
		t.Errorf("bias = %v, want [0.5, -0.5]", bias.Data)
	}
}

func TestSafeTensors_MissingTensor(t *testing.T) {
	data := buildSyntheticSafeTensors(t, map[string][]float32{
		"weight": {1.0},
	})

	tmpFile, err := os.CreateTemp("", "test-*.safetensors")
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(tmpFile.Name())
	tmpFile.Write(data)
	tmpFile.Close()

	sf, err := LoadSafeTensors(tmpFile.Name())
	if err != nil {
		t.Fatal(err)
	}

	_, err = sf.ReadTensor("nonexistent")
	if err == nil {
		t.Error("expected error for missing tensor")
	}
}

// ----- ModelFile Interface Tests -----

func TestModelFile_LoadGGUF(t *testing.T) {
	data := buildSyntheticGGUF(t)

	tmpFile, err := os.CreateTemp("", "test-*.gguf")
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(tmpFile.Name())
	tmpFile.Write(data)
	tmpFile.Close()

	mf, err := LoadModelFile(tmpFile.Name())
	if err != nil {
		t.Fatalf("LoadModelFile GGUF: %v", err)
	}

	names := mf.TensorNames()
	if len(names) != 1 {
		t.Fatalf("tensor count = %d, want 1", len(names))
	}

	tensor, err := mf.ReadTensor("test.weight")
	if err != nil {
		t.Fatalf("ReadTensor: %v", err)
	}
	if len(tensor.Data) != 6 {
		t.Errorf("data length = %d, want 6", len(tensor.Data))
	}
}

func TestModelFile_LoadSafeTensors(t *testing.T) {
	data := buildSyntheticSafeTensors(t, map[string][]float32{
		"layer.weight": {1.0, 2.0, 3.0},
	})

	tmpFile, err := os.CreateTemp("", "test-*.safetensors")
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(tmpFile.Name())
	tmpFile.Write(data)
	tmpFile.Close()

	mf, err := LoadModelFile(tmpFile.Name())
	if err != nil {
		t.Fatalf("LoadModelFile SafeTensors: %v", err)
	}

	tensor, err := mf.ReadTensor("layer.weight")
	if err != nil {
		t.Fatalf("ReadTensor: %v", err)
	}
	expected := []float32{1.0, 2.0, 3.0}
	for i, v := range expected {
		if tensor.Data[i] != v {
			t.Errorf("data[%d] = %f, want %f", i, tensor.Data[i], v)
		}
	}
}

func TestModelFile_UnsupportedFormat(t *testing.T) {
	tmpFile, err := os.CreateTemp("", "test-*.bin")
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(tmpFile.Name())
	tmpFile.Write([]byte("not a model"))
	tmpFile.Close()

	_, err = LoadModelFile(tmpFile.Name())
	if err == nil {
		t.Error("expected error for unsupported format")
	}
}

// ----- Shard Discovery Tests -----

func TestShardDiscovery(t *testing.T) {
	dir, err := os.MkdirTemp("", "shards-*")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(dir)

	// Create synthetic shards.
	for _, name := range []string{
		"model-00001-of-00002.safetensors",
		"model-00002-of-00002.safetensors",
	} {
		data := buildSyntheticSafeTensors(t, map[string][]float32{
			name + ".weight": {float32(len(name))},
		})
		os.WriteFile(filepath.Join(dir, name), data, 0644)
	}

	// Discover from first shard.
	files, err := discoverShards(filepath.Join(dir, "model-00001-of-00002.safetensors"))
	if err != nil {
		t.Fatal(err)
	}
	if len(files) != 2 {
		t.Errorf("discovered %d shards, want 2", len(files))
	}

	// Discover from directory.
	files2, err := discoverShards(dir)
	if err != nil {
		t.Fatal(err)
	}
	if len(files2) != 2 {
		t.Errorf("discovered %d shards from dir, want 2", len(files2))
	}
}
