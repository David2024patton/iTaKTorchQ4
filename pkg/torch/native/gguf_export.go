// gguf_export.go implements GGUF model file export for sharing fine-tuned models.
//
// WHAT: Writes model weights (including merged LoRA and trained AttnRes queries)
// back to the GGUF format. This allows models fine-tuned with Torch to be
// loaded by llama.cpp, Ollama, or any other GGUF-compatible runtime.
//
// FORMAT: GGUF v3 with F32 tensor data and architecture metadata.
package native

import (
	"encoding/binary"
	"fmt"
	"os"
)

// GGUFExporter writes model weights to GGUF format.
type GGUFExporter struct {
	engine *NativeEngine
}

// NewGGUFExporter creates an exporter for the given engine.
func NewGGUFExporter(engine *NativeEngine) *GGUFExporter {
	return &GGUFExporter{engine: engine}
}

// Export writes the model to a GGUF file.
func (ex *GGUFExporter) Export(path string) error {
	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("create %s: %w", path, err)
	}
	defer f.Close()

	e := ex.engine

	// Count tensors.
	numTensors := 2 // embeddings + lmHead
	numTensors += len(e.layers) * 9 // WQ,WK,WV,WO,WGate,WUp,WDown,AttnNorm,FFNNorm per layer
	numTensors += len(e.layers) * 2 // AttnResQuery, FFNResQuery per layer

	// Build metadata.
	metadata := map[string]interface{}{
		"general.architecture":     "llama",
		"general.name":             e.name,
		"llama.vocab_size":         uint32(e.vocabSize),
		"llama.embedding_length":   uint32(e.hiddenDim),
		"llama.attention.head_count": uint32(e.numHeads),
		"llama.attention.head_count_kv": uint32(e.numKVHeads),
		"llama.block_count":        uint32(e.numLayers),
		"llama.feed_forward_length": uint32(e.ffnDim),
	}

	// GGUF magic + version.
	f.Write([]byte("GGUF"))                                   // Magic
	binary.Write(f, binary.LittleEndian, uint32(3))           // Version 3
	binary.Write(f, binary.LittleEndian, uint64(numTensors))  // Tensor count
	binary.Write(f, binary.LittleEndian, uint64(len(metadata))) // Metadata count

	// Write metadata KV pairs.
	for key, value := range metadata {
		writeExportString(f, key)
		switch v := value.(type) {
		case string:
			binary.Write(f, binary.LittleEndian, uint32(8)) // GGUF_TYPE_STRING
			writeExportString(f, v)
		case uint32:
			binary.Write(f, binary.LittleEndian, uint32(4)) // GGUF_TYPE_UINT32
			binary.Write(f, binary.LittleEndian, v)
		}
	}

	// Build tensor info list (name, shape, offset).
	type tensorInfo struct {
		name   string
		tensor *Tensor
	}
	var tensors []tensorInfo

	tensors = append(tensors, tensorInfo{"token_embd.weight", e.embeddings})
	for i, layer := range e.layers {
		tensors = append(tensors,
			tensorInfo{fmt.Sprintf("blk.%d.attn_q.weight", i), layer.WQ},
			tensorInfo{fmt.Sprintf("blk.%d.attn_k.weight", i), layer.WK},
			tensorInfo{fmt.Sprintf("blk.%d.attn_v.weight", i), layer.WV},
			tensorInfo{fmt.Sprintf("blk.%d.attn_output.weight", i), layer.WO},
			tensorInfo{fmt.Sprintf("blk.%d.ffn_gate.weight", i), layer.WGate},
			tensorInfo{fmt.Sprintf("blk.%d.ffn_up.weight", i), layer.WUp},
			tensorInfo{fmt.Sprintf("blk.%d.ffn_down.weight", i), layer.WDown},
			tensorInfo{fmt.Sprintf("blk.%d.attn_norm.weight", i), layer.AttnNorm},
			tensorInfo{fmt.Sprintf("blk.%d.ffn_norm.weight", i), layer.FFNNorm},
		)
		// AttnRes queries.
		if layer.AttnResQuery != nil {
			tensors = append(tensors,
				tensorInfo{fmt.Sprintf("blk.%d.attnres_q.weight", i), layer.AttnResQuery},
				tensorInfo{fmt.Sprintf("blk.%d.ffnres_q.weight", i), layer.FFNResQuery},
			)
		}
	}
	tensors = append(tensors, tensorInfo{"output.weight", e.lmHead})

	// Write tensor info headers.
	offset := uint64(0)
	for _, ti := range tensors {
		writeExportString(f, ti.name)
		// Number of dimensions.
		binary.Write(f, binary.LittleEndian, uint32(len(ti.tensor.Shape)))
		// Shape (reversed for GGUF convention).
		for j := len(ti.tensor.Shape) - 1; j >= 0; j-- {
			binary.Write(f, binary.LittleEndian, uint64(ti.tensor.Shape[j]))
		}
		// Data type: F32 = 0
		binary.Write(f, binary.LittleEndian, uint32(0))
		// Offset from start of tensor data section.
		binary.Write(f, binary.LittleEndian, offset)
		offset += uint64(len(ti.tensor.Data) * 4)
	}

	// Alignment padding to 32 bytes.
	pos, _ := f.Seek(0, 1)
	padding := (32 - (pos % 32)) % 32
	if padding > 0 {
		f.Write(make([]byte, padding))
	}

	// Write tensor data.
	for _, ti := range tensors {
		binary.Write(f, binary.LittleEndian, ti.tensor.Data)
	}

	totalMB := float64(offset) / (1024 * 1024)
	fmt.Printf("[GGUF Export] Saved %d tensors to %s (%.1f MB)\n", len(tensors), path, totalMB)
	return nil
}

// writeExportString writes a length-prefixed string in GGUF format.
func writeExportString(f *os.File, s string) {
	binary.Write(f, binary.LittleEndian, uint64(len(s)))
	f.Write([]byte(s))
}
