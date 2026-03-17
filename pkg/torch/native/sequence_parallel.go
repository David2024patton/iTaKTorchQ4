// sequence_parallel.go implements sequence parallelism for long-context training.
//
// WHAT: Sequence parallelism splits the sequence dimension (not weights) across
// GPUs. Each GPU processes a portion of the input sequence through LayerNorm
// and dropout, then uses AllGather before attention (which needs the full sequence).
//
// WHY: During training with long context, activation memory scales linearly
// with sequence length. Sequence parallelism distributes this memory across
// GPUs, enabling training with 4x longer contexts at the same per-GPU VRAM.
//
// COMBINED WITH TENSOR PARALLEL: In practice, sequence parallelism is used
// together with tensor parallelism. The tensor parallel group handles QKV
// projections, while sequence parallelism handles LayerNorm and dropout.
//
// GAIN: Linear memory reduction proportional to the number of GPUs.
// 4 GPUs = 4x longer context capacity during training.
package native

import (
	"fmt"
	"sync"
)

// SequenceParallelConfig configures sequence parallelism.
type SequenceParallelConfig struct {
	WorldSize  int // Number of GPUs
	Rank       int // This GPU's rank
	SeqLen     int // Total sequence length
	HiddenDim  int // Model hidden dimension
}

// SequenceParallelGroup manages sequence-parallel operations.
type SequenceParallelGroup struct {
	mu sync.Mutex

	worldSize int
	rank      int
	chunkSize int // seqLen / worldSize

	// Shared buffers for AllGather/ReduceScatter simulation.
	gatherBuffers [][]float32

	// Stats.
	allGatherCount      int64
	reduceScatterCount  int64
	totalBytes          int64
}

// NewSequenceParallelGroup creates a sequence parallel group.
func NewSequenceParallelGroup(config SequenceParallelConfig) *SequenceParallelGroup {
	chunkSize := config.SeqLen / config.WorldSize
	return &SequenceParallelGroup{
		worldSize:     config.WorldSize,
		rank:          config.Rank,
		chunkSize:     chunkSize,
		gatherBuffers: make([][]float32, config.WorldSize),
	}
}

// ScatterInput splits input data along the sequence dimension.
// input: [seqLen, hiddenDim] -> output: [seqLen/worldSize, hiddenDim].
func (sp *SequenceParallelGroup) ScatterInput(input []float32, seqLen, hiddenDim int) []float32 {
	chunkSize := seqLen / sp.worldSize
	startPos := sp.rank * chunkSize * hiddenDim
	endPos := startPos + chunkSize*hiddenDim

	chunk := make([]float32, chunkSize*hiddenDim)
	copy(chunk, input[startPos:endPos])
	return chunk
}

// AllGatherSequence gathers sequence chunks from all GPUs to reconstruct
// the full sequence. Needed before attention (which requires the full sequence).
// local: [chunkSize, hiddenDim] -> output: [seqLen, hiddenDim].
func (sp *SequenceParallelGroup) AllGatherSequence(rank int, local []float32) []float32 {
	sp.mu.Lock()
	sp.gatherBuffers[rank] = make([]float32, len(local))
	copy(sp.gatherBuffers[rank], local)
	sp.allGatherCount++
	sp.totalBytes += int64(len(local) * 4)
	sp.mu.Unlock()

	// Concatenate all chunks in rank order.
	totalLen := len(local) * sp.worldSize
	result := make([]float32, 0, totalLen)

	sp.mu.Lock()
	for i := 0; i < sp.worldSize; i++ {
		if sp.gatherBuffers[i] != nil {
			result = append(result, sp.gatherBuffers[i]...)
		}
	}
	sp.mu.Unlock()

	return result
}

// ReduceScatterGrad reduces gradients across GPUs and scatters the result.
// Each GPU gets the reduced gradient for its sequence chunk.
// fullGrad: [seqLen, hiddenDim] -> output: [chunkSize, hiddenDim].
func (sp *SequenceParallelGroup) ReduceScatterGrad(rank int, fullGrad []float32, hiddenDim int) []float32 {
	sp.mu.Lock()
	sp.gatherBuffers[rank] = make([]float32, len(fullGrad))
	copy(sp.gatherBuffers[rank], fullGrad)
	sp.reduceScatterCount++
	sp.totalBytes += int64(len(fullGrad) * 4)
	sp.mu.Unlock()

	// Sum gradients from all ranks.
	sp.mu.Lock()
	reduced := make([]float32, len(fullGrad))
	for i := 0; i < sp.worldSize; i++ {
		if sp.gatherBuffers[i] != nil && len(sp.gatherBuffers[i]) == len(fullGrad) {
			for j := range reduced {
				reduced[j] += sp.gatherBuffers[i][j]
			}
		}
	}
	sp.mu.Unlock()

	// Scatter: extract this rank's chunk.
	startPos := rank * sp.chunkSize * hiddenDim
	endPos := startPos + sp.chunkSize*hiddenDim
	if endPos > len(reduced) {
		endPos = len(reduced)
	}

	chunk := make([]float32, sp.chunkSize*hiddenDim)
	copy(chunk, reduced[startPos:endPos])
	return chunk
}

// SequenceParallelLayerNorm applies LayerNorm on a sequence-parallel chunk.
// This runs on each GPU's local sequence chunk without communication.
func SequenceParallelLayerNorm(chunk, weight, bias []float32, chunkSize, hiddenDim int) []float32 {
	out := make([]float32, len(chunk))

	for s := 0; s < chunkSize; s++ {
		offset := s * hiddenDim

		// Compute mean.
		var mean float64
		for d := 0; d < hiddenDim; d++ {
			mean += float64(chunk[offset+d])
		}
		mean /= float64(hiddenDim)

		// Compute variance.
		var variance float64
		for d := 0; d < hiddenDim; d++ {
			diff := float64(chunk[offset+d]) - mean
			variance += diff * diff
		}
		variance /= float64(hiddenDim)

		// Normalize.
		invStd := float32(1.0 / (variance + 1e-5))
		for d := 0; d < hiddenDim; d++ {
			normalized := (chunk[offset+d] - float32(mean)) * invStd
			out[offset+d] = normalized*weight[d] + bias[d]
		}
	}

	return out
}

// SequenceParallelDropout applies dropout on a sequence-parallel chunk.
// Each GPU independently drops elements in its local chunk.
func SequenceParallelDropout(chunk []float32, dropRate float32, training bool) []float32 {
	if !training || dropRate <= 0 {
		return chunk
	}

	out := make([]float32, len(chunk))
	scale := 1.0 / (1.0 - dropRate)

	for i, v := range chunk {
		// Deterministic dropout based on position (reproducible across runs).
		hash := uint32(i*2654435761) >> 16
		if float32(hash%1000)/1000.0 > dropRate {
			out[i] = v * scale
		}
	}

	return out
}

// Stats returns sequence parallel metrics.
func (sp *SequenceParallelGroup) Stats() map[string]interface{} {
	sp.mu.Lock()
	defer sp.mu.Unlock()
	return map[string]interface{}{
		"world_size":           sp.worldSize,
		"chunk_size":           sp.chunkSize,
		"all_gather_count":     sp.allGatherCount,
		"reduce_scatter_count": sp.reduceScatterCount,
		"total_bytes":          fmt.Sprintf("%.1f MB", float64(sp.totalBytes)/(1024*1024)),
	}
}
