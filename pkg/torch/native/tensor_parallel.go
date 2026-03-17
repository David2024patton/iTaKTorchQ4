// tensor_parallel.go implements tensor parallelism for multi-GPU inference.
//
// WHAT: Tensor parallelism splits individual weight matrices across multiple
// GPUs. Each GPU holds a shard of every layer's weights and computes a
// partial result. Results are combined via AllReduce after each layer.
//
// EXAMPLE with 2 GPUs and a [4096, 4096] weight matrix:
//   GPU 0: W[0:2048, :] -> partial_out_0 = W_0 @ x
//   GPU 1: W[2048:4096, :] -> partial_out_1 = W_1 @ x
//   AllReduce: out = partial_out_0 + partial_out_1
//
// WHY: Models too large for one GPU can be served across multiple GPUs
// with near-linear speedup. A 70B model that doesn't fit on 1x 24GB GPU
// can run on 4x 24GB GPUs at ~3.5x speed.
//
// ARCHITECTURE CHOICES:
//   Column Parallel: Split output dimension (QKV projections, gate/up FFN)
//   Row Parallel:    Split input dimension (attention output, down FFN)
package native

import (
	"fmt"
	"sync"
)

// TPConfig configures tensor parallelism.
type TPConfig struct {
	WorldSize  int // Number of GPUs (tensor parallel degree)
	Rank       int // This GPU's rank (0..WorldSize-1)
}

// TPGroup manages communication between parallel GPU workers.
type TPGroup struct {
	mu sync.Mutex

	worldSize int
	rank      int

	// Simulated AllReduce buffers (in production, these would be NCCL calls).
	reduceBuffers [][]float32 // [worldSize][bufferSize]
	barriers      []chan struct{}

	// Stats.
	allReduceCount int64
	allGatherCount int64
	totalBytes     int64
}

// NewTPGroup creates a TP communication group.
func NewTPGroup(worldSize int) *TPGroup {
	tpg := &TPGroup{
		worldSize:     worldSize,
		reduceBuffers: make([][]float32, worldSize),
		barriers:      make([]chan struct{}, worldSize),
	}
	for i := 0; i < worldSize; i++ {
		tpg.barriers[i] = make(chan struct{}, 1)
	}
	return tpg
}

// ColumnParallelShard splits a weight matrix along the output dimension.
// Full weight: [outDim, inDim] -> Shard: [outDim/worldSize, inDim].
func ColumnParallelShard(weight []float32, outDim, inDim, rank, worldSize int) []float32 {
	shardOutDim := outDim / worldSize
	shard := make([]float32, shardOutDim*inDim)

	startRow := rank * shardOutDim
	for i := 0; i < shardOutDim; i++ {
		srcOff := (startRow + i) * inDim
		dstOff := i * inDim
		copy(shard[dstOff:dstOff+inDim], weight[srcOff:srcOff+inDim])
	}

	return shard
}

// RowParallelShard splits a weight matrix along the input dimension.
// Full weight: [outDim, inDim] -> Shard: [outDim, inDim/worldSize].
func RowParallelShard(weight []float32, outDim, inDim, rank, worldSize int) []float32 {
	shardInDim := inDim / worldSize
	shard := make([]float32, outDim*shardInDim)

	startCol := rank * shardInDim
	for i := 0; i < outDim; i++ {
		srcOff := i*inDim + startCol
		dstOff := i * shardInDim
		copy(shard[dstOff:dstOff+shardInDim], weight[srcOff:srcOff+shardInDim])
	}

	return shard
}

// ColumnParallelForward computes matmul with a column-sharded weight.
// Each GPU computes a portion of the output dimension.
// input: [inDim], weight: [shardOutDim, inDim] -> output: [shardOutDim].
func ColumnParallelForward(input, weight []float32, shardOutDim, inDim int) []float32 {
	output := make([]float32, shardOutDim)
	for i := 0; i < shardOutDim; i++ {
		var sum float64
		wOff := i * inDim
		for j := 0; j < inDim; j++ {
			sum += float64(weight[wOff+j]) * float64(input[j])
		}
		output[i] = float32(sum)
	}
	return output
}

// RowParallelForward computes matmul with a row-sharded weight.
// Each GPU processes a portion of the input dimension.
// input: [shardInDim], weight: [outDim, shardInDim] -> output: [outDim].
// The outputs from all GPUs must be summed (AllReduce).
func RowParallelForward(input, weight []float32, outDim, shardInDim int) []float32 {
	output := make([]float32, outDim)
	for i := 0; i < outDim; i++ {
		var sum float64
		wOff := i * shardInDim
		for j := 0; j < shardInDim; j++ {
			sum += float64(weight[wOff+j]) * float64(input[j])
		}
		output[i] = float32(sum)
	}
	return output
}

// AllReduce sums partial results from all GPUs.
// In production, this uses NCCL. Here we simulate with shared memory.
func (tpg *TPGroup) AllReduce(rank int, data []float32) []float32 {
	tpg.mu.Lock()

	// Store this rank's contribution.
	tpg.reduceBuffers[rank] = make([]float32, len(data))
	copy(tpg.reduceBuffers[rank], data)

	tpg.allReduceCount++
	tpg.totalBytes += int64(len(data) * 4)
	tpg.mu.Unlock()

	// In a real implementation, NCCL would handle the synchronization.
	// Here we simulate by summing all buffers.
	result := make([]float32, len(data))
	tpg.mu.Lock()
	for _, buf := range tpg.reduceBuffers {
		if buf != nil && len(buf) == len(data) {
			for i := range result {
				result[i] += buf[i]
			}
		}
	}
	tpg.mu.Unlock()

	return result
}

// AllGather concatenates partial results from all GPUs.
// Used after column-parallel layers to reconstruct full output.
func (tpg *TPGroup) AllGather(rank int, data []float32) []float32 {
	tpg.mu.Lock()
	tpg.reduceBuffers[rank] = make([]float32, len(data))
	copy(tpg.reduceBuffers[rank], data)
	tpg.allGatherCount++
	tpg.totalBytes += int64(len(data) * 4)
	tpg.mu.Unlock()

	// Concatenate all shards.
	result := make([]float32, 0, len(data)*tpg.worldSize)
	tpg.mu.Lock()
	for i := 0; i < tpg.worldSize; i++ {
		if tpg.reduceBuffers[i] != nil {
			result = append(result, tpg.reduceBuffers[i]...)
		}
	}
	tpg.mu.Unlock()

	return result
}

// ShardAttentionHeads distributes attention heads across GPUs.
// Returns the head indices this rank should compute.
func ShardAttentionHeads(numHeads, rank, worldSize int) (startHead, endHead int) {
	headsPerGPU := numHeads / worldSize
	startHead = rank * headsPerGPU
	endHead = startHead + headsPerGPU
	if rank == worldSize-1 {
		endHead = numHeads // Last GPU gets remaining heads.
	}
	return
}

// Stats returns tensor parallel metrics.
func (tpg *TPGroup) Stats() map[string]interface{} {
	tpg.mu.Lock()
	defer tpg.mu.Unlock()
	return map[string]interface{}{
		"world_size":       tpg.worldSize,
		"all_reduce_count": tpg.allReduceCount,
		"all_gather_count": tpg.allGatherCount,
		"total_bytes":      fmt.Sprintf("%.1f MB", float64(tpg.totalBytes)/(1024*1024)),
	}
}
