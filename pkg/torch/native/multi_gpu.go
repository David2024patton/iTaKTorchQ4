// multi_gpu.go implements tensor parallelism for splitting models across multiple GPUs.
//
// WHAT: For models too large to fit on a single GPU, tensor parallelism splits
// weight matrices across GPUs. Each GPU holds a shard of each weight and
// computes its portion of the output. Results are then combined via AllReduce.
//
// HOW: Weight matrices are split along their output dimension. For a [M, N]
// matrix on 2 GPUs, GPU 0 holds [M, N/2] and GPU 1 holds [M, N/2]. Each
// GPU computes a partial result, and AllReduce sums them.
package native

import (
	"fmt"
	"sync"
)

// TensorParallelConfig controls model distribution across GPUs.
type TensorParallelConfig struct {
	NumGPUs       int    // Number of GPUs to split across
	SplitStrategy string // "column" (default) or "row"
}

// TensorParallelGroup manages a set of GPUs for parallel computation.
type TensorParallelGroup struct {
	config  TensorParallelConfig
	shards  []*GPUBackend // One backend per GPU
	mu      sync.Mutex
}

// NewTensorParallelGroup creates a parallel group from available GPU backends.
func NewTensorParallelGroup(backends []*GPUBackend, config TensorParallelConfig) *TensorParallelGroup {
	if len(backends) == 0 {
		fmt.Println("[TensorParallel] No GPU backends provided")
		return nil
	}

	config.NumGPUs = len(backends)
	fmt.Printf("[TensorParallel] Initialized with %d GPUs\n", config.NumGPUs)

	return &TensorParallelGroup{
		config: config,
		shards: backends,
	}
}

// ShardWeight splits a weight tensor across GPUs along the output dimension.
func (tp *TensorParallelGroup) ShardWeight(weight *Tensor) []*Tensor {
	if len(weight.Shape) != 2 {
		shards := make([]*Tensor, tp.config.NumGPUs)
		for i := range shards {
			shards[i] = weight
		}
		return shards
	}

	rows := weight.Shape[0]
	cols := weight.Shape[1]
	numGPUs := tp.config.NumGPUs

	rowsPerGPU := rows / numGPUs
	shards := make([]*Tensor, numGPUs)

	for g := 0; g < numGPUs; g++ {
		startRow := g * rowsPerGPU
		endRow := startRow + rowsPerGPU
		if g == numGPUs-1 {
			endRow = rows
		}

		shardRows := endRow - startRow
		shard := NewTensor([]int{shardRows, cols})
		copy(shard.Data, weight.Data[startRow*cols:endRow*cols])
		shards[g] = shard
	}

	return shards
}

// ParallelMatMul executes MatMul across all GPUs in parallel.
func (tp *TensorParallelGroup) ParallelMatMul(weightShards []*Tensor, x *Tensor) *Tensor {
	numGPUs := tp.config.NumGPUs
	results := make([]*Tensor, numGPUs)
	var wg sync.WaitGroup

	for g := 0; g < numGPUs; g++ {
		wg.Add(1)
		go func(gpuIdx int) {
			defer wg.Done()
			shard := weightShards[gpuIdx]

			if gpuIdx < len(tp.shards) && tp.shards[gpuIdx] != nil && tp.shards[gpuIdx].IsAvailable() {
				results[gpuIdx] = tp.shards[gpuIdx].MatMulGPU(x, shard)
			} else {
				results[gpuIdx] = safeMatMul(x, shard, shard.Shape[0])
			}
		}(g)
	}

	wg.Wait()
	return tp.concatenateResults(results)
}

// concatenateResults combines partial results from multiple GPUs.
func (tp *TensorParallelGroup) concatenateResults(results []*Tensor) *Tensor {
	if len(results) == 1 {
		return results[0]
	}

	totalLen := 0
	for _, r := range results {
		if r != nil {
			totalLen += len(r.Data)
		}
	}

	combined := NewTensor([]int{totalLen})
	offset := 0
	for _, r := range results {
		if r != nil {
			copy(combined.Data[offset:], r.Data)
			offset += len(r.Data)
		}
	}

	return combined
}

// AllReduceSum sums tensors across all GPUs.
func (tp *TensorParallelGroup) AllReduceSum(tensors []*Tensor) *Tensor {
	if len(tensors) == 0 {
		return nil
	}

	result := NewTensor(append([]int(nil), tensors[0].Shape...))
	copy(result.Data, tensors[0].Data)

	for _, t := range tensors[1:] {
		if t == nil {
			continue
		}
		for i := range result.Data {
			if i < len(t.Data) {
				result.Data[i] += t.Data[i]
			}
		}
	}

	return result
}

// ShardModel distributes all model weights across GPUs.
func (tp *TensorParallelGroup) ShardModel(engine *NativeEngine) map[int]map[string][]*Tensor {
	shardedLayers := make(map[int]map[string][]*Tensor)

	for i, layer := range engine.layers {
		shardedLayers[i] = map[string][]*Tensor{
			"WQ":    tp.ShardWeight(layer.WQ),
			"WK":    tp.ShardWeight(layer.WK),
			"WV":    tp.ShardWeight(layer.WV),
			"WO":    tp.ShardWeight(layer.WO),
			"WGate": tp.ShardWeight(layer.WGate),
			"WUp":   tp.ShardWeight(layer.WUp),
			"WDown": tp.ShardWeight(layer.WDown),
		}
	}

	totalShards := len(engine.layers) * 7 * tp.config.NumGPUs
	fmt.Printf("[TensorParallel] Sharded %d weight matrices across %d GPUs (%d total shards)\n",
		len(engine.layers)*7, tp.config.NumGPUs, totalShards)

	return shardedLayers
}
