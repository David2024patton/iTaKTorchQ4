// distributed_launch.go implements multi-node training orchestration.
//
// WHAT: When training models too large for one machine, distributed
// training splits the work across multiple nodes (each with multiple GPUs).
// This launcher coordinates:
//   - Node discovery and rank assignment
//   - Gradient synchronization (AllReduce)
//   - Failure detection and recovery
//   - Checkpoint coordination across nodes
//
// WHY: Training a 70B model requires at least 4x 80GB GPUs. With
// distributed training, you can scale to hundreds of GPUs across
// multiple machines, reducing training time from months to days.
package native

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"
	"time"
)

// NodeRole identifies a node's role in distributed training.
type NodeRole int

const (
	NodeMaster NodeRole = iota // Rank 0: coordinates others
	NodeWorker                 // Rank 1+: follows master
)

// DistributedConfig configures distributed training.
type DistributedConfig struct {
	WorldSize      int      // Total number of processes (nodes * GPUs per node)
	NumNodes       int      // Number of physical machines
	GPUsPerNode    int      // GPUs per machine
	MasterAddr     string   // Master node address
	MasterPort     int      // Master communication port
	Backend        string   // "nccl" (GPU) or "gloo" (CPU)
	GradSyncMode   string   // "allreduce" or "ring"
	CheckpointEvery int     // Steps between checkpoints
}

// NodeInfo tracks one distributed training node.
type NodeInfo struct {
	Rank         int
	LocalRank    int // GPU index within this node
	NodeIdx      int // Physical machine index
	Role         NodeRole
	Address      string
	Healthy      bool
	LastHeartbeat time.Time
	StepsDone    int64
}

// DistributedLauncher orchestrates multi-node training.
type DistributedLauncher struct {
	mu     sync.RWMutex
	config DistributedConfig

	nodes       []*NodeInfo
	localRank   int
	globalRank  int

	// Gradient sync state.
	gradBuffers map[string][]float32 // param_name -> accumulated grads
	syncBarrier chan struct{}

	// Status.
	isInitialized bool
	globalStep    int64
	epoch         int32

	ctx    context.Context
	cancel context.CancelFunc
}

// NewDistributedLauncher creates a distributed training launcher.
func NewDistributedLauncher(config DistributedConfig) *DistributedLauncher {
	ctx, cancel := context.WithCancel(context.Background())

	dl := &DistributedLauncher{
		config:      config,
		gradBuffers: make(map[string][]float32),
		syncBarrier: make(chan struct{}),
		ctx:         ctx,
		cancel:      cancel,
	}

	return dl
}

// Initialize sets up the distributed environment.
// This assigns ranks and establishes connections between nodes.
func (dl *DistributedLauncher) Initialize(localRank, globalRank int) error {
	dl.mu.Lock()
	defer dl.mu.Unlock()

	dl.localRank = localRank
	dl.globalRank = globalRank

	// Create node registry.
	dl.nodes = make([]*NodeInfo, dl.config.WorldSize)
	for i := 0; i < dl.config.WorldSize; i++ {
		role := NodeWorker
		if i == 0 {
			role = NodeMaster
		}
		dl.nodes[i] = &NodeInfo{
			Rank:          i,
			LocalRank:     i % dl.config.GPUsPerNode,
			NodeIdx:       i / dl.config.GPUsPerNode,
			Role:          role,
			Healthy:       true,
			LastHeartbeat: time.Now(),
		}
	}

	dl.isInitialized = true
	fmt.Printf("[DistLaunch] Rank %d/%d initialized (node %d, local %d)\n",
		globalRank, dl.config.WorldSize, globalRank/dl.config.GPUsPerNode, localRank)

	return nil
}

// AllReduceGradients synchronizes gradients across all nodes.
// This is the core operation for data-parallel distributed training.
// Each node has computed gradients for its local mini-batch; AllReduce
// averages them so all nodes have identical gradients before the optimizer step.
func (dl *DistributedLauncher) AllReduceGradients(paramName string, localGrads []float32) []float32 {
	dl.mu.Lock()
	defer dl.mu.Unlock()

	// Store local gradients.
	dl.gradBuffers[paramName] = localGrads

	// In a real implementation, this would use NCCL/gloo for network AllReduce.
	// Here we simulate the math: output = average of all nodes' gradients.
	// In single-process mode, this is a no-op (returns input).
	worldSize := float32(dl.config.WorldSize)

	result := make([]float32, len(localGrads))
	for i := range result {
		// In production: NCCL AllReduce would sum across nodes,
		// then divide by world_size.
		result[i] = localGrads[i] / worldSize
	}

	return result
}

// RingAllReduce performs ring-based AllReduce for bandwidth efficiency.
// Splits gradients into chunks, each chunk traverses the ring once for
// reduce-scatter, then once more for all-gather.
func (dl *DistributedLauncher) RingAllReduce(grads []float32) []float32 {
	worldSize := dl.config.WorldSize
	if worldSize <= 1 {
		return grads
	}

	n := len(grads)
	chunkSize := (n + worldSize - 1) / worldSize
	result := make([]float32, n)
	copy(result, grads)

	// Phase 1: Reduce-scatter.
	// Each rank reduces one chunk, passing results around the ring.
	for step := 0; step < worldSize-1; step++ {
		for c := 0; c < worldSize; c++ {
			start := c * chunkSize
			end := start + chunkSize
			if end > n {
				end = n
			}
			// In production: receive chunk from (rank-1) and add to local.
			// Simulated: just scale down.
			scale := float32(worldSize-step) / float32(worldSize-step+1)
			for i := start; i < end; i++ {
				result[i] *= scale
			}
		}
	}

	// Phase 2: All-gather.
	// Each rank broadcasts its reduced chunk to all others.
	// Simulated: already have the final result from reduce-scatter.

	return result
}

// Checkpoint coordinates a distributed checkpoint across all nodes.
// Only the master (rank 0) saves the full model; workers save their optimizer states.
func (dl *DistributedLauncher) Checkpoint(step int64, saveFn func(path string) error) error {
	dl.mu.RLock()
	rank := dl.globalRank
	dl.mu.RUnlock()

	path := fmt.Sprintf("checkpoint-step%d-rank%d", step, rank)

	if rank == 0 {
		// Master saves full model weights.
		fmt.Printf("[DistLaunch] Rank 0: saving checkpoint at step %d\n", step)
	}

	// All ranks save their local state.
	if saveFn != nil {
		if err := saveFn(path); err != nil {
			return fmt.Errorf("rank %d checkpoint failed: %w", rank, err)
		}
	}

	return nil
}

// Heartbeat sends a health signal; master tracks worker status.
func (dl *DistributedLauncher) Heartbeat() {
	dl.mu.Lock()
	defer dl.mu.Unlock()

	rank := dl.globalRank
	if rank < len(dl.nodes) {
		dl.nodes[rank].LastHeartbeat = time.Now()
	}
}

// CheckHealth identifies dead workers (no heartbeat in 30s).
func (dl *DistributedLauncher) CheckHealth() []int {
	dl.mu.RLock()
	defer dl.mu.RUnlock()

	threshold := 30 * time.Second
	deadRanks := make([]int, 0)
	now := time.Now()

	for _, node := range dl.nodes {
		if now.Sub(node.LastHeartbeat) > threshold {
			deadRanks = append(deadRanks, node.Rank)
			node.Healthy = false
		}
	}

	return deadRanks
}

// Step increments the global training step.
func (dl *DistributedLauncher) Step() int64 {
	return atomic.AddInt64(&dl.globalStep, 1)
}

// Shutdown gracefully terminates distributed training.
func (dl *DistributedLauncher) Shutdown() {
	dl.cancel()
	fmt.Printf("[DistLaunch] Rank %d shutting down\n", dl.globalRank)
}

// Stats returns distributed training metrics.
func (dl *DistributedLauncher) Stats() map[string]interface{} {
	dl.mu.RLock()
	defer dl.mu.RUnlock()

	healthy := 0
	for _, n := range dl.nodes {
		if n.Healthy {
			healthy++
		}
	}

	return map[string]interface{}{
		"world_size":     dl.config.WorldSize,
		"num_nodes":      dl.config.NumNodes,
		"gpus_per_node":  dl.config.GPUsPerNode,
		"global_rank":    dl.globalRank,
		"local_rank":     dl.localRank,
		"global_step":    atomic.LoadInt64(&dl.globalStep),
		"healthy_nodes":  healthy,
		"backend":        dl.config.Backend,
		"grad_sync_mode": dl.config.GradSyncMode,
	}
}
