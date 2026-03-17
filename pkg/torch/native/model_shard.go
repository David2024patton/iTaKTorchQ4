// model_shard.go implements automatic model sharding across available GPUs.
//
// WHAT: When a model is too large for one GPU, model sharding automatically
// determines how to split the model across available GPUs. It combines
// tensor parallelism (split within layers) and pipeline parallelism
// (split between layers) based on available memory.
//
// STRATEGY:
//   1. Profile available GPU memory on each device
//   2. Estimate per-layer memory (weights + KV cache + activations)
//   3. Choose the best parallelism strategy:
//      - Model fits on 1 GPU: no sharding
//      - 2-4 GPUs: tensor parallel (best latency)
//      - 4+ GPUs: hybrid TP + PP (best throughput)
//   4. Generate a device map: layerIdx -> gpuIdx
//
// GAIN: Zero-config multi-GPU deployment. User loads a model and the
// engine automatically figures out the optimal placement.
package native

import (
	"fmt"
)

// DeviceInfo describes one available GPU.
type DeviceInfo struct {
	DeviceID     int
	Name         string
	TotalMemory  int64 // Total VRAM in bytes
	FreeMemory   int64 // Available VRAM in bytes
	ComputeUnits int   // SM count or CU count
}

// ShardStrategy identifies the parallelism approach.
type ShardStrategy int

const (
	ShardNone     ShardStrategy = iota // Single GPU
	ShardTP                            // Tensor parallel only
	ShardPP                            // Pipeline parallel only
	ShardHybridTP_PP                   // Tensor + Pipeline parallel
)

// ShardPlan describes how a model is distributed across GPUs.
type ShardPlan struct {
	Strategy   ShardStrategy
	DeviceMap  map[int]int   // layerIdx -> deviceID
	TPDegree   int           // Tensor parallel degree (1 = no TP)
	PPDegree   int           // Pipeline parallel degree (1 = no PP)
	Devices    []DeviceInfo
	MemoryUsed map[int]int64 // deviceID -> estimated bytes used
}

// ModelMemProfile estimates the memory footprint of a model.
type ModelMemProfile struct {
	WeightsPerLayer   int64 // Bytes for one layer's weights
	KVCachePerToken   int64 // Bytes for KV cache per token per layer
	ActivationPerToken int64 // Bytes for activations per token
	EmbeddingBytes    int64 // Embedding + LM head
	TotalWeights      int64 // Total model weight bytes
	NumLayers         int
}

// ProfileModel estimates memory requirements from a model config.
func ProfileModel(config ModelConfig) ModelMemProfile {
	headDim := config.HeadDim
	if headDim == 0 && config.NumHeads > 0 {
		headDim = config.HiddenDim / config.NumHeads
	}

	kvHeads := config.NumKVHeads
	if kvHeads == 0 {
		kvHeads = config.NumHeads
	}

	// Per-layer weights (FP16 = 2 bytes per param).
	qkvParams := int64(config.HiddenDim * config.NumHeads * headDim * 3)  // Q, K, V
	outParams := int64(config.HiddenDim * config.NumHeads * headDim)       // Output proj
	ffnParams := int64(config.HiddenDim * config.IntermediateDim * 3)      // Gate, Up, Down
	normParams := int64(config.HiddenDim * 2)                              // RMSNorm * 2
	layerParams := qkvParams + outParams + ffnParams + normParams

	if config.IsMoE {
		// MoE: multiply FFN by number of experts.
		ffnMoE := int64(config.HiddenDim*config.IntermediateDim*3) * int64(config.NumExperts)
		layerParams = qkvParams + outParams + ffnMoE + normParams
	}

	weightsPerLayer := layerParams * 2 // FP16

	// KV cache per token per layer: 2 * numKVHeads * headDim * 2bytes.
	kvPerToken := int64(2 * kvHeads * headDim * 2)

	// Embedding: vocabSize * hiddenDim * 2bytes.
	embedBytes := int64(config.VocabSize * config.HiddenDim * 2)

	return ModelMemProfile{
		WeightsPerLayer:    weightsPerLayer,
		KVCachePerToken:    kvPerToken,
		ActivationPerToken: int64(config.HiddenDim * 4), // Rough estimate.
		EmbeddingBytes:     embedBytes,
		TotalWeights:       weightsPerLayer*int64(config.NumLayers) + embedBytes,
		NumLayers:          config.NumLayers,
	}
}

// AutoShard determines the optimal sharding plan for a model across devices.
func AutoShard(profile ModelMemProfile, devices []DeviceInfo) ShardPlan {
	plan := ShardPlan{
		DeviceMap:  make(map[int]int),
		MemoryUsed: make(map[int]int64),
		Devices:    devices,
		TPDegree:   1,
		PPDegree:   1,
	}

	numDevices := len(devices)
	if numDevices == 0 {
		return plan
	}

	// Can the model fit on a single GPU?
	// Reserve 20% for KV cache and activations.
	overhead := float64(0.80)
	if float64(devices[0].FreeMemory)*overhead >= float64(profile.TotalWeights) {
		// Single GPU: no sharding needed.
		plan.Strategy = ShardNone
		for i := 0; i < profile.NumLayers; i++ {
			plan.DeviceMap[i] = devices[0].DeviceID
		}
		plan.MemoryUsed[devices[0].DeviceID] = profile.TotalWeights
		return plan
	}

	if numDevices <= 4 {
		// Tensor parallel: split each layer across all GPUs.
		plan.Strategy = ShardTP
		plan.TPDegree = numDevices
		perDeviceWeight := profile.TotalWeights / int64(numDevices)
		for i := 0; i < profile.NumLayers; i++ {
			// All layers on all devices (sharded).
			plan.DeviceMap[i] = -1 // -1 = all devices
		}
		for _, d := range devices {
			plan.MemoryUsed[d.DeviceID] = perDeviceWeight
		}
		return plan
	}

	// Hybrid TP + PP for >=4 GPUs.
	// Use TP within groups, PP across groups.
	tpDegree := 2
	if numDevices >= 8 {
		tpDegree = 4
	}
	ppDegree := numDevices / tpDegree

	plan.Strategy = ShardHybridTP_PP
	plan.TPDegree = tpDegree
	plan.PPDegree = ppDegree

	layersPerStage := profile.NumLayers / ppDegree
	for i := 0; i < profile.NumLayers; i++ {
		stageIdx := i / layersPerStage
		if stageIdx >= ppDegree {
			stageIdx = ppDegree - 1
		}
		// Map to the first device in this PP stage's TP group.
		deviceIdx := stageIdx * tpDegree
		plan.DeviceMap[i] = devices[deviceIdx].DeviceID
	}

	// Estimate per-device memory.
	perStageWeight := int64(layersPerStage) * profile.WeightsPerLayer / int64(tpDegree)
	for _, d := range devices {
		plan.MemoryUsed[d.DeviceID] = perStageWeight
	}

	return plan
}

// Describe returns a human-readable description of the shard plan.
func (p ShardPlan) Describe() string {
	stratNames := map[ShardStrategy]string{
		ShardNone:        "Single GPU",
		ShardTP:          "Tensor Parallel",
		ShardPP:          "Pipeline Parallel",
		ShardHybridTP_PP: "Hybrid TP+PP",
	}

	desc := fmt.Sprintf("Strategy: %s (TP=%d, PP=%d)\n", stratNames[p.Strategy], p.TPDegree, p.PPDegree)
	desc += fmt.Sprintf("Devices: %d GPUs\n", len(p.Devices))

	for devID, memUsed := range p.MemoryUsed {
		desc += fmt.Sprintf("  GPU %d: %.1f MB used\n", devID, float64(memUsed)/(1024*1024))
	}

	return desc
}
