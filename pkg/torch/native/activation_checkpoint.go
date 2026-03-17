// activation_checkpoint.go implements gradient checkpointing for training.
//
// WHAT: During backpropagation, we need the intermediate activations from the
// forward pass. Normally these are all kept in memory. Activation
// checkpointing trades compute for memory by only saving activations at
// checkpoint boundaries and recomputing the rest during backward.
//
// SAVINGS: For a 32-layer model, checkpointing every 4 layers reduces
// activation memory by ~8x at the cost of ~33% more compute.
package native

import (
	"fmt"
)

// CheckpointConfig controls activation checkpointing behavior.
type CheckpointConfig struct {
	Enabled       bool
	Interval      int  // Checkpoint every N layers (default: 4)
	FullRecompute bool // If true, recompute all; if false, only between checkpoints
}

// DefaultCheckpointConfig returns standard settings.
func DefaultCheckpointConfig() CheckpointConfig {
	return CheckpointConfig{
		Enabled:  true,
		Interval: 4,
	}
}

// ActivationCheckpoint stores saved activations at checkpoint boundaries.
type ActivationCheckpoint struct {
	config      CheckpointConfig
	checkpoints map[int]*Tensor // Layer index -> saved activation
	numLayers   int
	savedBytes  int64
	recomputes  int64
}

// NewActivationCheckpoint creates a checkpoint manager.
func NewActivationCheckpoint(numLayers int, config CheckpointConfig) *ActivationCheckpoint {
	c := &ActivationCheckpoint{
		config:      config,
		checkpoints: make(map[int]*Tensor),
		numLayers:   numLayers,
	}
	fmt.Printf("[Checkpoint] Enabled: every %d layers (%d checkpoints for %d layers)\n",
		config.Interval, numLayers/config.Interval, numLayers)
	return c
}

// IsCheckpointLayer returns true if this layer should save its activation.
func (c *ActivationCheckpoint) IsCheckpointLayer(layerIdx int) bool {
	if !c.config.Enabled {
		return false
	}
	return layerIdx%c.config.Interval == 0
}

// SaveActivation stores the activation at a checkpoint boundary.
func (c *ActivationCheckpoint) SaveActivation(layerIdx int, activation *Tensor) {
	if !c.IsCheckpointLayer(layerIdx) {
		return
	}

	// Deep copy the activation.
	saved := NewTensor(activation.Shape)
	copy(saved.Data, activation.Data)
	c.checkpoints[layerIdx] = saved
	c.savedBytes += int64(len(saved.Data) * 4)
}

// GetActivation retrieves a saved activation. If not at a checkpoint,
// returns the nearest previous checkpoint for recomputation.
func (c *ActivationCheckpoint) GetActivation(layerIdx int) (*Tensor, int, bool) {
	// Direct checkpoint hit.
	if act, ok := c.checkpoints[layerIdx]; ok {
		return act, layerIdx, true
	}

	// Find nearest previous checkpoint.
	checkpointLayer := (layerIdx / c.config.Interval) * c.config.Interval
	if act, ok := c.checkpoints[checkpointLayer]; ok {
		c.recomputes++
		return act, checkpointLayer, false // Caller must recompute from checkpointLayer to layerIdx
	}

	return nil, 0, false
}

// RecomputeRange returns the layer range that needs recomputation
// to get the activation at targetLayer.
func (c *ActivationCheckpoint) RecomputeRange(targetLayer int) (startLayer, endLayer int) {
	startLayer = (targetLayer / c.config.Interval) * c.config.Interval
	endLayer = targetLayer
	return startLayer, endLayer
}

// Clear removes all saved activations (call after backward pass).
func (c *ActivationCheckpoint) Clear() {
	c.checkpoints = make(map[int]*Tensor)
	c.savedBytes = 0
}

// MemorySavingsRatio returns the approximate memory savings vs full caching.
func (c *ActivationCheckpoint) MemorySavingsRatio() float64 {
	if c.numLayers == 0 {
		return 1.0
	}
	checkpoints := c.numLayers / c.config.Interval
	if c.numLayers%c.config.Interval != 0 {
		checkpoints++
	}
	return float64(c.numLayers) / float64(checkpoints)
}

// Stats returns checkpoint statistics.
func (c *ActivationCheckpoint) Stats() map[string]interface{} {
	return map[string]interface{}{
		"num_checkpoints":  len(c.checkpoints),
		"total_layers":     c.numLayers,
		"interval":         c.config.Interval,
		"saved_mb":         float64(c.savedBytes) / (1024 * 1024),
		"recomputes":       c.recomputes,
		"memory_savings_x": fmt.Sprintf("%.1fx", c.MemorySavingsRatio()),
	}
}
