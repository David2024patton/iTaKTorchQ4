// pipeline_parallel.go implements pipeline parallelism for training.
//
// WHAT: Pipeline parallelism splits the model's layers across multiple GPUs.
// GPU 0 runs layers 0-7, GPU 1 runs layers 8-15, GPU 2 runs layers 16-23, etc.
// Micro-batches flow through the pipeline, keeping all GPUs busy.
//
// WHY: For training very large models that don't fit on one GPU even with
// tensor parallelism. Pipeline parallelism requires less communication than
// tensor parallelism (only activations at stage boundaries vs AllReduce
// after every layer).
//
// SCHEDULE:
//   1F1B (One Forward One Backward): The optimal schedule that minimizes
//   the "pipeline bubble" (idle time). Each GPU alternates between forward
//   and backward passes on different micro-batches.
//
//   GPU 0: F0 F1 F2 F3 B0 B1 B2 B3
//   GPU 1:    F0 F1 F2 F3 B0 B1 B2 B3
//   GPU 2:       F0 F1 F2 F3 B0 B1 B2 B3
//   GPU 3:          F0 F1 F2 F3 B0 B1 B2 B3
package native

import (
	"context"
	"fmt"
	"sync"
)

// PipelineStage represents one GPU's portion of the model.
type PipelineStage struct {
	StageID    int
	StartLayer int
	EndLayer   int // Exclusive.

	// Communication buffers.
	InputCh  chan *PipelinePayload // Receives activations from previous stage.
	OutputCh chan *PipelinePayload // Sends activations to next stage.
	GradCh   chan *PipelinePayload // Receives gradients from next stage (backward).
}

// PipelinePayload carries data between pipeline stages.
type PipelinePayload struct {
	MicroBatchID int
	Data         []float32 // Activation or gradient data.
	IsBackward   bool
}

// PipelineConfig configures pipeline parallelism.
type PipelineConfig struct {
	NumStages      int // Number of pipeline stages (GPUs).
	NumLayers      int // Total model layers.
	NumMicroBatches int // Micro-batches per pipeline step.
	BufferSize     int  // Channel buffer size.
}

// PipelineScheduler manages the 1F1B pipeline schedule.
type PipelineScheduler struct {
	mu     sync.Mutex
	config PipelineConfig
	stages []*PipelineStage

	// Stats.
	totalForward  int64
	totalBackward int64
	bubbleSteps   int64 // Idle steps (pipeline bubble overhead).
}

// NewPipelineScheduler creates a pipeline parallel scheduler.
func NewPipelineScheduler(config PipelineConfig) *PipelineScheduler {
	if config.BufferSize <= 0 {
		config.BufferSize = config.NumMicroBatches
	}

	layersPerStage := config.NumLayers / config.NumStages
	stages := make([]*PipelineStage, config.NumStages)

	for i := 0; i < config.NumStages; i++ {
		stages[i] = &PipelineStage{
			StageID:    i,
			StartLayer: i * layersPerStage,
			EndLayer:   (i + 1) * layersPerStage,
			InputCh:    make(chan *PipelinePayload, config.BufferSize),
			OutputCh:   make(chan *PipelinePayload, config.BufferSize),
			GradCh:     make(chan *PipelinePayload, config.BufferSize),
		}
		// Last stage gets remaining layers.
		if i == config.NumStages-1 {
			stages[i].EndLayer = config.NumLayers
		}
	}

	// Wire stages together: stage[i].OutputCh -> stage[i+1].InputCh.
	for i := 0; i < config.NumStages-1; i++ {
		stages[i].OutputCh = stages[i+1].InputCh
	}
	// Wire backward: stage[i+1].GradCh -> stage[i].GradCh.
	for i := config.NumStages - 1; i > 0; i-- {
		stages[i].GradCh = stages[i-1].GradCh
	}

	return &PipelineScheduler{
		config: config,
		stages: stages,
	}
}

// GetStage returns the pipeline stage for a given rank.
func (ps *PipelineScheduler) GetStage(rank int) *PipelineStage {
	if rank < 0 || rank >= len(ps.stages) {
		return nil
	}
	return ps.stages[rank]
}

// Run1F1BSchedule executes the 1F1B pipeline schedule for one training step.
// Each stage processes numMicroBatches forward and backward passes.
//
// forwardFn: processes one micro-batch through this stage's layers.
// backwardFn: computes gradients for one micro-batch through this stage's layers.
func (ps *PipelineScheduler) Run1F1BSchedule(
	ctx context.Context,
	rank int,
	forwardFn func(microBatchID int, input []float32) ([]float32, error),
	backwardFn func(microBatchID int, gradOutput []float32) ([]float32, error),
) error {
	stage := ps.stages[rank]
	numMB := ps.config.NumMicroBatches
	numStages := ps.config.NumStages

	// Warmup phase: forward passes to fill the pipeline.
	warmupSteps := numStages - rank - 1
	if warmupSteps > numMB {
		warmupSteps = numMB
	}

	// Phase 1: Warmup forward passes.
	for i := 0; i < warmupSteps; i++ {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		input, err := ps.receiveForward(stage, rank)
		if err != nil {
			return err
		}

		output, err := forwardFn(i, input)
		if err != nil {
			return err
		}

		if rank < numStages-1 {
			stage.OutputCh <- &PipelinePayload{MicroBatchID: i, Data: output}
		}
		ps.mu.Lock()
		ps.totalForward++
		ps.mu.Unlock()
	}

	// Phase 2: Steady state (1F1B alternation).
	steadySteps := numMB - warmupSteps
	for i := 0; i < steadySteps; i++ {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		mbIdx := warmupSteps + i

		// Forward.
		input, err := ps.receiveForward(stage, rank)
		if err != nil {
			return err
		}
		output, err := forwardFn(mbIdx, input)
		if err != nil {
			return err
		}
		if rank < numStages-1 {
			stage.OutputCh <- &PipelinePayload{MicroBatchID: mbIdx, Data: output}
		}

		// Backward (for the micro-batch that has completed its forward).
		gradOut, err := ps.receiveBackward(stage, rank, numStages)
		if err != nil {
			return err
		}
		gradInput, err := backwardFn(i, gradOut)
		if err != nil {
			return err
		}
		if rank > 0 {
			ps.stages[rank-1].GradCh <- &PipelinePayload{MicroBatchID: i, Data: gradInput, IsBackward: true}
		}

		ps.mu.Lock()
		ps.totalForward++
		ps.totalBackward++
		ps.mu.Unlock()
	}

	// Phase 3: Cooldown backward passes.
	for i := steadySteps; i < numMB; i++ {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		gradOut, err := ps.receiveBackward(stage, rank, numStages)
		if err != nil {
			return err
		}
		gradInput, err := backwardFn(i, gradOut)
		if err != nil {
			return err
		}
		if rank > 0 {
			ps.stages[rank-1].GradCh <- &PipelinePayload{MicroBatchID: i, Data: gradInput, IsBackward: true}
		}

		ps.mu.Lock()
		ps.totalBackward++
		ps.mu.Unlock()
	}

	return nil
}

// receiveForward gets input data for a forward pass.
func (ps *PipelineScheduler) receiveForward(stage *PipelineStage, rank int) ([]float32, error) {
	if rank == 0 {
		return nil, nil // First stage generates its own input.
	}
	payload := <-stage.InputCh
	return payload.Data, nil
}

// receiveBackward gets gradient data for a backward pass.
func (ps *PipelineScheduler) receiveBackward(stage *PipelineStage, rank, numStages int) ([]float32, error) {
	if rank == numStages-1 {
		return nil, nil // Last stage computes loss gradient.
	}
	payload := <-stage.GradCh
	return payload.Data, nil
}

// BubbleRatio returns the fraction of time wasted in the pipeline bubble.
// Optimal 1F1B: bubble = (numStages - 1) / numMicroBatches.
func (ps *PipelineScheduler) BubbleRatio() float64 {
	return float64(ps.config.NumStages-1) / float64(ps.config.NumMicroBatches)
}

// Stats returns pipeline parallel metrics.
func (ps *PipelineScheduler) Stats() map[string]interface{} {
	ps.mu.Lock()
	defer ps.mu.Unlock()
	return map[string]interface{}{
		"num_stages":       ps.config.NumStages,
		"num_micro_batches": ps.config.NumMicroBatches,
		"total_forward":    ps.totalForward,
		"total_backward":   ps.totalBackward,
		"bubble_ratio":     fmt.Sprintf("%.1f%%", ps.BubbleRatio()*100),
	}
}
