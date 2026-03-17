// attn_residual.go implements Attention Residuals (AttnRes) from the
// Kimi/MoonshotAI paper "Attention Residuals" (March 2026).
//
// WHAT: Replaces fixed-weight residual connections (x = x + layer_output)
// with learned softmax attention across network depth. Each layer has a
// pseudo-query vector that attends over all previous layer outputs,
// selectively retrieving useful information instead of blindly accumulating.
//
// WHY: Standard residual connections cause PreNorm dilution -- hidden states
// grow continuously with depth, burying early layer signals. AttnRes solves
// this by letting each layer learn exactly which previous layers matter.
//
// BLOCK AttnRes: Groups layers into blocks (default 8), attends over block
// summaries instead of individual layers. Reduces memory from O(L^2) to O(B^2)
// where B is the number of blocks. <2% inference overhead.
//
// RESULTS (from paper):
//   - Matches baseline trained with 1.25x more compute
//   - +7.5 on GPQA-Diamond (reasoning)
//   - +3.6 on Minerva Math
//   - +significant on HumanEval (code generation)
package native

import (
	"math"
)

// AttnResConfig controls the Attention Residual mechanism.
type AttnResConfig struct {
	Enabled    bool // When false, falls back to standard residual Add
	BlockSize  int  // Layers per block for Block AttnRes (0 = full AttnRes)
	NumBlocks  int  // Computed from numLayers / BlockSize
}

// AttnResState tracks layer outputs during a forward pass for depth-wise attention.
// One instance per inference call (not shared across requests).
type AttnResState struct {
	// Per-layer output history for full AttnRes.
	layerOutputs []*Tensor // outputs[i] = output of layer i

	// Block-level summaries for Block AttnRes.
	blockSums    []*Tensor // accumulated sum within current block
	blockOutputs []*Tensor // finalized block summaries

	config    AttnResConfig
	hiddenDim int
	curLayer  int
}

// NewAttnResState creates a new state tracker for one forward pass.
func NewAttnResState(config AttnResConfig, numLayers, hiddenDim int) *AttnResState {
	s := &AttnResState{
		config:    config,
		hiddenDim: hiddenDim,
	}

	if config.BlockSize > 0 {
		// Block AttnRes mode.
		numBlocks := (numLayers + config.BlockSize - 1) / config.BlockSize
		s.blockSums = make([]*Tensor, numBlocks)
		s.blockOutputs = make([]*Tensor, 0, numBlocks)
	} else {
		// Full AttnRes mode.
		s.layerOutputs = make([]*Tensor, 0, numLayers)
	}

	return s
}

// AttnResidual computes the attention-weighted residual for the current layer.
//
// Instead of: x = x + layer_output (standard residual)
// We compute:  x = softmax_attention(query, previous_outputs) + layer_output
//
// Parameters:
//   - layerOutput: the current layer's output (post-attention or post-FFN)
//   - queryVec: this layer's learned pseudo-query vector [hiddenDim]
//   - seqLen: current sequence length (for batched tensors)
//
// The pseudo-query attends over all previous layer outputs to produce a
// content-aware weighted combination that replaces the uniform accumulation.
func (s *AttnResState) AttnResidual(layerOutput *Tensor, queryVec *Tensor) *Tensor {
	if !s.config.Enabled || queryVec == nil {
		// Fallback: standard residual (no AttnRes).
		// Caller is responsible for adding x + layerOutput.
		return nil
	}

	if s.config.BlockSize > 0 {
		return s.blockAttnResidual(layerOutput, queryVec)
	}
	return s.fullAttnResidual(layerOutput, queryVec)
}

// RecordLayerOutput stores the current layer's output for future attention.
func (s *AttnResState) RecordLayerOutput(output *Tensor) {
	if s.config.BlockSize > 0 {
		s.recordBlockOutput(output)
	} else {
		s.layerOutputs = append(s.layerOutputs, output)
	}
	s.curLayer++
}

// ---------- Full AttnRes ----------

// fullAttnResidual applies attention over all individual previous layer outputs.
func (s *AttnResState) fullAttnResidual(layerOutput *Tensor, queryVec *Tensor) *Tensor {
	numPrev := len(s.layerOutputs)
	if numPrev == 0 {
		// First layer: no previous outputs to attend over.
		return layerOutput
	}

	dim := s.hiddenDim
	isVec := len(layerOutput.Shape) == 1 || (len(layerOutput.Shape) == 2 && layerOutput.Shape[0] == 1)

	if isVec {
		return s.attnOverPrevious(queryVec.Data, s.layerOutputs, layerOutput, dim)
	}

	// Batched: apply attention per row.
	seqLen := layerOutput.Shape[0]
	result := NewPooledTensor(layerOutput.Shape)

	for row := 0; row < seqLen; row++ {
		rowStart := row * dim
		rowEnd := rowStart + dim

		// Extract row slices from previous outputs.
		prevRows := make([]*Tensor, numPrev)
		for i, prev := range s.layerOutputs {
			rowTensor := &Tensor{Data: prev.Data[rowStart:rowEnd], Shape: []int{dim}}
			prevRows[i] = rowTensor
		}

		currentRow := &Tensor{Data: layerOutput.Data[rowStart:rowEnd], Shape: []int{dim}}
		combined := s.attnOverPrevious(queryVec.Data, prevRows, currentRow, dim)
		copy(result.Data[rowStart:rowEnd], combined.Data)
	}

	return result
}

// ---------- Block AttnRes ----------

// recordBlockOutput accumulates layer output into the current block sum.
func (s *AttnResState) recordBlockOutput(output *Tensor) {
	blockIdx := s.curLayer / s.config.BlockSize

	if s.blockSums[blockIdx] == nil {
		// Start a new block: clone the output.
		s.blockSums[blockIdx] = &Tensor{
			Data:  make([]float32, len(output.Data)),
			Shape: append([]int(nil), output.Shape...),
		}
		copy(s.blockSums[blockIdx].Data, output.Data)
	} else {
		// Accumulate into existing block sum.
		for i := range output.Data {
			s.blockSums[blockIdx].Data[i] += output.Data[i]
		}
	}

	// If this is the last layer in the block, finalize the block summary.
	if (s.curLayer+1)%s.config.BlockSize == 0 || s.curLayer == cap(s.layerOutputs)-1 {
		s.blockOutputs = append(s.blockOutputs, s.blockSums[blockIdx])
	}
}

// blockAttnResidual applies attention over block-level summaries.
func (s *AttnResState) blockAttnResidual(layerOutput *Tensor, queryVec *Tensor) *Tensor {
	if len(s.blockOutputs) == 0 {
		return layerOutput
	}

	dim := s.hiddenDim
	isVec := len(layerOutput.Shape) == 1 || (len(layerOutput.Shape) == 2 && layerOutput.Shape[0] == 1)

	if isVec {
		return s.attnOverPrevious(queryVec.Data, s.blockOutputs, layerOutput, dim)
	}

	// Batched: apply per row.
	seqLen := layerOutput.Shape[0]
	result := NewPooledTensor(layerOutput.Shape)

	for row := 0; row < seqLen; row++ {
		rowStart := row * dim
		rowEnd := rowStart + dim

		prevRows := make([]*Tensor, len(s.blockOutputs))
		for i, block := range s.blockOutputs {
			rowTensor := &Tensor{Data: block.Data[rowStart:rowEnd], Shape: []int{dim}}
			prevRows[i] = rowTensor
		}

		currentRow := &Tensor{Data: layerOutput.Data[rowStart:rowEnd], Shape: []int{dim}}
		combined := s.attnOverPrevious(queryVec.Data, prevRows, currentRow, dim)
		copy(result.Data[rowStart:rowEnd], combined.Data)
	}

	return result
}

// ---------- Core Attention Computation ----------

// attnOverPrevious computes softmax attention of a query over previous layer outputs,
// then adds the current layer output.
//
// scores[i] = dot(query, prevOutputs[i]) / sqrt(dim)
// weights = softmax(scores)
// result = sum(weights[i] * prevOutputs[i]) + currentOutput
func (s *AttnResState) attnOverPrevious(query []float32, prevOutputs []*Tensor, currentOutput *Tensor, dim int) *Tensor {
	n := len(prevOutputs)
	scale := float32(1.0 / math.Sqrt(float64(dim)))

	// Compute attention scores.
	scores := make([]float32, n+1) // +1 for the current layer output itself
	for i, prev := range prevOutputs {
		scores[i] = DotProduct(query, prev.Data[:dim]) * scale
	}
	// The current output also participates in the attention.
	scores[n] = DotProduct(query, currentOutput.Data[:dim]) * scale

	// Softmax over scores.
	maxScore := scores[0]
	for _, s := range scores[1:] {
		if s > maxScore {
			maxScore = s
		}
	}
	var sumExp float32
	for i := range scores {
		scores[i] = float32(math.Exp(float64(scores[i] - maxScore)))
		sumExp += scores[i]
	}
	invSum := float32(1.0) / sumExp
	for i := range scores {
		scores[i] *= invSum
	}

	// Weighted combination.
	result := NewPooledTensor([]int{dim})
	for i, prev := range prevOutputs {
		w := scores[i]
		for d := 0; d < dim; d++ {
			result.Data[d] += w * prev.Data[d]
		}
	}
	// Add current output's weighted contribution.
	w := scores[n]
	for d := 0; d < dim; d++ {
		result.Data[d] += w * currentOutput.Data[d]
	}

	return result
}

// DefaultAttnResConfig returns the recommended Block AttnRes configuration.
// BlockSize of 4-8 gives best memory/performance tradeoff per the paper.
func DefaultAttnResConfig(numLayers int) AttnResConfig {
	blockSize := 4
	if numLayers > 32 {
		blockSize = 8
	}
	if numLayers <= 8 {
		blockSize = 0 // Use full AttnRes for small models
	}

	return AttnResConfig{
		Enabled:   true,
		BlockSize: blockSize,
		NumBlocks: (numLayers + blockSize - 1) / blockSize,
	}
}
