// mtp_decoder.go implements Multi-Token Prediction (MTP) draft generation.
//
// WHAT: MTP (introduced in DeepSeek V3) is an alternative to Medusa for
// speculative decoding. Instead of attaching multiple independent MLP heads
// to the base model, MTP attaches a single, shared lightweight transformer
// block that is applied sequentially to predict depth.
//
// HOW:
//   Base Model -> h_0 -> outputs token_1
//   MTP Block (takes h_0, emb(token_1)) -> h_1 -> outputs token_2
//   MTP Block (takes h_1, emb(token_2)) -> h_2 -> outputs token_3
//
// In training, the MTP block learns to predict depth sequentially.
// In inference, we run the MTP block K times sequentially to generate a
// K-token draft, which is then verified by the massive base model in ONE
// forward pass.
//
// WHY: Medusa requires K separate heads (inflexible, harder to train).
// MTP uses one shared block, scales to any depth dynamically, and uses
// the causal chain (h_i depends on token_i) making the draft much more
// accurate than Medusa's independent predictions.
package native

import (
	"math"
)

// MTPConfig configures the Multi-Token Prediction draft generator.
type MTPConfig struct {
	Depth         int     // Number of future tokens to predict (default: 3)
	HiddenDim     int     // Base model hidden dimension
	MTPHiddenDim  int     // Usually same as HiddenDim
	VocabSize     int     
}

// DefaultMTPConfig returns standard settings for DeepSeek V3 style MTP.
func DefaultMTPConfig(hidden, vocab int) MTPConfig {
	return MTPConfig{
		Depth:        3, // Typically depth 1 (EAGLE) or 3 (MTP)
		HiddenDim:    hidden,
		MTPHiddenDim: hidden,
		VocabSize:    vocab,
	}
}

// MTPBlock is a lightweight transformer layer + linear projection.
type MTPBlock struct {
	// 1. Layer Norm for hidden state
	NormWeight []float32
	
	// 2. Token Embedding projection (if different from base model)
	// Often shares the base model's embedding matrix to save memory.
	EmbProjWeight []float32 // [HiddenDim, MTPHiddenDim]

	// 3. Shared Linear combination: W_h * h_i + W_e * E(token_i)
	HiddenProjWeight []float32 // [HiddenDim, MTPHiddenDim]
	
	// 4. Output head (shared with base model usually, or dedicated)
	OutputWeight []float32 // [MTPHiddenDim, VocabSize]
}

// MTPDecoder orchestrates generating draft sequences.
type MTPDecoder struct {
	config MTPConfig
	block  *MTPBlock
	
	// Pre-allocated buffers for sequential predictions
	hBuffer   [][]float32
	logitsBuf []float64
}

// NewMTPDecoder initializes the draft generator.
func NewMTPDecoder(config MTPConfig) *MTPDecoder {
	decoder := &MTPDecoder{
		config:    config,
		hBuffer:   make([][]float32, config.Depth+1),
		logitsBuf: make([]float64, config.VocabSize),
	}
	
	for i := range decoder.hBuffer {
		decoder.hBuffer[i] = make([]float32, config.MTPHiddenDim)
	}
	return decoder
}

// SetBlock assigns the weights for the MTP module.
func (md *MTPDecoder) SetBlock(block *MTPBlock) {
	md.block = block
}

// GenerateDraft produces a sequence of candidate tokens.
// baseHidden: The final hidden state from the base model for the current token.
// baseToken: The actual token predicted by the base model.
// getEmbeddingFn: Callback to fetch the token embedding for a given token ID.
//
// Returns an array of Draft tokens of length `config.Depth`.
func (md *MTPDecoder) GenerateDraft(
	baseHidden []float32,
	baseToken int32,
	getEmbeddingFn func(tokenID int32) []float32,
) []int32 {
	
	draftTokens := make([]int32, md.config.Depth)
	
	// h_0 is the base model's hidden state.
	copy(md.hBuffer[0], baseHidden)
	
	currentToken := baseToken

	// Sequentially unfold the MTP block.
	for d := 0; d < md.config.Depth; d++ {
		// 1. Get embedding for the token we just predicted (or base token).
		emb := getEmbeddingFn(currentToken)

		// 2. Apply MTP Block: h_{d+1} = MTP(h_d, emb)
		// Usually this is a concatenation and linear projection, or a lightweight attention.
		// DeepSeek V3 approach: RMSNorm(h_d) + Linear(emb).
		
		hD := md.hBuffer[d]
		hNext := md.hBuffer[d+1]
		
		// RMSNorm on h_d
		normed := rmsNormFP32(hD, md.block.NormWeight)
		
		// Project hidden state and embedding, sum them.
		// hNext = hD_proj + emb_proj
		for i := 0; i < md.config.MTPHiddenDim; i++ {
			var sumH float64
			var sumE float64
			for j := 0; j < md.config.HiddenDim; j++ {
				sumH += float64(normed[j]) * float64(md.block.HiddenProjWeight[j*md.config.MTPHiddenDim + i])
				sumE += float64(emb[j]) * float64(md.block.EmbProjWeight[j*md.config.MTPHiddenDim + i])
			}
			hNext[i] = float32(sumH + sumE)
		}
		
		// 3. Output Projection -> logits
		for i := 0; i < md.config.VocabSize; i++ {
			var sum float64
			for j := 0; j < md.config.MTPHiddenDim; j++ {
				sum += float64(hNext[j]) * float64(md.block.OutputWeight[j*md.config.VocabSize + i])
			}
			md.logitsBuf[i] = sum
		}
		
		// 4. Greedy sample the draft token.
		// In a real implementation, you might sample Top-P to create a tree like Medusa.
		best := 0
		for i := 1; i < md.config.VocabSize; i++ {
			if md.logitsBuf[i] > md.logitsBuf[best] {
				best = i
			}
		}
		
		draftToken := int32(best)
		draftTokens[d] = draftToken
		
		// Draft token becomes input to the next MTP depth step.
		currentToken = draftToken
	}
	
	return draftTokens
}

// rmsNormFP32 is a simple un-optimized RMSNorm for the draft model.
func rmsNormFP32(x, w []float32) []float32 {
	out := make([]float32, len(x))
	var sumSq float64
	for _, v := range x {
		sumSq += float64(v * v)
	}
	
	meanSq := sumSq / float64(len(x))
	rsqrt := 1.0 / math.Sqrt(meanSq + 1e-5)
	
	for i, v := range x {
		out[i] = float32(float64(v) * rsqrt) * w[i]
	}
	return out
}

// calculateEntropy is exported for reasoning_optimizer usage
func calculateEntropy(logits []float64) float64 {
	// Very simple entropy approx for logits
	maxLogit := logits[0]
	for _, l := range logits[1:] {
		if l > maxLogit {
			maxLogit = l
		}
	}
	
	var sumExp float64
	for _, l := range logits {
		sumExp += math.Exp(l - maxLogit)
	}
	
	var entropy float64
	for _, l := range logits {
		p := math.Exp(l - maxLogit) / sumExp
		if p > 1e-10 {
			entropy -= p * math.Log(p)
		}
	}
	return entropy
}
