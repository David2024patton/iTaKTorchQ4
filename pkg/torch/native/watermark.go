// watermark.go implements invisible text watermarking for generated output.
//
// WHAT: Embeds a detectable but invisible signature in model output by
// biasing token selection during generation. This allows identifying
// AI-generated text without affecting readability.
//
// METHOD: Uses the "green/red list" approach (Kirchenbauer et al., 2023).
// At each generation step, tokens are split into "green" (favored) and
// "red" (disfavored) lists based on a hash of the previous token.
// Green tokens get a small logit boost. The pattern is statistically
// detectable but invisible to humans.
//
// DETECTION: Given suspected text, hash each token pair and count how many
// tokens fell in the "green" list. Natural text has ~50% green tokens;
// watermarked text has significantly more.
package native

import (
	"crypto/sha256"
	"encoding/binary"
	"math"
)

// WatermarkConfig controls watermark embedding.
type WatermarkConfig struct {
	Enabled    bool
	Delta      float32 // Logit boost for green tokens (default: 2.0)
	GreenRatio float32 // Fraction of vocab in green list (default: 0.5)
	Key        []byte  // Secret key for hash function
}

// DefaultWatermarkConfig returns standard settings.
func DefaultWatermarkConfig() WatermarkConfig {
	return WatermarkConfig{
		Enabled:    true,
		Delta:      2.0,
		GreenRatio: 0.5,
		Key:        []byte("itak-torch-watermark-2026"),
	}
}

// WatermarkProcessor implements LogitProcessor for watermark embedding.
type WatermarkProcessor struct {
	config    WatermarkConfig
	prevToken int
}

// NewWatermarkProcessor creates a watermark-embedding logit processor.
func NewWatermarkProcessor(config WatermarkConfig) *WatermarkProcessor {
	return &WatermarkProcessor{config: config, prevToken: -1}
}

// SetPrevToken updates the previous token for hash computation.
func (w *WatermarkProcessor) SetPrevToken(token int) {
	w.prevToken = token
}

// Process applies the watermark bias to logits.
func (w *WatermarkProcessor) Process(logits []float32) {
	if !w.config.Enabled || w.prevToken < 0 {
		return
	}

	greenList := w.computeGreenList(w.prevToken, len(logits))

	for _, idx := range greenList {
		if idx < len(logits) {
			logits[idx] += w.config.Delta
		}
	}
}

// computeGreenList determines which tokens are "green" based on prev token hash.
func (w *WatermarkProcessor) computeGreenList(prevToken, vocabSize int) []int {
	// Hash: SHA256(key || prevToken)
	h := sha256.New()
	h.Write(w.config.Key)
	var buf [4]byte
	binary.BigEndian.PutUint32(buf[:], uint32(prevToken))
	h.Write(buf[:])
	hash := h.Sum(nil)

	// Use hash to seed a simple RNG for selecting green tokens.
	seed := binary.BigEndian.Uint64(hash[:8])

	greenCount := int(float32(vocabSize) * w.config.GreenRatio)
	greenList := make([]int, 0, greenCount)

	// Fisher-Yates-style selection using hash-based RNG.
	rng := seed
	selected := make(map[int]bool, greenCount)
	for len(greenList) < greenCount {
		rng = rng*6364136223846793005 + 1442695040888963407 // LCG
		idx := int(rng>>33) % vocabSize
		if !selected[idx] {
			selected[idx] = true
			greenList = append(greenList, idx)
		}
	}

	return greenList
}

// WatermarkDetector checks if text was watermarked.
type WatermarkDetector struct {
	config WatermarkConfig
}

// NewWatermarkDetector creates a detector with the same key used for embedding.
func NewWatermarkDetector(config WatermarkConfig) *WatermarkDetector {
	return &WatermarkDetector{config: config}
}

// DetectionResult holds watermark detection metrics.
type DetectionResult struct {
	IsWatermarked bool    `json:"is_watermarked"`
	GreenFraction float64 `json:"green_fraction"` // Fraction of tokens in green list
	ZScore        float64 `json:"z_score"`         // Statistical significance
	TokensChecked int     `json:"tokens_checked"`
	GreenCount    int     `json:"green_count"`
}

// Detect checks if a token sequence was watermarked.
func (d *WatermarkDetector) Detect(tokens []int, vocabSize int) DetectionResult {
	if len(tokens) < 2 {
		return DetectionResult{TokensChecked: len(tokens)}
	}

	greenCount := 0
	proc := NewWatermarkProcessor(d.config)

	for i := 1; i < len(tokens); i++ {
		greenList := proc.computeGreenList(tokens[i-1], vocabSize)
		greenSet := make(map[int]bool, len(greenList))
		for _, g := range greenList {
			greenSet[g] = true
		}
		if greenSet[tokens[i]] {
			greenCount++
		}
	}

	checked := len(tokens) - 1
	greenFrac := float64(greenCount) / float64(checked)

	// Z-score: how many standard deviations above expected (0.5).
	// Under null (no watermark), each token has GreenRatio probability of being green.
	expectedFrac := float64(d.config.GreenRatio)
	variance := expectedFrac * (1 - expectedFrac) / float64(checked)
	zScore := (greenFrac - expectedFrac) / math.Sqrt(variance)

	return DetectionResult{
		IsWatermarked: zScore > 4.0, // p < 0.00003
		GreenFraction: greenFrac,
		ZScore:        zScore,
		TokensChecked: checked,
		GreenCount:    greenCount,
	}
}
