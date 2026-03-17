// awq_quantize.go implements Activation-Aware Weight Quantization.
//
// WHAT: AWQ (Lin et al., 2023) is a weight-only quantization method that
// identifies which weight channels are most important based on activation
// magnitudes, then protects those channels during quantization.
//
// HOW:
//   1. Calibrate: run a small dataset through the model, record per-channel
//      activation magnitudes.
//   2. Find salient channels: channels with high activation values have
//      disproportionate impact on output quality.
//   3. Scale salient channels UP before quantization (less quantization error
//      on important values).
//   4. Scale the corresponding activations DOWN to compensate (math stays same).
//
// WHY: Standard quantization treats all weights equally. AWQ recognizes
// that ~1% of weight channels matter far more than the rest. Protecting
// those channels gives INT4 quality approaching FP16, whereas naive INT4
// degrades significantly.
//
// GAIN: INT4 with near-FP16 quality. 4x memory reduction, 2-3x inference
// speedup. Better than GPTQ for models under 13B parameters.
package native

import (
	"math"
	"sort"
)

// AWQConfig configures activation-aware quantization.
type AWQConfig struct {
	Bits          int     // Target bit width (default: 4)
	GroupSize     int     // Quantization group size (default: 128)
	ScaleFactor   float32 // How much to scale salient channels (default: 2.0)
	SalientPct    float32 // Top percentage of channels to protect (default: 0.01)
	ZeroPoint     bool    // Use asymmetric quantization with zero point
}

// DefaultAWQConfig returns standard AWQ settings.
func DefaultAWQConfig() AWQConfig {
	return AWQConfig{
		Bits:        4,
		GroupSize:   128,
		ScaleFactor: 2.0,
		SalientPct:  0.01,
		ZeroPoint:   true,
	}
}

// AWQCalibrationData holds activation statistics collected during calibration.
type AWQCalibrationData struct {
	// Per-channel activation magnitudes [numChannels].
	ChannelMagnitudes []float32

	// Per-channel importance scores (derived from magnitudes).
	ImportanceScores []float32

	// Salient channel indices.
	SalientChannels []int

	// Number of calibration samples processed.
	NumSamples int
}

// NewAWQCalibration creates calibration data storage.
func NewAWQCalibration(numChannels int) *AWQCalibrationData {
	return &AWQCalibrationData{
		ChannelMagnitudes: make([]float32, numChannels),
		ImportanceScores:  make([]float32, numChannels),
	}
}

// RecordActivation accumulates activation statistics for calibration.
// activations: [batchSize, numChannels] - one batch of activations.
func (c *AWQCalibrationData) RecordActivation(activations []float32, batchSize, numChannels int) {
	for b := 0; b < batchSize; b++ {
		for ch := 0; ch < numChannels; ch++ {
			val := activations[b*numChannels+ch]
			absVal := float32(math.Abs(float64(val)))
			c.ChannelMagnitudes[ch] += absVal
		}
	}
	c.NumSamples += batchSize
}

// ComputeImportance calculates per-channel importance from accumulated statistics.
func (c *AWQCalibrationData) ComputeImportance(config AWQConfig) {
	numChannels := len(c.ChannelMagnitudes)
	if c.NumSamples == 0 {
		return
	}

	// Average magnitudes.
	invN := float32(1.0 / float64(c.NumSamples))
	for i := range c.ChannelMagnitudes {
		c.ChannelMagnitudes[i] *= invN
	}

	// Importance = activation magnitude (higher = more important to protect).
	copy(c.ImportanceScores, c.ChannelMagnitudes)

	// Find salient channels (top SalientPct by importance).
	type chanScore struct {
		idx   int
		score float32
	}
	scored := make([]chanScore, numChannels)
	for i := range scored {
		scored[i] = chanScore{i, c.ImportanceScores[i]}
	}
	sort.Slice(scored, func(i, j int) bool {
		return scored[i].score > scored[j].score // Descending.
	})

	numSalient := int(float32(numChannels) * config.SalientPct)
	if numSalient < 1 {
		numSalient = 1
	}

	c.SalientChannels = make([]int, numSalient)
	for i := 0; i < numSalient; i++ {
		c.SalientChannels[i] = scored[i].idx
	}
}

// AWQQuantizedWeight holds the quantized weight matrix.
type AWQQuantizedWeight struct {
	// Quantized data (packed INT4 or INT8).
	Data []int8

	// Per-group scales.
	Scales []float32

	// Per-group zero points (if asymmetric).
	ZeroPoints []float32

	// Channel scaling factors applied before quantization.
	ChannelScales []float32

	// Original dimensions.
	Rows, Cols int
	GroupSize  int
	Bits       int
}

// QuantizeAWQ applies activation-aware quantization to a weight matrix.
// weight: [rows, cols], calibData: activation statistics from calibration.
func QuantizeAWQ(weight []float32, rows, cols int, calibData *AWQCalibrationData, config AWQConfig) *AWQQuantizedWeight {
	if calibData.SalientChannels == nil {
		calibData.ComputeImportance(config)
	}

	// Build channel scale map: salient channels get scaled up.
	channelScales := make([]float32, cols)
	for i := range channelScales {
		channelScales[i] = 1.0
	}
	for _, ch := range calibData.SalientChannels {
		if ch < cols {
			channelScales[ch] = config.ScaleFactor
		}
	}

	// Apply channel scaling to weights.
	scaledWeight := make([]float32, rows*cols)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			scaledWeight[i*cols+j] = weight[i*cols+j] * channelScales[j]
		}
	}

	// Group quantization.
	groupSize := config.GroupSize
	numGroups := (rows*cols + groupSize - 1) / groupSize
	quantized := make([]int8, rows*cols)
	scales := make([]float32, numGroups)
	zeroPoints := make([]float32, numGroups)

	maxQ := float32(math.Pow(2, float64(config.Bits-1)) - 1)
	minQ := -maxQ
	if config.ZeroPoint {
		minQ = 0
		maxQ = float32(math.Pow(2, float64(config.Bits)) - 1)
	}

	for g := 0; g < numGroups; g++ {
		start := g * groupSize
		end := start + groupSize
		if end > len(scaledWeight) {
			end = len(scaledWeight)
		}

		// Find min/max in group.
		gMin, gMax := scaledWeight[start], scaledWeight[start]
		for _, v := range scaledWeight[start:end] {
			if v < gMin {
				gMin = v
			}
			if v > gMax {
				gMax = v
			}
		}

		// Compute scale.
		var scale, zp float32
		if config.ZeroPoint {
			dataRange := gMax - gMin
			if dataRange < 1e-10 {
				dataRange = 1e-10
			}
			scale = dataRange / float32(maxQ-minQ)
			zp = float32(minQ) - gMin/scale
		} else {
			absMax := float32(math.Max(math.Abs(float64(gMin)), math.Abs(float64(gMax))))
			if absMax < 1e-10 {
				absMax = 1e-10
			}
			scale = absMax / maxQ
		}

		scales[g] = scale
		zeroPoints[g] = zp

		// Quantize.
		for i := start; i < end; i++ {
			q := math.Round(float64(scaledWeight[i]/scale)) + float64(zp)
			q = math.Max(float64(minQ), math.Min(float64(maxQ), q))
			quantized[i] = int8(q)
		}
	}

	return &AWQQuantizedWeight{
		Data:          quantized,
		Scales:        scales,
		ZeroPoints:    zeroPoints,
		ChannelScales: channelScales,
		Rows:          rows,
		Cols:          cols,
		GroupSize:     groupSize,
		Bits:          config.Bits,
	}
}

// Dequantize reconstructs the weight matrix from quantized data.
// Used during inference for matmul.
func (q *AWQQuantizedWeight) Dequantize() []float32 {
	result := make([]float32, q.Rows*q.Cols)

	for g := 0; g < len(q.Scales); g++ {
		start := g * q.GroupSize
		end := start + q.GroupSize
		if end > len(result) {
			end = len(result)
		}

		scale := q.Scales[g]
		zp := q.ZeroPoints[g]

		for i := start; i < end; i++ {
			// Dequantize.
			val := (float32(q.Data[i]) - zp) * scale

			// Undo channel scaling.
			col := i % q.Cols
			if q.ChannelScales[col] > 0 {
				val /= q.ChannelScales[col]
			}
			result[i] = val
		}
	}

	return result
}

// CompressionRatio returns the memory savings.
func (q *AWQQuantizedWeight) CompressionRatio() float64 {
	originalBytes := float64(q.Rows*q.Cols) * 4 // FP32
	quantizedBytes := float64(len(q.Data))       // INT8/INT4
	quantizedBytes += float64(len(q.Scales)) * 4 // Scales
	quantizedBytes += float64(len(q.ZeroPoints)) * 4
	return originalBytes / quantizedBytes
}
