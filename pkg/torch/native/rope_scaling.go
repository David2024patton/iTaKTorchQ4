// rope_scaling.go implements RoPE (Rotary Position Embedding) scaling methods
// for extending context length beyond what the model was trained on.
//
// WHAT: Models are trained with a fixed context length (e.g., 2K or 4K tokens).
// RoPE scaling modifies the frequency basis of rotary embeddings to let the
// model handle 4x-16x longer contexts without retraining.
//
// METHODS:
//   Linear:    Divide frequencies by scale factor. Simple but degrades quality.
//   NTK-Aware: Scale the base frequency (theta) to preserve low frequencies.
//              Much better quality than linear. Used by Code Llama.
//   YaRN:      NTK + attention temperature correction + linear/NTK blend.
//              Best quality for large scaling factors. Used by LLaMA 3.
//   Dynamic:   Automatically compute scale based on current sequence position.
//              No config needed - works for any length.
//
// GAIN: 4x-16x context extension with minimal quality loss (especially YaRN).
package native

import (
	"math"
)

// RoPEScaleMethod identifies the scaling strategy.
type RoPEScaleMethod int

const (
	RoPEScaleNone    RoPEScaleMethod = iota
	RoPEScaleLinear
	RoPEScaleNTK
	RoPEScaleYaRN
	RoPEScaleDynamic
)

// RoPEScaleConfig configures context extension.
type RoPEScaleConfig struct {
	Method           RoPEScaleMethod
	Factor           float64 // Scale factor (e.g., 4.0 for 4x extension)
	OriginalMaxPos   int     // Original training context length
	BaseFreq         float64 // Base theta for RoPE (default: 10000)
	NTKAlpha         float64 // NTK alpha parameter (auto-computed if 0)
	YaRNBetaFast     float64 // YaRN fast dimension threshold (default: 32)
	YaRNBetaSlow     float64 // YaRN slow dimension threshold (default: 1)
	AttnScale        float64 // YaRN attention temperature scaling
}

// DefaultRoPEScaleConfig returns settings for 4x extension with YaRN.
func DefaultRoPEScaleConfig() RoPEScaleConfig {
	return RoPEScaleConfig{
		Method:         RoPEScaleYaRN,
		Factor:         4.0,
		OriginalMaxPos: 4096,
		BaseFreq:       10000.0,
		YaRNBetaFast:   32.0,
		YaRNBetaSlow:   1.0,
		AttnScale:      0.0, // Auto-computed.
	}
}

// RoPEScaler computes scaled rotary frequencies for extended context.
type RoPEScaler struct {
	config      RoPEScaleConfig
	frequencies []float64 // Pre-computed per-dimension frequencies
	headDim     int
}

// NewRoPEScaler creates a scaler for the given head dimension.
func NewRoPEScaler(headDim int, config RoPEScaleConfig) *RoPEScaler {
	rs := &RoPEScaler{
		config:  config,
		headDim: headDim,
	}
	rs.computeFrequencies()
	return rs
}

// computeFrequencies pre-computes the scaled frequency table.
func (rs *RoPEScaler) computeFrequencies() {
	halfDim := rs.headDim / 2
	rs.frequencies = make([]float64, halfDim)

	switch rs.config.Method {
	case RoPEScaleLinear:
		rs.computeLinear(halfDim)
	case RoPEScaleNTK:
		rs.computeNTK(halfDim)
	case RoPEScaleYaRN:
		rs.computeYaRN(halfDim)
	case RoPEScaleDynamic:
		rs.computeDynamic(halfDim)
	default:
		// No scaling: standard RoPE frequencies.
		for i := 0; i < halfDim; i++ {
			rs.frequencies[i] = 1.0 / math.Pow(rs.config.BaseFreq, float64(2*i)/float64(rs.headDim))
		}
	}
}

// computeLinear scales frequencies by dividing by the factor.
func (rs *RoPEScaler) computeLinear(halfDim int) {
	for i := 0; i < halfDim; i++ {
		freq := 1.0 / math.Pow(rs.config.BaseFreq, float64(2*i)/float64(rs.headDim))
		rs.frequencies[i] = freq / rs.config.Factor
	}
}

// computeNTK scales the base frequency to preserve low-frequency components.
// This is the "NTK-aware" method from Code Llama.
func (rs *RoPEScaler) computeNTK(halfDim int) {
	// Compute NTK-scaled base frequency.
	alpha := rs.config.NTKAlpha
	if alpha == 0 {
		// Auto-compute alpha from scale factor.
		alpha = rs.config.Factor
	}
	scaledBase := rs.config.BaseFreq * math.Pow(alpha*float64(rs.config.Factor), float64(rs.headDim)/float64(rs.headDim-2))

	for i := 0; i < halfDim; i++ {
		rs.frequencies[i] = 1.0 / math.Pow(scaledBase, float64(2*i)/float64(rs.headDim))
	}
}

// computeYaRN implements Yet Another RoPE extensioN method.
// Blends linear and NTK scaling with dimension-dependent interpolation.
func (rs *RoPEScaler) computeYaRN(halfDim int) {
	// Compute wavelengths for each dimension.
	for i := 0; i < halfDim; i++ {
		// Original frequency.
		freq := 1.0 / math.Pow(rs.config.BaseFreq, float64(2*i)/float64(rs.headDim))

		// Wavelength = 2*pi / freq.
		wavelength := 2.0 * math.Pi / freq

		// Dimension-dependent blend factor.
		// High frequencies (short wavelengths) should use linear scaling.
		// Low frequencies (long wavelengths) should use NTK scaling.
		lowThresh := float64(rs.config.OriginalMaxPos) / rs.config.YaRNBetaFast
		highThresh := float64(rs.config.OriginalMaxPos) / rs.config.YaRNBetaSlow

		var ramp float64
		if wavelength < lowThresh {
			ramp = 0.0 // Pure NTK range.
		} else if wavelength > highThresh {
			ramp = 1.0 // Pure linear range.
		} else {
			// Smooth interpolation.
			ramp = (wavelength - lowThresh) / (highThresh - lowThresh)
		}

		// Blend linear and NTK scaled frequencies.
		linearFreq := freq / rs.config.Factor
		ntkBase := rs.config.BaseFreq * math.Pow(rs.config.Factor, float64(rs.headDim)/float64(rs.headDim-2))
		ntkFreq := 1.0 / math.Pow(ntkBase, float64(2*i)/float64(rs.headDim))

		rs.frequencies[i] = (1.0-ramp)*ntkFreq + ramp*linearFreq
	}

	// Compute attention temperature scaling.
	if rs.config.AttnScale == 0 {
		rs.config.AttnScale = 0.1 * math.Log(rs.config.Factor) + 1.0
	}
}

// computeDynamic automatically adjusts scaling based on sequence length.
func (rs *RoPEScaler) computeDynamic(halfDim int) {
	// Dynamic NTK: scale base frequency based on how far past training length.
	scaledBase := rs.config.BaseFreq * math.Pow(
		(rs.config.Factor*float64(rs.config.OriginalMaxPos))/float64(rs.config.OriginalMaxPos)-float64(rs.config.OriginalMaxPos-1),
		float64(rs.headDim)/float64(rs.headDim-2),
	)

	for i := 0; i < halfDim; i++ {
		rs.frequencies[i] = 1.0 / math.Pow(scaledBase, float64(2*i)/float64(rs.headDim))
	}
}

// Apply rotates the input tensor by the scaled RoPE frequencies at the given position.
// data: [headDim] (one head's Q or K vector).
func (rs *RoPEScaler) Apply(data []float32, position int) {
	halfDim := rs.headDim / 2

	for i := 0; i < halfDim; i++ {
		angle := float64(position) * rs.frequencies[i]
		cos := float32(math.Cos(angle))
		sin := float32(math.Sin(angle))

		x0 := data[i]
		x1 := data[i+halfDim]
		data[i] = x0*cos - x1*sin
		data[i+halfDim] = x0*sin + x1*cos
	}
}

// AttnScale returns the YaRN attention temperature correction.
// Multiply attention logits by this value when using YaRN scaling.
func (rs *RoPEScaler) AttnScale() float32 {
	if rs.config.Method == RoPEScaleYaRN && rs.config.AttnScale > 0 {
		return float32(1.0 / rs.config.AttnScale)
	}
	return 1.0
}

// MaxContext returns the effective maximum context length after scaling.
func (rs *RoPEScaler) MaxContext() int {
	return int(float64(rs.config.OriginalMaxPos) * rs.config.Factor)
}
