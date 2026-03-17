// mixed_precision.go implements FP16/BF16 mixed precision for faster computation.
//
// WHAT: Mixed precision uses lower-precision formats (FP16 or BF16) for
// matrix multiplications while keeping master weights in FP32. This
// provides 2x throughput on hardware with native half-precision support
// (GPUs, Apple AMX) while maintaining FP32 accuracy via loss scaling.
//
// FLOW:
//   1. Convert weights to FP16 for forward/backward passes
//   2. Compute gradients in FP16
//   3. Scale gradients back to FP32
//   4. Update master weights in FP32
//   5. Dynamic loss scaling prevents underflow
package native

import (
	"fmt"
	"math"
)

// PrecisionMode specifies the compute precision.
type PrecisionMode int

const (
	PrecisionFP32 PrecisionMode = iota
	PrecisionFP16
	PrecisionBF16
)

func (p PrecisionMode) String() string {
	switch p {
	case PrecisionFP16:
		return "FP16"
	case PrecisionBF16:
		return "BF16"
	default:
		return "FP32"
	}
}

// MixedPrecisionConfig controls mixed precision behavior.
type MixedPrecisionConfig struct {
	Mode             PrecisionMode
	DynamicLossScale bool    // Auto-adjust loss scale to prevent underflow
	InitialScale     float32 // Starting loss scale (default: 65536)
	ScaleGrowth      float32 // Multiply scale by this after N good steps
	ScaleShrink      float32 // Multiply scale by this after overflow
	GrowthInterval   int     // Steps between scale increases
}

// DefaultMixedPrecisionConfig returns recommended FP16 settings.
func DefaultMixedPrecisionConfig() MixedPrecisionConfig {
	return MixedPrecisionConfig{
		Mode:             PrecisionFP16,
		DynamicLossScale: true,
		InitialScale:     65536.0,
		ScaleGrowth:      2.0,
		ScaleShrink:      0.5,
		GrowthInterval:   100,
	}
}

// LossScaler manages dynamic loss scaling for mixed precision training.
type LossScaler struct {
	config       MixedPrecisionConfig
	currentScale float32
	goodSteps    int
	overflows    int
}

// NewLossScaler creates a dynamic loss scaler.
func NewLossScaler(config MixedPrecisionConfig) *LossScaler {
	fmt.Printf("[MixedPrecision] %s mode, loss scale=%.0f\n",
		config.Mode, config.InitialScale)
	return &LossScaler{
		config:       config,
		currentScale: config.InitialScale,
	}
}

// Scale returns the current loss scale factor.
func (ls *LossScaler) Scale() float32 {
	return ls.currentScale
}

// ScaleGradients multiplies gradients by the loss scale before backward.
func (ls *LossScaler) ScaleGradients(grads *Tensor) {
	for i := range grads.Data {
		grads.Data[i] *= ls.currentScale
	}
}

// UnscaleGradients divides gradients by loss scale after backward.
// Returns true if gradients are valid (no overflow/NaN).
func (ls *LossScaler) UnscaleGradients(grads *Tensor) bool {
	invScale := 1.0 / ls.currentScale
	hasOverflow := false

	for i := range grads.Data {
		grads.Data[i] *= invScale

		// Check for overflow/NaN.
		if math.IsInf(float64(grads.Data[i]), 0) || math.IsNaN(float64(grads.Data[i])) {
			hasOverflow = true
			grads.Data[i] = 0 // Zero out invalid gradients.
		}
	}

	if hasOverflow {
		ls.overflows++
		if ls.config.DynamicLossScale {
			ls.currentScale *= ls.config.ScaleShrink
			if ls.currentScale < 1.0 {
				ls.currentScale = 1.0
			}
		}
		ls.goodSteps = 0
		return false
	}

	ls.goodSteps++
	if ls.config.DynamicLossScale && ls.goodSteps >= ls.config.GrowthInterval {
		ls.currentScale *= ls.config.ScaleGrowth
		ls.goodSteps = 0
	}

	return true
}

// Float32ToFloat16 converts FP32 data to FP16 representation (stored as uint16).
func Float32ToFloat16(data []float32) []uint16 {
	result := make([]uint16, len(data))
	for i, v := range data {
		result[i] = float32ToFloat16Bits(v)
	}
	return result
}

// Float16ToFloat32 converts FP16 data back to FP32.
func Float16ToFloat32(data []uint16) []float32 {
	result := make([]float32, len(data))
	for i, v := range data {
		result[i] = float16ToFloat32(v)
	}
	return result
}

// Float32ToBFloat16 converts FP32 data to BF16 representation.
func Float32ToBFloat16(data []float32) []uint16 {
	result := make([]uint16, len(data))
	for i, v := range data {
		result[i] = uint16(math.Float32bits(v) >> 16)
	}
	return result
}

// BFloat16ToFloat32 converts BF16 data back to FP32.
func BFloat16ToFloat32(data []uint16) []float32 {
	result := make([]float32, len(data))
	for i, v := range data {
		result[i] = bfloat16ToFloat32(v)
	}
	return result
}

// float32ToFloat16Bits converts a single FP32 to FP16 bits.
func float32ToFloat16Bits(f float32) uint16 {
	bits := math.Float32bits(f)
	sign := (bits >> 31) & 1
	exp := int((bits>>23)&0xFF) - 127
	mant := bits & 0x7FFFFF

	if exp > 15 {
		// Overflow to infinity.
		return uint16((sign << 15) | 0x7C00)
	}
	if exp < -14 {
		// Underflow to zero or subnormal.
		if exp < -24 {
			return uint16(sign << 15)
		}
		mant |= 0x800000
		shift := uint(-exp - 14 + 13)
		return uint16((sign << 15) | uint32(mant>>shift))
	}

	return uint16((sign << 15) | uint32(exp+15)<<10 | (mant >> 13))
}

// Stats returns loss scaler metrics.
func (ls *LossScaler) Stats() map[string]interface{} {
	return map[string]interface{}{
		"mode":          ls.config.Mode.String(),
		"current_scale": ls.currentScale,
		"good_steps":    ls.goodSteps,
		"overflows":     ls.overflows,
	}
}
