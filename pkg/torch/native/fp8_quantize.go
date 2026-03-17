// fp8_quantize.go implements OCP FP8 (8-bit floating point) formats.
//
// WHAT: FP8 is the new hardware standard for LLM inference and training
// (NVIDIA Hopper/Blackwell, AMD MI300). Unlike INT8 quantization which
// uses a uniform grid, FP8 uses exponent+mantissa to maintain dynamic
// range while halving the memory footprint vs FP16.
//
// FORMATS:
//   - E4M3: 1 sign bit, 4 exponent bits, 3 mantissa bits.
//     Used for weights and activations (higher precision, lower range).
//     Max value: 448. Min non-zero: 2^-9 (0.001953125).
//   - E5M2: 1 sign bit, 5 exponent bits, 2 mantissa bits.
//     Used for gradients (higher range, lower precision).
//     Max value: 57344. Min non-zero: 2^-16 (0.000015259).
//
// WHY: FP8 offers nearly the same accuracy as FP16 but with 2x the
// throughput on tensor cores, without the complex calibration required by
// INT8 or INT4 methods.
package native

import (
	"math"
)

// FP8Format specifies the 8-bit float layout.
type FP8Format int

const (
	FP8_E4M3 FP8Format = iota // Standard for weights/activations
	FP8_E5M2                  // Standard for gradients
)

// FP8Tensor represents a block of memory quantized to FP8.
//
// In hardware, NVFP4 (4-bit) and FP8 are natively supported by
// Tensor Cores. In pure Go, we implement the bit-level pack/unpack
// software reference to support GGUF models that load FP8 weights.
type FP8Tensor struct {
	Format FP8Format
	Data   []byte    // Raw 8-bit values
	Shape  []int     // Tensor dimensions
	Scales []float32 // Block or per-tensor scales for dynamic range matching
}

// Float32ToE4M3 converts a standard 32-bit float to FP8 E4M3 format.
func Float32ToE4M3(f float32) byte {
	// E4M3: SEEE EMMM
	// Exponent bias: 7

	bits := math.Float32bits(f)
	sign := (bits >> 31) & 0x01
	exp32 := int((bits >> 23) & 0xFF)
	mant32 := bits & 0x7FFFFF

	// Handle zero, denormals, and NaN/Inf in fp32
	if exp32 == 0 {
		return byte(sign << 7) // return +/- 0
	}
	if exp32 == 255 {
		if mant32 == 0 {
			// FP32 Inf -> E4M3 NaN (E4M3 has no Inf, only NaN at S.1111.111)
			return byte(0x7F | (sign << 7)) // NaN
		}
		// FP32 NaN
		return byte(0x7F | (sign << 7)) // NaN
	}

	// Calculate E4M3 exponent (bias 7 vs fp32 bias 127) -> offset = 120
	exp8 := exp32 - 120

	// Handle overflow (Max E4M3 is 448.0 = 0x43E00000 in fp32)
	// Bias 7 means max exp is 15 -> 15-7 = 8.
	if exp8 > 15 {
		// E4M3 has no infinity, so max value is E=1111, M=110 (254 w/o sign)
		// 1111_111 is reserved for NaN. Max is 0x7E.
		return byte(0x7E | (sign << 7))
	}

	// Handle underflow / denormals in E4M3
	if exp8 <= 0 {
		// Convert to denormal
		shift := 1 - exp8
		if shift > 5 { // Too small even for denormal
			return byte(sign << 7) // zero
		}
		// Add implicit leading 1 to mantissa, then shift
		mant := (0x800000 | mant32) >> (23 - 3 + shift)
		// Rounding (nearest even)
		roundBit := (mant32 >> (23 - 3 + shift - 1)) & 1
		if roundBit == 1 {
			mant++ // Simplistic rounding
		}
		if mant > 0x07 { // Overflowed denormal becomes normal
			mant = 0
			exp8 = 1
		} else {
			exp8 = 0
		}
		return byte((sign << 7) | uint32(exp8<<3) | (mant & 0x07))
	}

	// Normal E4M3 number
	mant8 := (mant32 >> 20) // 23 - 3 = 20 bits shift
	// Rounding (simplistic tie-break)
	if (mant32 >> 19) & 1 == 1 {
		mant8++
		if mant8 > 7 { // mantissa overflowed
			mant8 = 0
			exp8++
			if exp8 > 15 {
				return byte(0x7E | (sign << 7)) // saturate to max
			}
		}
	}
	
	return byte((sign << 7) | uint32(exp8<<3) | (mant8 & 0x07))
}

// E4M3ToFloat32 converts an FP8 E4M3 byte back to a 32-bit float.
func E4M3ToFloat32(b byte) float32 {
	sign := uint32(b>>7) & 0x01
	exp8 := uint32(b>>3) & 0x0F
	mant8 := uint32(b) & 0x07

	var exp32, mant32 uint32

	if exp8 == 15 && mant8 == 7 {
		// E4M3 NaN
		exp32 = 255
		mant32 = 0x7FFFFF // A standard NaN payload
	} else if exp8 == 0 {
		if mant8 == 0 {
			// Zero
			exp32 = 0
			mant32 = 0
		} else {
			// Denormal in E4M3 -> Normal in FP32
			exp32 = 127 - 7 + 1 // Denormal exp represents 2^-6
			// Normalize mantissa
			for (mant8 & 0x04) == 0 {
				mant8 <<= 1
				exp32--
			}
			mant8 &= 0x03 // Remove implicit 1
			mant32 = mant8 << 21 // 23 - 2 = 21
		}
	} else {
		// Normal number
		exp32 = exp8 + 120 // 127 - 7 = 120
		mant32 = mant8 << 20 // 23 - 3 = 20
	}

	bits := (sign << 31) | (exp32 << 23) | mant32
	return math.Float32frombits(bits)
}

// Float32ToE5M2 converts a standard 32-bit float to FP8 E5M2 format.
// E5M2 exactly follows IEEE 754 structure (like an 8-bit FP16).
func Float32ToE5M2(f float32) byte {
	// FP32 is SEEEEEEE EMMMMMMM MMMMMMMM MMMMMMMM
	// E5M2 is SEEEEE MM
	// Just chop off the bottom 16 bits of FP16, or shift FP32 heavily.

	bits := math.Float32bits(f)
	
	if bits == 0 || bits == 0x80000000 {
		return byte(bits >> 24) // +/- 0
	}

	sign := (bits >> 31) & 0x01
	exp32 := int((bits >> 23) & 0xFF)
	mant32 := bits & 0x7FFFFF

	if exp32 == 255 {
		// Inf or NaN
		if mant32 == 0 {
			return byte((sign << 7) | 0x7C) // Inf
		}
		return byte((sign << 7) | 0x7F) // NaN
	}

	// Bias conversion (127 -> 15 bias) offset = 112
	exp8 := exp32 - 112

	if exp8 > 30 {
		// Overflow -> Inf
		return byte((sign << 7) | 0x7C)
	}

	if exp8 <= 0 {
		// Underflow / denormal
		shift := 1 - exp8
		if shift > 4 {
			return byte(sign << 7) // zero
		}
		mant := (0x800000 | mant32) >> (23 - 2 + shift)
		if (mant32 >> (23 - 2 + shift - 1)) & 1 == 1 {
			mant++
		}
		return byte((sign << 7) | (mant & 0x03))
	}

	// Normal
	mant8 := mant32 >> 21 // 23 - 2 = 21
	// Rounding
	if (mant32 >> 20) & 1 == 1 {
		mant8++
		if mant8 > 3 {
			mant8 = 0
			exp8++
			if exp8 > 30 {
				return byte((sign << 7) | 0x7C) // Inf
			}
		}
	}

	return byte((sign << 7) | uint32(exp8<<2) | (mant8 & 0x03))
}

// E5M2ToFloat32 converts an FP8 E5M2 byte back to a 32-bit float.
func E5M2ToFloat32(b byte) float32 {
	sign := uint32(b>>7) & 0x01
	exp8 := uint32(b>>2) & 0x1F
	mant8 := uint32(b) & 0x03

	var exp32, mant32 uint32

	if exp8 == 31 {
		exp32 = 255
		if mant8 == 0 {
			mant32 = 0 // Inf
		} else {
			mant32 = 0x7FFFFF // NaN
		}
	} else if exp8 == 0 {
		if mant8 == 0 {
			exp32 = 0
			mant32 = 0 // Zero
		} else {
			// Denormal
			exp32 = 127 - 15 + 1
			for (mant8 & 0x02) == 0 {
				mant8 <<= 1
				exp32--
			}
			mant8 &= 0x01
			mant32 = mant8 << 22 // 23 - 1 = 22
		}
	} else {
		// Normal
		exp32 = exp8 + 112 // 127 - 15 = 112
		mant32 = mant8 << 21 // 23 - 2 = 21
	}

	bits := (sign << 31) | (exp32 << 23) | mant32
	return math.Float32frombits(bits)
}

// QuantizeBlockFP8 quantizes a block of float32s into FP8 (E4M3) with a single scale.
func QuantizeBlockFP8(src []float32, dst []byte) float32 {
	// Find max absolute value to determine scale.
	maxAbs := float32(0)
	for _, v := range src {
		abs := float32(math.Abs(float64(v)))
		if abs > maxAbs {
			maxAbs = abs
		}
	}

	// Map maxAbs to 448.0 (Max E4M3).
	scale := maxAbs / 448.0
	if scale == 0 {
		scale = 1e-9
	}
	invScale := 1.0 / scale

	for i, v := range src {
		scaled := v * invScale
		dst[i] = Float32ToE4M3(scaled)
	}

	return scale
}

// DequantizeBlockFP8 restores a block of FP8 back to float32.
func DequantizeBlockFP8(src []byte, dst []float32, scale float32) {
	for i, b := range src {
		val := E4M3ToFloat32(b)
		dst[i] = val * scale
	}
}
