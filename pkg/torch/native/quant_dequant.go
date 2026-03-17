// quant_dequant.go implements on-the-fly dequantization for quantized weight formats.
//
// WHY: Quantized weights (Q4, Q8) use 4x-8x less memory than FP32.
// Instead of dequantizing the entire model at load time (doubling memory),
// we dequantize each row on-the-fly during MatVecMul. This keeps memory
// usage at the quantized size while compute happens in FP32.
//
// FORMATS:
//   Q8_0: 32-element blocks, each with one FP16 scale factor + 32 int8 values
//   Q4_0: 32-element blocks, each with one FP16 scale factor + 16 bytes (2 nibbles each)
//
// The block size of 32 matches llama.cpp/GGUF conventions.
package native

import (
	"math"
)

const qBlockSize = 32 // elements per quantization block (matches GGUF)

// QuantBlock8 represents a Q8_0 quantization block.
// 32 int8 values with a shared FP16 scale factor.
type QuantBlock8 struct {
	Scale float32  // dequantization scale (stored as f32 for speed)
	Quant [32]int8 // quantized values
}

// QuantBlock4 represents a Q4_0 quantization block.
// 32 values packed into 16 bytes (4 bits each) with a shared scale.
type QuantBlock4 struct {
	Scale float32   // dequantization scale
	Quant [16]uint8 // packed 4-bit values (2 per byte)
}

// DequantDotQ8 computes the dot product of a Q8_0 quantized row with an FP32 vector.
// This is the hot path for quantized inference - no intermediate f32 buffer needed.
func DequantDotQ8(blocks []QuantBlock8, v []float32) float32 {
	var sum float32
	vOff := 0

	for _, block := range blocks {
		scale := block.Scale
		// Unrolled 8-way accumulation within each block.
		var s0, s1, s2, s3 float32
		for j := 0; j < 32; j += 4 {
			s0 += float32(block.Quant[j]) * v[vOff+j]
			s1 += float32(block.Quant[j+1]) * v[vOff+j+1]
			s2 += float32(block.Quant[j+2]) * v[vOff+j+2]
			s3 += float32(block.Quant[j+3]) * v[vOff+j+3]
		}
		sum += scale * (s0 + s1 + s2 + s3)
		vOff += 32
	}

	return sum
}

// DequantDotQ4 computes the dot product of a Q4_0 quantized row with an FP32 vector.
// Each byte holds two 4-bit values: low nibble first, high nibble second.
func DequantDotQ4(blocks []QuantBlock4, v []float32) float32 {
	var sum float32
	vOff := 0

	for _, block := range blocks {
		scale := block.Scale
		var blockSum float32

		for j := 0; j < 16; j++ {
			packed := block.Quant[j]
			// Low nibble (signed 4-bit: -8 to 7).
			lo := int8(packed&0x0F) - 8
			// High nibble (signed 4-bit: -8 to 7).
			hi := int8(packed>>4) - 8

			blockSum += float32(lo) * v[vOff]
			blockSum += float32(hi) * v[vOff+1]
			vOff += 2
		}

		sum += scale * blockSum
	}

	return sum
}

// QuantizeRowQ8 quantizes a float32 row into Q8_0 blocks.
// Used during model loading to convert FP32 weights to Q8 format.
func QuantizeRowQ8(data []float32) []QuantBlock8 {
	numBlocks := (len(data) + qBlockSize - 1) / qBlockSize
	blocks := make([]QuantBlock8, numBlocks)

	for b := 0; b < numBlocks; b++ {
		start := b * qBlockSize
		end := start + qBlockSize
		if end > len(data) {
			end = len(data)
		}

		// Find max absolute value for the scale.
		var maxAbs float32
		for _, v := range data[start:end] {
			abs := float32(math.Abs(float64(v)))
			if abs > maxAbs {
				maxAbs = abs
			}
		}

		scale := maxAbs / 127.0
		if scale == 0 {
			scale = 1.0 // avoid division by zero
		}
		blocks[b].Scale = scale
		invScale := 127.0 / maxAbs
		if maxAbs == 0 {
			invScale = 0
		}

		for i := start; i < end; i++ {
			q := int8(math.Round(float64(data[i]) * float64(invScale)))
			blocks[b].Quant[i-start] = q
		}
	}

	return blocks
}

// QuantizeRowQ4 quantizes a float32 row into Q4_0 blocks.
func QuantizeRowQ4(data []float32) []QuantBlock4 {
	numBlocks := (len(data) + qBlockSize - 1) / qBlockSize
	blocks := make([]QuantBlock4, numBlocks)

	for b := 0; b < numBlocks; b++ {
		start := b * qBlockSize
		end := start + qBlockSize
		if end > len(data) {
			end = len(data)
		}

		// Find max absolute value.
		var maxAbs float32
		for _, v := range data[start:end] {
			abs := float32(math.Abs(float64(v)))
			if abs > maxAbs {
				maxAbs = abs
			}
		}

		scale := maxAbs / 7.0
		if scale == 0 {
			scale = 1.0
		}
		blocks[b].Scale = scale
		invScale := 7.0 / maxAbs
		if maxAbs == 0 {
			invScale = 0
		}

		chunk := data[start:end]
		for j := 0; j < len(chunk); j += 2 {
			lo := int8(math.Round(float64(chunk[j]) * float64(invScale)))
			if lo < -8 {
				lo = -8
			}
			if lo > 7 {
				lo = 7
			}

			var hi int8
			if j+1 < len(chunk) {
				hi = int8(math.Round(float64(chunk[j+1]) * float64(invScale)))
				if hi < -8 {
					hi = -8
				}
				if hi > 7 {
					hi = 7
				}
			}

			blocks[b].Quant[j/2] = uint8(lo+8) | (uint8(hi+8) << 4)
		}
	}

	return blocks
}
