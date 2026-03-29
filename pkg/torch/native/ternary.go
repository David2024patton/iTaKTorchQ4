// ternary.go implements BitNet 1.58-bit (ternary) inference kernels.
//
// WHAT: BitNet i2_s format packs 128 ternary weights {-1, 0, 1} into 32 bytes.
// Weights are stored in 2-bit interleaved format across 4 groups of 32 elements.
//
// WHY: Native ternary inference is significantly faster than floating point
// because multiplications are replaced by conditional additions/subtractions.
//
// HOW: UnpackI2S extracts the 2-bit values and TernaryDot performs the
// dot product without multiplications.
package native

import (
	"encoding/binary"
	"math"
	"unsafe"
)

// UnpackI2S dequantizes a 32-byte BitNet i2_s block into 128 ternary weights.
// Returns a slice of int8 values {-1, 0, 1}.
func UnpackI2S(block []byte) []int8 {
	if len(block) < 32 {
		return nil
	}
	
	weights := make([]int8, 128)
	
	// Packing: interleaved 4 groups of 32.
	// Byte i (0-31) contains elements i, i+32, i+64, i+96.
	// Bits: [7:6] Group 0, [5:4] Group 1, [3:2] Group 2, [1:0] Group 3.
	for i := 0; i < 32; i++ {
		b := block[i]
		
		// Map 2-bit to ternary: 00=0, 01=+1, 10=-1
		weights[i]    = mapTernary((b >> 6) & 0x03)
		weights[i+32] = mapTernary((b >> 4) & 0x03)
		weights[i+64] = mapTernary((b >> 2) & 0x03)
		weights[i+96] = mapTernary(b & 0x03)
	}
	
	return weights
}

// mapTernary converts 2-bit GGUF i2_s encoding to ternary int8.
func mapTernary(val byte) int8 {
	switch val {
	case 1:  return 1  // 01 -> +1
	case 2:  return -1 // 10 -> -1
	default: return 0  // 00 -> 0 (11 is fallback to 0)
	}
}

// TernaryDot computes the dot product of a float32 vector x and a ternary weight vector w.
// Optimized to avoid multiplications: sum += x[i] if w[i]==1, sum -= x[i] if w[i]==-1.
func TernaryDot(x []float32, w []int8, scale float32) float32 {
	var sum float32
	n := len(x)
	if len(w) < n {
		n = len(w)
	}
	
	for i := 0; i < n; i++ {
		switch w[i] {
		case 1:  sum += x[i]
		case -1: sum -= x[i]
		}
	}
	
	return sum * scale
}

// TernaryDotBlock computes dot product for a single 128-element i2_s block.
// Does NOT unpack into intermediary int8 slice, processes bits directly.
func TernaryDotBlock(x []float32, block []byte, scale float32) float32 {
	var sum float32
	
	// Process 32 bytes, each contributing to 4 positions in x.
	for i := 0; i < 32; i++ {
		b := block[i]
		
		// Group 0: i
		v0 := (b >> 6) & 0x03
		if v0 == 1 { sum += x[i] } else if v0 == 2 { sum -= x[i] }
		
		// Group 1: i+32
		v1 := (b >> 4) & 0x03
		if v1 == 1 { sum += x[i+32] } else if v1 == 2 { sum -= x[i+32] }
		
		// Group 2: i+64
		v2 := (b >> 2) & 0x03
		if v2 == 1 { sum += x[i+64] } else if v2 == 2 { sum -= x[i+64] }
		
		// Group 3: i+96
		v3 := b & 0x03
		if v3 == 1 { sum += x[i+96] } else if v3 == 2 { sum -= x[i+96] }
	}
	
	return sum * scale
}

// TernaryDotSparseBlock performs a dot product skipping zero weights (Hyper-Sparsity).
// This is Phase 33: "Bleeding Edge" optimization.
func TernaryDotSparseBlock(x []float32, w []byte, scale float32) float32 {
	var dot float32
	// Each 32-byte block stores 128 elements.
	// 2 bits per element.
	for i := 0; i < 32; i++ {
		b := w[i]
		if b == 0 { // All 4 elements in this byte are zero.
			continue
		}
		
		// Unpack 4 elements from the byte.
		for j := 0; j < 4; j++ {
			val := (b >> uint(j*2)) & 0x03
			if val == 0 {
				continue // Skip zero
			}
			
			pixel := x[i*4+j]
			if val == 0x01 || val == 0x02 { // Positive (sign bit 0)
				dot += pixel
			} else { // Negative (sign bit 1)
				dot -= pixel
			}
		}
	}
	return dot * scale
}

// MoELayer holds the gating network and experts for a Mixtral-style block.
// GGUF i2_s tensors store 8 copies of the float32 scale at the end.
func ExtractScale(data []byte) float32 {
	if len(data) < 32 {
		return 1.0
	}
	// The scale is at the very end of the weight data.
	return math.Float32frombits(binary.LittleEndian.Uint32(data[len(data)-32 : len(data)-28]))
}

// Binary dequantization helper for F32 conversion
func binaryToF32(v uint32) float32 {
	return *(*float32)(unsafe.Pointer(&v))
}
