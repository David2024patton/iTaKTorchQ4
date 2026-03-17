// dequant.go implements GGML quantized tensor dequantization in pure Go.
//
// Each GGML quantization format stores weights in fixed-size blocks with
// scaling factors. This file converts each format back to float32 at load time.
//
// Block structures (all little-endian):
//   - Q4_0:  32 values per 18-byte block  (FP16 scale + 16 nibble-packed bytes)
//   - Q4_1:  32 values per 20-byte block  (FP16 scale + FP16 min + 16 nibbles)
//   - Q5_0:  32 values per 22-byte block  (FP16 scale + 4-byte high-bit mask + 16 nibbles)
//   - Q5_1:  32 values per 24-byte block  (FP16 scale + FP16 min + 4-byte mask + 16 nibbles)
//   - Q8_0:  32 values per 34-byte block  (FP16 scale + 32 int8 values)
//   - Q2_K: 256 values per 84-byte block  (super-block with sub-scales)
//   - Q4_K: 256 values per 144-byte block (super-block with sub-scales + nibble values)
//   - Q6_K: 256 values per 210-byte block (super-block with 6-bit values)
package native

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
)

// bfloat16ToFloat32 converts a BF16 value to float32.
// BF16 uses the same 8-bit exponent as float32 but only 7 mantissa bits.
// Conversion is simply shifting left by 16 bits.
func bfloat16ToFloat32(b uint16) float32 {
	return math.Float32frombits(uint32(b) << 16)
}

// --- Q4_0: 32 values per 18-byte block ---
// Layout: [2 bytes FP16 scale] [16 bytes: 32 nibbles packed as low/high]
// Formula: float32(nibble - 8) * scale

func dequantQ4_0(r io.Reader, out []float32, n uint64) error {
	const blockSize = 32
	nBlocks := n / blockSize
	buf := make([]byte, 18)

	for b := uint64(0); b < nBlocks; b++ {
		if _, err := io.ReadFull(r, buf); err != nil {
			return fmt.Errorf("read Q4_0 block %d: %w", b, err)
		}

		scale := float16ToFloat32(binary.LittleEndian.Uint16(buf[0:2]))
		base := b * blockSize

		for j := 0; j < 16; j++ {
			nibByte := buf[2+j]
			lo := float32(nibByte&0x0F) - 8.0
			hi := float32(nibByte>>4) - 8.0
			out[base+uint64(j)] = lo * scale
			out[base+uint64(j+16)] = hi * scale
		}
	}
	return nil
}

// --- Q4_1: 32 values per 20-byte block ---
// Layout: [2 bytes FP16 scale] [2 bytes FP16 min] [16 bytes nibbles]
// Formula: float32(nibble) * scale + min

func dequantQ4_1(r io.Reader, out []float32, n uint64) error {
	const blockSize = 32
	nBlocks := n / blockSize
	buf := make([]byte, 20)

	for b := uint64(0); b < nBlocks; b++ {
		if _, err := io.ReadFull(r, buf); err != nil {
			return fmt.Errorf("read Q4_1 block %d: %w", b, err)
		}

		scale := float16ToFloat32(binary.LittleEndian.Uint16(buf[0:2]))
		min := float16ToFloat32(binary.LittleEndian.Uint16(buf[2:4]))
		base := b * blockSize

		for j := 0; j < 16; j++ {
			nibByte := buf[4+j]
			lo := float32(nibByte & 0x0F)
			hi := float32(nibByte >> 4)
			out[base+uint64(j)] = lo*scale + min
			out[base+uint64(j+16)] = hi*scale + min
		}
	}
	return nil
}

// --- Q5_0: 32 values per 22-byte block ---
// Layout: [2 bytes FP16 scale] [4 bytes high-bit mask] [16 bytes nibbles]
// Formula: float32(nibble | (high_bit << 4) - 16) * scale

func dequantQ5_0(r io.Reader, out []float32, n uint64) error {
	const blockSize = 32
	nBlocks := n / blockSize
	buf := make([]byte, 22)

	for b := uint64(0); b < nBlocks; b++ {
		if _, err := io.ReadFull(r, buf); err != nil {
			return fmt.Errorf("read Q5_0 block %d: %w", b, err)
		}

		scale := float16ToFloat32(binary.LittleEndian.Uint16(buf[0:2]))
		highBits := binary.LittleEndian.Uint32(buf[2:6])
		base := b * blockSize

		for j := 0; j < 16; j++ {
			nibByte := buf[6+j]
			lo4 := uint8(nibByte & 0x0F)
			hi4 := uint8(nibByte >> 4)

			// Extract the 5th bit from the high-bit mask.
			loBit5 := uint8((highBits >> uint(j)) & 1)
			hiBit5 := uint8((highBits >> uint(j+16)) & 1)

			loVal := float32(int8((lo4|(loBit5<<4)))) - 16.0
			hiVal := float32(int8((hi4|(hiBit5<<4)))) - 16.0
			out[base+uint64(j)] = loVal * scale
			out[base+uint64(j+16)] = hiVal * scale
		}
	}
	return nil
}

// --- Q5_1: 32 values per 24-byte block ---
// Layout: [2 bytes FP16 scale] [2 bytes FP16 min] [4 bytes high-bit mask] [16 bytes nibbles]
// Formula: float32(nibble | (high_bit << 4)) * scale + min

func dequantQ5_1(r io.Reader, out []float32, n uint64) error {
	const blockSize = 32
	nBlocks := n / blockSize
	buf := make([]byte, 24)

	for b := uint64(0); b < nBlocks; b++ {
		if _, err := io.ReadFull(r, buf); err != nil {
			return fmt.Errorf("read Q5_1 block %d: %w", b, err)
		}

		scale := float16ToFloat32(binary.LittleEndian.Uint16(buf[0:2]))
		min := float16ToFloat32(binary.LittleEndian.Uint16(buf[2:4]))
		highBits := binary.LittleEndian.Uint32(buf[4:8])
		base := b * blockSize

		for j := 0; j < 16; j++ {
			nibByte := buf[8+j]
			lo4 := uint8(nibByte & 0x0F)
			hi4 := uint8(nibByte >> 4)

			loBit5 := uint8((highBits >> uint(j)) & 1)
			hiBit5 := uint8((highBits >> uint(j+16)) & 1)

			loVal := float32(lo4 | (loBit5 << 4))
			hiVal := float32(hi4 | (hiBit5 << 4))
			out[base+uint64(j)] = loVal*scale + min
			out[base+uint64(j+16)] = hiVal*scale + min
		}
	}
	return nil
}

// --- Q8_0: 32 values per 34-byte block ---
// Layout: [2 bytes FP16 scale] [32 bytes int8 values]
// Formula: float32(int8_value) * scale

func dequantQ8_0(r io.Reader, out []float32, n uint64) error {
	const blockSize = 32
	nBlocks := n / blockSize
	buf := make([]byte, 34)

	for b := uint64(0); b < nBlocks; b++ {
		if _, err := io.ReadFull(r, buf); err != nil {
			return fmt.Errorf("read Q8_0 block %d: %w", b, err)
		}

		scale := float16ToFloat32(binary.LittleEndian.Uint16(buf[0:2]))
		base := b * blockSize

		for j := 0; j < 32; j++ {
			out[base+uint64(j)] = float32(int8(buf[2+j])) * scale
		}
	}
	return nil
}

// --- Q2_K: 256 values per 84-byte block ---
// Super-block layout:
//   [32 bytes: scales - packed 4-bit scale/min for 16 sub-blocks]
//   [16 bytes: quantized values - 2 bits each, 4 values per byte]
//   [... more quant data ...]
//   [2 bytes FP16 d (super-scale)]
//   [2 bytes FP16 dmin (super-min)]

func dequantQ2_K(r io.Reader, out []float32, n uint64) error {
	const blockSize = 256
	nBlocks := n / blockSize
	buf := make([]byte, 84)

	for b := uint64(0); b < nBlocks; b++ {
		if _, err := io.ReadFull(r, buf); err != nil {
			return fmt.Errorf("read Q2_K block %d: %w", b, err)
		}

		// Scales are in first 16 bytes (packed nibbles for 16 sub-blocks).
		scales := buf[0:16]
		// Quantized data: 64 bytes, 4 values per byte (2 bits each).
		qs := buf[16:80]
		// Super-block scale and min.
		d := float16ToFloat32(binary.LittleEndian.Uint16(buf[80:82]))
		dmin := float16ToFloat32(binary.LittleEndian.Uint16(buf[82:84]))

		base := b * blockSize
		idx := 0

		for subBlock := 0; subBlock < 16; subBlock++ {
			scaleByte := scales[subBlock]
			sc := float32(scaleByte & 0x0F)
			mn := float32(scaleByte >> 4)

			blockScale := d * sc
			blockMin := dmin * mn

			for j := 0; j < 16; j++ {
				byteIdx := idx / 4
				shift := uint((idx % 4) * 2)
				q := float32((qs[byteIdx] >> shift) & 0x03)
				out[base+uint64(idx)] = q*blockScale - blockMin
				idx++
			}
		}
	}
	return nil
}

// --- Q4_K: 256 values per 144-byte block ---
// Super-block with sub-block scales and 4-bit quantized values.

func dequantQ4_K(r io.Reader, out []float32, n uint64) error {
	const blockSize = 256
	nBlocks := n / blockSize
	buf := make([]byte, 144)

	for b := uint64(0); b < nBlocks; b++ {
		if _, err := io.ReadFull(r, buf); err != nil {
			return fmt.Errorf("read Q4_K block %d: %w", b, err)
		}

		// Super-block scale and min (FP16).
		d := float16ToFloat32(binary.LittleEndian.Uint16(buf[0:2]))
		dmin := float16ToFloat32(binary.LittleEndian.Uint16(buf[2:4]))

		// Sub-block scales and mins (12 bytes = 8 sub-blocks, 6-bit each, packed).
		scalesRaw := buf[4:16]

		// Quantized nibbles: 128 bytes = 256 values * 4 bits.
		qs := buf[16:144]

		base := b * blockSize

		for subBlock := 0; subBlock < 8; subBlock++ {
			// Extract 6-bit scale and min from packed representation.
			var sc, mn float32
			if subBlock < 4 {
				sc = float32(scalesRaw[subBlock] & 0x3F)
				mn = float32(scalesRaw[subBlock+4] & 0x3F)
			} else {
				sc = float32((scalesRaw[subBlock+4]&0x0F) | ((scalesRaw[subBlock-4]>>6)<<4))
				mn = float32((scalesRaw[subBlock+4]>>4) | ((scalesRaw[subBlock]>>6)<<4))
			}

			blockScale := d * sc
			blockMin := dmin * mn

			for j := 0; j < 32; j++ {
				globalIdx := subBlock*32 + j
				byteIdx := globalIdx / 2
				var nibble uint8
				if globalIdx%2 == 0 {
					nibble = qs[byteIdx] & 0x0F
				} else {
					nibble = qs[byteIdx] >> 4
				}
				out[base+uint64(globalIdx)] = float32(nibble)*blockScale - blockMin
			}
		}
	}
	return nil
}

// --- Q6_K: 256 values per 210-byte block ---
// 6-bit quantization with sub-block scales.

func dequantQ6_K(r io.Reader, out []float32, n uint64) error {
	const blockSize = 256
	nBlocks := n / blockSize
	buf := make([]byte, 210)

	for b := uint64(0); b < nBlocks; b++ {
		if _, err := io.ReadFull(r, buf); err != nil {
			return fmt.Errorf("read Q6_K block %d: %w", b, err)
		}

		// Layout: [128 bytes ql] [64 bytes qh] [16 bytes scales] [2 bytes FP16 d]
		ql := buf[0:128]   // Low 4 bits of each 6-bit value.
		qh := buf[128:192] // High 2 bits of each 6-bit value.
		sc := buf[192:208] // Per-sub-block int8 scales.
		d := float16ToFloat32(binary.LittleEndian.Uint16(buf[208:210]))

		base := b * blockSize

		for subBlock := 0; subBlock < 16; subBlock++ {
			blockScale := d * float32(int8(sc[subBlock]))

			for j := 0; j < 16; j++ {
				idx := subBlock*16 + j

				// Extract 4 low bits.
				qlIdx := idx / 2
				var lo4 uint8
				if idx%2 == 0 {
					lo4 = ql[qlIdx] & 0x0F
				} else {
					lo4 = ql[qlIdx] >> 4
				}

				// Extract 2 high bits.
				qhIdx := idx / 4
				qhShift := uint((idx % 4) * 2)
				hi2 := (qh[qhIdx] >> qhShift) & 0x03

				// Combine: 6-bit value, then subtract 32 for signed range.
				q6 := float32(int8(lo4|(hi2<<4))) - 32.0
				out[base+uint64(idx)] = q6 * blockScale
			}
		}
	}
	return nil
}
