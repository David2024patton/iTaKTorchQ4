// matmul.go implements matrix multiplication for the GOTensor engine.
//
// Two implementations:
//   1. Naive: simple triple-nested loop (O(n^3), cache-unfriendly)
//   2. Blocked: processes tiles that fit in L1/L2 cache (2-3x faster)
//
// Both are pure Go with no SIMD or assembly. The blocked version
// is the default because modern CPUs have 32-256KB L1 data cache,
// and a 32x32 tile of float32 (4KB) fits comfortably.
package native

import (
	"encoding/binary"
	"fmt"
	"math"
	"runtime"
	"sync"
)

// MatMul multiplies two 2D tensors: C = A * B.
//
// Rules:
//   - A must be [M, K]
//   - B must be [K, N]
//   - Result is [M, N]
//
// Uses the blocked implementation for better cache performance.
func MatMul(a, b *Tensor) *Tensor {
	if len(a.Shape) != 2 || len(b.Shape) != 2 {
		panic(fmt.Sprintf("MatMul: expected 2D tensors, got %v and %v", a.Shape, b.Shape))
	}
	m, k := a.Shape[0], a.Shape[1]
	k2, n := b.Shape[0], b.Shape[1]
	if k != k2 {
		panic(fmt.Sprintf("MatMul: inner dimensions mismatch: %d vs %d", k, k2))
	}

	if b.Type == 12 { // ggmlTypeQ4_K
		return matMulQ4(a, b, m, k, n)
	}
	if a.Type == 12 {
		panic("MatMul: 'A' Tensor is Q4_K, which is unexpected for Activations.")
	}

	return matMulBlocked(a.Data, b.Data, m, k, n)
}

// matMulBlocked performs tiled matrix multiplication.
//
// Why blocked? The naive triple loop has terrible cache behavior because
// accessing B column-wise causes a cache miss every K elements.
// By processing 32x32 tiles, all data fits in L1 cache, giving 2-3x speedup.
//
//	for each tile (I, J, L):
//	  for i in tile_I:
//	    for l in tile_L:
//	      a_val = A[i,l]  // sequential access (cache-friendly)
//	      for j in tile_J:
//	        C[i,j] += a_val * B[l,j]  // sequential access (cache-friendly)
func matMulBlocked(a, b []float32, m, k, n int) *Tensor {
	const tileSize = 32 // 32x32 * 4 bytes = 4KB fits in L1 cache

	result := NewTensor([]int{m, n})
	c := result.Data

	numWorkers := runtime.NumCPU()
	if numWorkers < 1 {
		numWorkers = 1
	}

	rowsPerWorker := ((m + numWorkers - 1) / numWorkers)
	rowsPerWorker = ((rowsPerWorker + tileSize - 1) / tileSize) * tileSize
	if rowsPerWorker == 0 {
		rowsPerWorker = tileSize
	}

	var wg sync.WaitGroup

	for w := 0; w < numWorkers; w++ {
		startRow := w * rowsPerWorker
		if startRow >= m {
			break
		}
		endRow := startRow + rowsPerWorker
		if endRow > m {
			endRow = m
		}

		wg.Add(1)
		go func(rStart, rEnd int) {
			defer wg.Done()
			for i0 := rStart; i0 < rEnd; i0 += tileSize {
				iEnd := i0 + tileSize
				if iEnd > rEnd {
					iEnd = rEnd
				}
				for l0 := 0; l0 < k; l0 += tileSize {
					lEnd := l0 + tileSize
					if lEnd > k {
						lEnd = k
					}
					for j0 := 0; j0 < n; j0 += tileSize {
						jEnd := j0 + tileSize
						if jEnd > n {
							jEnd = n
						}
						// Inner tile: all accesses are cache-friendly.
						for i := i0; i < iEnd; i++ {
							for l := l0; l < lEnd; l++ {
								aVal := a[i*k+l]
								for j := j0; j < jEnd; j++ {
									c[i*n+j] += aVal * b[l*n+j]
								}
							}
						}
					}
				}
			}
		}(startRow, endRow)
	}
	wg.Wait()

	return result
}

// MatVecMul multiplies a matrix by a vector: result = A * v.
// A is [M, K], v is [K], result is [M].
// Optimized for the common case of applying a weight matrix to a hidden state.
func MatVecMul(a *Tensor, v *Tensor) *Tensor {
	if len(a.Shape) != 2 || len(v.Shape) != 1 {
		panic(fmt.Sprintf("MatVecMul: expected [M,K] * [K], got %v * %v", a.Shape, v.Shape))
	}
	m, k := a.Shape[0], a.Shape[1]
	if v.Shape[0] != k {
		panic(fmt.Sprintf("MatVecMul: dimension mismatch: A[%d,%d] * v[%d]", m, k, v.Shape[0]))
	}

	result := NewTensor([]int{m})

	numWorkers := runtime.NumCPU()
	if numWorkers < 1 {
		numWorkers = 1
	}

	rowsPerWorker := (m + numWorkers - 1) / numWorkers
	if rowsPerWorker == 0 {
		rowsPerWorker = 1
	}

	var wg sync.WaitGroup
	for w := 0; w < numWorkers; w++ {
		startRow := w * rowsPerWorker
		if startRow >= m {
			break
		}
		endRow := startRow + rowsPerWorker
		if endRow > m {
			endRow = m
		}

		wg.Add(1)
		go func(rStart, rEnd int) {
			defer wg.Done()
			for i := rStart; i < rEnd; i++ {
				rowOff := i * k
				result.Data[i] = Dot(a.Data[rowOff:rowOff+k], v.Data)
			}
		}(startRow, endRow)
	}
	wg.Wait()

	return result
}

func matMulQ4(a, b *Tensor, m, k, n int) *Tensor {
	result := NewTensor([]int{m, n})
	bytesPerRow := (k / 256) * 144

	numWorkers := runtime.NumCPU()
	if numWorkers < 1 {
		numWorkers = 1
	}

	for i := 0; i < m; i++ {
		aRow := a.Data[i*k : i*k+k]

		chunk := (n + numWorkers - 1) / numWorkers
		var wg sync.WaitGroup

		for w := 0; w < numWorkers; w++ {
			startJ := w * chunk
			if startJ >= n {
				break
			}
			endJ := startJ + chunk
			if endJ > n {
				endJ = n
			}

			wg.Add(1)
			go func(sj, ej int) {
				defer wg.Done()
				for j := sj; j < ej; j++ {
					offset := j * bytesPerRow
					q4Slice := b.DataQ4[offset : offset+bytesPerRow]
					result.Data[i*n+j] = Dot_Q4_K(q4Slice, aRow)
				}
			}(startJ, endJ)
		}
		wg.Wait()
	}
	return result
}

// float16ToFloat32Inline converts a BF16/FP16 value to float32 inline for the Q4 unpacker.
func float16ToFloat32Inline(h uint16) float32 {
	sign := uint32(h>>15) & 1
	exp := uint32(h>>10) & 0x1F
	mant := uint32(h) & 0x3FF
	if exp == 0 {
		if mant == 0 {
			return math.Float32frombits(sign << 31)
		}
		for mant&0x400 == 0 {
			mant <<= 1
			exp--
		}
		exp++
		mant &= 0x3FF
	} else if exp == 0x1F {
		return math.Float32frombits((sign << 31) | 0x7F800000 | (mant << 13))
	}
	exp = exp + (127 - 15)
	f32bits := (sign << 31) | (exp << 23) | (mant << 13)
	return math.Float32frombits(f32bits)
}

func Dot_Q4_K(q4Buf []byte, x []float32) float32 {
	var sum float32
	nBlocks := len(q4Buf) / 144
	var scales, mins [8]float32

	for b := 0; b < nBlocks; b++ {
		offset := b * 144
		blockBuf := q4Buf[offset : offset+144]

		d := float16ToFloat32Inline(binary.LittleEndian.Uint16(blockBuf[0:2]))
		dmin := float16ToFloat32Inline(binary.LittleEndian.Uint16(blockBuf[2:4]))
		scalesRaw := blockBuf[4:16]
		qs := blockBuf[16:144]

		for subBlock := 0; subBlock < 8; subBlock++ {
			var sc, mn float32
			if subBlock < 4 {
				sc = float32(scalesRaw[subBlock]&0x3F)
				mn = float32(scalesRaw[subBlock+4]&0x3F)
			} else {
				sc = float32((scalesRaw[subBlock+4]&0x0F) | ((scalesRaw[subBlock-4]>>6)<<4))
				mn = float32((scalesRaw[subBlock+4]>>4) | ((scalesRaw[subBlock]>>6)<<4))
			}
			scales[subBlock] = d * sc
			mins[subBlock] = dmin * mn
		}

		base := b * 256
		if base+256 <= len(x) {
			xBlock := x[base : base+256]
			sum += dotQ4BlockAVX2(qs, xBlock, &scales, &mins)
		}
	}
	return sum
}
