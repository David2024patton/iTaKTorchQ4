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
	"fmt"
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
