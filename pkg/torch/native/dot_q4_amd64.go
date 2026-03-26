//go:build amd64
package native

// dotQ4BlockAVX2 performs AVX2 hardware unzipping and multiplication of a contiguous 256-weight chunk.
func dotQ4BlockAVX2(qs []byte, x []float32, scales *[8]float32, mins *[8]float32) float32
