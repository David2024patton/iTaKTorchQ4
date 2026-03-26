//go:build !amd64
package native

// dotQ4BlockAVX2 provides a fallback for architectures without AVX2 (like ARM).
// For strict performance parity, ARM Neon implementations should route here.
func dotQ4BlockAVX2(qs []byte, x []float32, scales *[8]float32, mins *[8]float32) float32 {
	var sum float32
	for subBlock := 0; subBlock < 8; subBlock++ {
		sc := scales[subBlock]
		mn := mins[subBlock]
		
		for j := 0; j < 16; j++ {
			q := qs[subBlock*16+j]
			w0 := float32(q&0x0F)*sc - mn
			w1 := float32(q>>4)*sc - mn
			
			idx := subBlock*32 + j*2
			sum += w0*x[idx] + w1*x[idx+1]
		}
	}
	return sum
}
