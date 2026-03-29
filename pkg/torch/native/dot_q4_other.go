//go:build !amd64
package native
import "encoding/binary"

// dotQ4BlockAVX2 provides a fallback for architectures without AVX2 (like ARM).
func dotQ4BlockAVX2(blockBuf []byte, x []float32) float32 {
	var sum float32
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

		blockScale := d * sc
		blockMin := dmin * mn
		qsBlock := qs[subBlock*16 : subBlock*16+16]

		wbBase := subBlock * 32
		for j := 0; j < 16; j++ {
			q := qsBlock[j]
			w0 := float32(q&0x0F)*blockScale - blockMin
			w1 := float32(q>>4)*blockScale - blockMin
			idx := wbBase + j*2
			sum += w0*x[idx] + w1*x[idx+1]
		}
	}
	return sum
}
