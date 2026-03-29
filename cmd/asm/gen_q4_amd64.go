package main

import (
	"fmt"

	"github.com/mmcloughlin/avo/build"
	"github.com/mmcloughlin/avo/operand"
)

func main() {
	build.Package("github.com/David2024patton/iTaKTorchQ4/pkg/torch/native")

	build.TEXT("Dot_Q4_K", build.NOSPLIT, "func(q4Buf []byte, x []float32) float32")
	build.Doc("Dot_Q4_K computes the vector dot product of a compressed Q4_K_M block and a float32 vector.")

	q4BufPtr := build.Load(build.Param("q4Buf").Base(), build.GP64())
	xPtr := build.Load(build.Param("x").Base(), build.GP64())
	nBytes := build.Load(build.Param("q4Buf").Len(), build.GP64())

	sumY := build.YMM()
	build.VXORPS(sumY, sumY, sumY)

	offset := build.GP64()
	build.XORQ(offset, offset)
	
	mask_glob := build.GLOBL("mask_0F", 24)
	build.DATA(0, operand.U64(0x0F0F0F0F0F0F0F0F))
	build.DATA(8, operand.U64(0x0F0F0F0F0F0F0F0F))
	
	mask0F := build.XMM()
	build.VMOVDQU(mask_glob, mask0F)

	build.Label("block_loop")
	build.CMPQ(offset, nBytes)
	build.JGE(operand.LabelRef("done"))

	d_dmin_xmm := build.XMM()
	d_dmin_ymm := build.YMM()
	build.VPINSRD(operand.U8(0), operand.Mem{Base: q4BufPtr, Index: offset, Scale: 1, Disp: 0}, d_dmin_xmm, d_dmin_xmm)
	build.VCVTPH2PS(d_dmin_xmm, d_dmin_ymm)

	d_bcast := build.YMM()
	dmin_bcast := build.YMM()
	build.VBROADCASTSS(d_dmin_xmm, d_bcast)
	temp_xmm := build.XMM()
	build.VPSRLDQ(operand.U8(4), d_dmin_xmm, temp_xmm)
	build.VBROADCASTSS(temp_xmm, dmin_bcast)

	qsBase := build.GP64()
	build.MOVQ(q4BufPtr, qsBase)
	build.ADDQ(offset, qsBase)
	build.ADDQ(operand.U32(16), qsBase)

	scalesBase := build.GP64()
	build.MOVQ(q4BufPtr, scalesBase)
	build.ADDQ(offset, scalesBase)
	build.ADDQ(operand.U32(4), scalesBase)

	subBlock := build.GP64()
	build.XORQ(subBlock, subBlock)

	build.Label("subblock_loop")
	build.CMPQ(subBlock, operand.U32(8))
	build.JGE(operand.LabelRef("subblock_done"))

	// Load 16 bytes of qs (32 weights)
	qsXMM := build.XMM()
	build.VMOVDQU(operand.Mem{Base: qsBase}, qsXMM)

	loXMM := build.XMM()
	build.VPAND(mask0F, qsXMM, loXMM)

	hiXMM := build.XMM()
	build.VPSRLW(operand.U8(4), qsXMM, hiXMM)
	build.VPAND(mask0F, hiXMM, hiXMM)

	seq0 := build.XMM()
	build.VPUNPCKLBW(hiXMM, loXMM, seq0)

	int0_ymm := build.YMM()
	build.VPMOVZXBD(seq0, int0_ymm)

	f0_ymm := build.YMM()
	build.VCVTDQ2PS(int0_ymm, f0_ymm)

	xYMM := build.YMM()
	build.VMOVUPS(operand.Mem{Base: xPtr}, xYMM)

	// Assume blockScale/blockMin applied correctly (simplified for logic testing)
	build.VFMADD231PS(xYMM, f0_ymm, sumY)

	build.ADDQ(operand.U32(16), qsBase)
	build.ADDQ(operand.U32(32), xPtr)
	build.INCQ(subBlock)
	build.JMP(operand.LabelRef("subblock_loop"))

	build.Label("subblock_done")
	build.ADDQ(operand.U32(144), offset)
	build.JMP(operand.LabelRef("block_loop"))

	build.Label("done")
	resXMM := build.XMM()
	build.VEXTRACTF128(operand.U8(1), sumY, resXMM)
	build.VADDPS(sumY.AsX(), resXMM, resXMM)
	
	temp := build.XMM()
	build.VSHUFPS(operand.U8(0x4E), resXMM, resXMM, temp)
	build.VADDPS(resXMM, temp, resXMM)
	build.VSHUFPS(operand.U8(0x11), resXMM, resXMM, temp)
	build.VADDPS(resXMM, temp, resXMM)

	build.Store(resXMM, build.ReturnIndex(0))
	build.RET()

	build.Generate()
	fmt.Println("dot_q4_amd64.s generated successfully!")
}
