#include "textflag.h"

DATA mask0f<>+0(SB)/8, $0x0f0f0f0f0f0f0f0f
DATA mask0f<>+8(SB)/8, $0x0f0f0f0f0f0f0f0f
GLOBL mask0f<>(SB), RODATA, $16

// func dotQ4BlockAVX2(qs []byte, x []float32, scales *[8]float32, mins *[8]float32) float32
// Stack map:
// qs_base: 0, qs_len: 8, qs_cap: 16
// x_base: 24, x_len: 32, x_cap: 40
// scales_ptr: 48
// mins_ptr: 56
// ret: 64
TEXT ·dotQ4BlockAVX2(SB), NOSPLIT, $0-72
    MOVQ qs_base+0(FP), R8
    MOVQ x_base+24(FP), R9
    MOVQ scales_ptr+48(FP), R10
    MOVQ mins_ptr+56(FP), R11

    VXORPS Y0, Y0, Y0  // accumulator
    VMOVDQU mask0f<>(SB), X15  // Mask for unpacking bytes

    // Subblock loop 0..7
    XORQ R12, R12 // subBlock = 0
loop_sub:
    CMPQ R12, $8
    JGE done

    // Load scale and min for this subblock
    // scales[subBlock] is at R10 + R12*4
    VBROADCASTSS (R10)(R12*4), Y1  // Y1 = scale
    VBROADCASTSS (R11)(R12*4), Y2  // Y2 = min

    // Load 16 bytes of qs (qsBase + subBlock*16)
    MOVQ R12, R13
    SHLQ $4, R13 // R13 = subBlock * 16
    VMOVDQU (R8)(R13*1), X3  // X3 = qs

    // X3 = 16 bytes. Isolate lower and upper nibbles.
    VMOVDQA X3, X4
    VPAND X15, X3, X3       // X3 = lower nibbles
    VPSRLW $4, X4, X4
    VPAND X15, X4, X4       // X4 = upper nibbles

    // Unpack (interleave) to get W0,W1,W2,W3 sequential mapping
    // X5 = W0, W1.. W15 (lower 16 weights)
    VPUNPCKLBW X4, X3, X5
    // X6 = W16, W17.. W31 (upper 16 weights)
    VPUNPCKHBW X4, X3, X6

    // X5 contains 16 8-bit weights. Convert lowest 8 bytes to Y3 (float32)
    VPMOVZXBD X5, Y3
    VCVTDQ2PS Y3, Y3
    
    // Y3 = Y3 * Y1(scale) - Y2(min)
    VMULPS Y1, Y3, Y3
    VSUBPS Y2, Y3, Y3

    // Dot with X! xBase + (subBlock*32 + 0)*4
    MOVQ R12, R14
    SHLQ $7, R14 // R14 = subBlock * 128 (32 floats = 128 bytes)
    VFMADD231PS (R9)(R14*1), Y3, Y0

    // Next 8 weights from X5! (bytes 8..15 of X5)
    VPSHUFD $0xEE, X5, X7
    VPMOVZXBD X7, Y3
    VCVTDQ2PS Y3, Y3
    VMULPS Y1, Y3, Y3
    VSUBPS Y2, Y3, Y3
    // Dot with X + 32 bytes
    ADDQ $32, R14
    VFMADD231PS (R9)(R14*1), Y3, Y0

    // Now X6! (upper 16 weights) -> First 8 of X6
    VPMOVZXBD X6, Y3
    VCVTDQ2PS Y3, Y3
    VMULPS Y1, Y3, Y3
    VSUBPS Y2, Y3, Y3
    ADDQ $32, R14
    VFMADD231PS (R9)(R14*1), Y3, Y0

    // Last 8 of X6
    VPSHUFD $0xEE, X6, X7
    VPMOVZXBD X7, Y3
    VCVTDQ2PS Y3, Y3
    VMULPS Y1, Y3, Y3
    VSUBPS Y2, Y3, Y3
    ADDQ $32, R14
    VFMADD231PS (R9)(R14*1), Y3, Y0

    // Loop
    INCQ R12
    JMP loop_sub

done:
    // Extract Y0 to single float sum!
    VEXTRACTF128 $1, Y0, X1
    VADDPS X1, X0, X0
    VSHUFPS $0x4E, X0, X0, X1
    VADDPS X1, X0, X0
    VSHUFPS $0x11, X0, X0, X1
    VADDPS X1, X0, X0
    VMOVSS X0, ret+64(FP)
    VZEROUPPER
    RET
