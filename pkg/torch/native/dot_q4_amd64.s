#include "textflag.h"

DATA mask0f<>+0(SB)/8, $0x0f0f0f0f0f0f0f0f
DATA mask0f<>+8(SB)/8, $0x0f0f0f0f0f0f0f0f
GLOBL mask0f<>(SB), RODATA, $16

// func dotQ4BlockAVX2(blockBuf []byte, x []float32) float32
// Stack map:
// blockBuf_base: 0, blockBuf_len: 8, blockBuf_cap: 16
// x_base: 24, x_len: 32, x_cap: 40
// ret: 48
TEXT ·dotQ4BlockAVX2(SB), NOSPLIT, $0-56
    MOVQ blockBuf_base+0(FP), R8
    MOVQ x_base+24(FP), R9

    VXORPS Y0, Y0, Y0  // accumulator
    VMOVDQU mask0f<>(SB), X15  // Mask for unpacking bytes

    // Pre-calculate d and dmin from Float16 bytes 0:4
    // R8 points to blockBuf. First 4 bytes are d(2) and dmin(2).
    VMOVD (R8), X0
    VCVTPH2PS X0, X1
    // X1 now holds `d` in the 0th float, `dmin` in the 1st float.
    
    // Broadcast dmin to X2's 0th float for easy multiplication
    VPSHUFD $0x55, X1, X2

    LEAQ 4(R8), R10 // R10 points to scalesRaw (bytes 4..15)

    XORQ R12, R12 // subBlock = 0
loop_sub:
    CMPQ R12, $8
    JGE done

    CMPQ R12, $4
    JGE upper_half

    // Lower half logic (R12 < 4)
    MOVBLZX (R10)(R12*1), AX       // scalesRaw[subBlock]
    MOVQ AX, BX
    ANDQ $0x3F, BX                 // sc

    MOVBLZX 4(R10)(R12*1), CX      // scalesRaw[subBlock+4]
    ANDQ $0x3F, CX                 // mn
    JMP calculate_floats

upper_half:
    // R12 >= 4. R13 = R12 - 4
    MOVQ R12, R13
    SUBQ $4, R13

    // sc = (scalesRaw[R12] & 0x0F) | ((scalesRaw[R13] >> 6) << 4)
    MOVBLZX (R10)(R12*1), AX
    MOVQ AX, BX
    ANDQ $0x0F, BX

    MOVBLZX (R10)(R13*1), CX
    SHRQ $6, CX
    SHLQ $4, CX
    ORQ CX, BX                     // BX = sc

    // mn = (scalesRaw[R12] >> 4) | ((scalesRaw[R13] >> 6) << 4)
    MOVQ AX, CX                    // AX is scalesRaw[R12]
    SHRQ $4, CX
    
    MOVBLZX (R10)(R13*1), DX
    SHRQ $6, DX
    SHLQ $4, DX
    ORQ DX, CX                     // CX = mn

calculate_floats:
    // BX = sc, CX = mn
    MOVQ BX, X8
    VCVTDQ2PS X8, X8
    VMULSS X1, X8, X8  // sc * d
    VBROADCASTSS X8, Y14 // Y14 = blockScale

    MOVQ CX, X9
    VCVTDQ2PS X9, X9
    VMULSS X2, X9, X9  // mn * dmin
    VBROADCASTSS X9, Y15 // Y15 = blockMin

    // Load 16 bytes of qs (R8+16 + subBlock*16)
    MOVQ R12, R13
    SHLQ $4, R13 // R13 = subBlock * 16
    ADDQ $16, R13 // Offset of qs is 16
    VMOVDQU (R8)(R13*1), X3  // X3 = qs chunk

    // Isolate nibbles
    VMOVDQA X3, X4
    VPAND X15, X3, X3       // X3 = lower nibbles
    VPSRLW $4, X4, X4
    VPAND X15, X4, X4       // X4 = upper nibbles

    // Interleave
    VPUNPCKLBW X4, X3, X5   // Lower 16 weights
    VPUNPCKHBW X4, X3, X6   // Upper 16 weights

    // Convert and Accumulate Lower 16 weights
    VPMOVZXBD X5, Y3
    VCVTDQ2PS Y3, Y3
    VMULPS Y14, Y3, Y3
    VSUBPS Y15, Y3, Y3

    // Dot with X! xBase + (subBlock*32 + 0)*4
    MOVQ R12, R14
    SHLQ $7, R14 // R14 = subBlock * 128 (32 floats)
    VFMADD231PS (R9)(R14*1), Y3, Y0

    VPSHUFD $0xEE, X5, X7
    VPMOVZXBD X7, Y3
    VCVTDQ2PS Y3, Y3
    VMULPS Y14, Y3, Y3
    VSUBPS Y15, Y3, Y3
    ADDQ $32, R14
    VFMADD231PS (R9)(R14*1), Y3, Y0

    // Convert and Accumulate Upper 16 weights
    VPMOVZXBD X6, Y3
    VCVTDQ2PS Y3, Y3
    VMULPS Y14, Y3, Y3
    VSUBPS Y15, Y3, Y3
    ADDQ $32, R14
    VFMADD231PS (R9)(R14*1), Y3, Y0

    VPSHUFD $0xEE, X6, X7
    VPMOVZXBD X7, Y3
    VCVTDQ2PS Y3, Y3
    VMULPS Y14, Y3, Y3
    VSUBPS Y15, Y3, Y3
    ADDQ $32, R14
    VFMADD231PS (R9)(R14*1), Y3, Y0

    INCQ R12
    JMP loop_sub

done:
    VEXTRACTF128 $1, Y0, X1
    VADDPS X1, X0, X0
    VSHUFPS $0x4E, X0, X0, X1
    VADDPS X1, X0, X0
    VSHUFPS $0x11, X0, X0, X1
    VADDPS X1, X0, X0
    VMOVSS X0, ret+48(FP)
    VZEROUPPER
    RET
