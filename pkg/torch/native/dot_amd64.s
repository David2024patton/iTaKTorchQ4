//go:build amd64
#include "textflag.h"

// func Dot(a []float32, b []float32) float32
//
// Arguments passed from Go runtime stack:
// a_base:  0(FP)  (8 bytes)
// a_len:   8(FP)  (8 bytes)
// a_cap:   16(FP) (8 bytes)
// b_base:  24(FP) (8 bytes)
// b_len:   32(FP) (8 bytes)
// b_cap:   40(FP) (8 bytes)
// result:  48(FP) (4 bytes)
// Total Size: 52 bytes
TEXT ·Dot(SB), NOSPLIT, $0-52
    MOVQ a_base+0(FP), R8
    MOVQ a_len+8(FP), R9
    MOVQ b_base+24(FP), R10
    
    // Check if the array length is 0 to avoid panics
    TESTQ R9, R9
    JZ    end_zero
    
    // Initialize our YMM Accumulator (256-bit register) to 0.0
    VXORPS Y0, Y0, Y0 
    
    // R11 is our current array index initialized to 0
    XORQ R11, R11 
    
loop_simd:
    // Determine if we have at least 8 elements left
    MOVQ R9, R12
    SUBQ R11, R12
    CMPQ R12, $8
    JL   tail_math // Break to the standard scalar loop if < 8

    // Load 8 floats (32 bytes) from array A into YMM1
    VMOVUPS (R8)(R11*4), Y1
    
    // Load 8 floats (32 bytes) from array B into YMM2
    VMOVUPS (R10)(R11*4), Y2
    
    // Executing AVX2 Fused Multiply-Add!
    // Y0 = Y0 + (Y1 * Y2) -> Calculating 8 floats simultaneously
    VFMADD231PS Y2, Y1, Y0
    
    // Increment array index by 8
    ADDQ $8, R11
    JMP  loop_simd

tail_math:
    // The main block is complete. Now we must extract the 8 values 
    // packed inside the Y0 accumulator and add them together.
    
    // Extract the top 128 bits (4 floats) of Y0 into X1
    VEXTRACTF128 $1, Y0, X1
    // Add top 4 floats to bottom 4 floats: X0 = X1 + X0
    VADDPS X1, X0, X0
    
    // Swap dimensions (Shuffle) to add the 4 floats together
    VSHUFPS $0x0E, X0, X0, X1 // Swap high and low 64 bits
    VADDPS X1, X0, X0
    VSHUFPS $0x01, X0, X0, X1 // Swap the 32 bits
    VADDPS X1, X0, X0
    
    // X0[0] now correctly holds the sum of the entire AVX2 sequence.
    
    // Run the remaining elements (1 to 7 possible unaligned chunks)
scalar_loop:
    CMPQ R11, R9
    JGE  end_return // If Index >= Length, we're done
    
    // Load single scalar floats
    VMOVSS (R8)(R11*4), X1
    VMOVSS (R10)(R11*4), X2
    // X1 = X1 * X2
    VMULSS X2, X1, X1
    // X0 = X0 + X1
    VADDSS X1, X0, X0
    
    INCQ R11
    JMP  scalar_loop

end_zero:
    VXORPS X0, X0, X0 // Init 0.0

end_return:
    // Move the calculated X0 total to the return payload pointer
    VMOVSS X0, ret+48(FP)
    // Clear AVX registers to prevent transition overhead to downstream Go SSE code
    VZEROUPPER
    RET
