package main

import (
	"fmt"
	"math/rand"
	"time"
	"github.com/David2024patton/iTaKTorchQ4/pkg/torch/native"
)

func main() {
	fmt.Println("=== Ternary-SIMD GOTensor Test ===")
	
	// 1. Create a dummy i2_s block (128 elements, 32 bytes)
	// Byte 0: Bit 7-6 (Elem 0), Bit 5-4 (Elem 32), Bit 3-2 (Elem 64), Bit 1-0 (Elem 96)
	// Bits: 00=0, 01=1, 10=-1
	block := make([]byte, 32)
	for i := range block {
		// Pack random ternary values
		// 0: (00), 1: (01), 2: (10)
		v0 := byte(rand.Intn(3)) << 6
		v1 := byte(rand.Intn(3)) << 4
		v2 := byte(rand.Intn(3)) << 2
		v3 := byte(rand.Intn(3))
		block[i] = v0 | v1 | v2 | v3
	}
	
	// 2. Unpack to verify mapping
	weights := native.UnpackI2S(block)
	fmt.Printf("Unpacked first 4 elements: %v %v %v %v\n", weights[0], weights[32], weights[64], weights[96])
	
	// 3. Create input vector x
	x := make([]float32, 128)
	for i := range x {
		x[i] = rand.Float32()
	}
	
	// 4. Compute with TernaryDot (float path) and TernaryDotBlock (bit path)
	scale := float32(0.5)
	
	start := time.Now()
	resFloat := native.TernaryDot(x, weights, scale)
	durFloat := time.Since(start)
	
	start = time.Now()
	resBit := native.TernaryDotBlock(x, block, scale)
	durBit := time.Since(start)
	
	fmt.Printf("Result (Float Path): %f (%v)\n", resFloat, durFloat)
	fmt.Printf("Result (Bit Path):   %f (%v)\n", resBit, durBit)
	
	if fmt.Sprintf("%.4f", resFloat) == fmt.Sprintf("%.4f", resBit) {
		fmt.Println("SUCCESS: Bit path matches float path!")
	} else {
		fmt.Println("FAILURE: Mismatch detected!")
	}
}
