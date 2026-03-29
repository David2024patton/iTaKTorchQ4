package main

import (
	"fmt"
	"math/rand"
	"time"
	"github.com/David2024patton/iTaKTorchQ4/pkg/torch/native"
)

// BenchmarkTernary measures the throughput of the Ternary-SIMD kernel.
func BenchmarkTernary(k, n int) {
	fmt.Printf("--- Benchmarking Ternary-SIMD (K=%d, N=%d) ---\n", k, n)
	
	// Create weights (I2_S packed)
	bytesPerRow := (k / 128) * 32
	weightData := make([]byte, n*bytesPerRow)
	rand.Read(weightData)
	
	weight := &native.Tensor{
		Type:   30, // I2_S
		DataQ4: weightData,
		Shape:  []int{n, k},
	}
	
	// Create input vector
	x := native.NewTensor([]int{k})
	for i := range x.Data {
		x.Data[i] = rand.Float32()
	}
	
	// Warmup
	native.MatVecMul(weight, x)
	
	// Benchmark
	iters := 100
	start := time.Now()
	for i := 0; i < iters; i++ {
		native.MatVecMul(weight, x)
	}
	elapsed := time.Since(start)
	
	avg := elapsed / time.Duration(iters)
	tps := float64(iters) / elapsed.Seconds()
	
	// Calculate GFLOPS equivalent (though it's multiplication-free)
	// Each dot product is K elements (Add/Sub).
	// Total ops = iters * N * K
	ops := float64(iters) * float64(n) * float64(k)
	gflops := (ops / elapsed.Seconds()) / 1e9
	
	fmt.Printf("Average Latency: %v\n", avg)
	fmt.Printf("Throughput:      %.2f ops/sec\n", tps)
	fmt.Printf("Effective throughput: %.2f GFLOPS (weight-processing speed)\n", gflops)
}

func main() {
	// Typical layer dimensions for a 1B-3B MoE expert
	// K=2048 (Hidden), N=4096 (FFN Intermediate)
	BenchmarkTernary(2048, 4096)
	
	fmt.Println("\n--- Comparison with Q4_K (Standard) ---")
	// For comparison, we'd need a Q4 benchmark, but we'll focus on Ternary performance here.
}
