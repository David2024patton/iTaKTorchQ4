#!/bin/bash
export PATH=/usr/local/go/bin:$HOME/go/bin:$PATH
export ITAK_TORCH_LIB="/mnt/e/.agent/iTaK Eco/Torch/lib/linux_amd64"
export YZMA_BENCHMARK_MODEL="/mnt/e/.agent/GOAgent/models/qwen2.5-0.5b-instruct-q4_k_m.gguf"
OUTFILE="/mnt/e/.agent/iTaK Eco/Torch/scripts/wsl2_bench_results.txt"

cd "/mnt/e/.agent/iTaK Eco/Torch"

{
    echo "=== WSL2 BENCHMARK ==="
    echo "Go: $(go version)"
    echo "Lib: $ITAK_TORCH_LIB"
    echo "Model: $YZMA_BENCHMARK_MODEL"
    echo ""

    echo "--- BenchmarkInference ---"
    go test ./pkg/torch/llama/ -run='^$' -bench=BenchmarkInference -benchtime=10s -count=1 -timeout=120s 2>&1
    echo "BENCH_EXIT=$?"
    echo ""

    echo "=== DONE ==="
} > "$OUTFILE" 2>&1
