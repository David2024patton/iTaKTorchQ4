#!/bin/bash
export PATH=/usr/local/go/bin:$HOME/go/bin:$PATH
export ITAK_TORCH_LIB="/mnt/e/.agent/iTaK Eco/Torch/lib/linux_amd64"
export YZMA_TEST_MODEL="/mnt/e/.agent/GOAgent/models/qwen2.5-0.5b-instruct-q4_k_m.gguf"
OUTFILE="/mnt/e/.agent/iTaK Eco/Torch/scripts/wsl2_results.txt"

cd "/mnt/e/.agent/iTaK Eco/Torch"

{
    echo "=== WSL2 CRASH TEST ==="
    echo "Go: $(go version)"
    echo "Lib: $ITAK_TORCH_LIB"
    echo ""

    echo "--- go vet ---"
    go vet ./... 2>&1
    echo "VET_EXIT=$?"
    echo ""

    echo "--- TestBackendInit ---"
    go test ./pkg/torch/llama/ -run TestBackendInit -v -count=1 -timeout=30s 2>&1
    echo "BACKEND_EXIT=$?"
    echo ""

    echo "--- TestGGML ---"
    go test ./pkg/torch/llama/ -run TestGGML -v -count=1 -timeout=30s 2>&1
    echo "GGML_EXIT=$?"
    echo ""

    echo "=== DONE ==="
} > "$OUTFILE" 2>&1
