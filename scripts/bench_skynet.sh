#!/bin/bash
set -e

export PATH=/usr/local/go/bin:/usr/bin:/bin:$HOME/go/bin:$PATH
export GOTOOLCHAIN=local
export LD_LIBRARY_PATH=/home/skynet/torch_lib
export ITAK_TORCH_LIB=/home/skynet/torch_lib
export YZMA_BENCHMARK_MODEL=/home/skynet/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf

TORCH_SRC=/home/skynet/torch_src
OUT_DIR=/home/skynet/bench_results
mkdir -p "$OUT_DIR"

run_bench() {
    local name=$1
    local nthreads=$2

    echo "=== BENCHMARK: $name (threads=$nthreads) ==="

    # Pre-benchmark resources
    pre_ram=$(free -m | awk '/Mem:/ {print $3}')
    pre_ram_total=$(free -m | awk '/Mem:/ {print $2}')
    echo "  [BEFORE] RAM: ${pre_ram}MB / ${pre_ram_total}MB"

    # Resource monitor
    local mon_file="$OUT_DIR/skynet_${name}_monitor.csv"
    echo "time,ram_used_mb,cpu_pct" > "$mon_file"
    (
        while true; do
            ram=$(free -m | awk '/Mem:/ {print $3}')
            cpu=$(top -bn1 | grep "Cpu(s)" | awk '{printf "%.1f", $2+$4}')
            echo "$(date +%s),$ram,$cpu" >> "$mon_file"
            sleep 3
        done
    ) &
    MON_PID=$!

    # Build and run
    cd "$TORCH_SRC"
    local bench_args="-run=^\$ -bench=BenchmarkInference -benchtime=30s -count=1 -timeout=300s -nctx=4096 -ngpulayers=0"
    if [ "$nthreads" -gt 0 ]; then
        bench_args="$bench_args -nthreads=$nthreads"
    fi

    local result
    result=$(go test ./pkg/torch/llama/ $bench_args 2>&1) || true

    # Stop monitor
    kill $MON_PID 2>/dev/null || true
    wait $MON_PID 2>/dev/null || true

    # Post-benchmark resources
    post_ram=$(free -m | awk '/Mem:/ {print $3}')
    echo "  [AFTER] RAM: ${post_ram}MB / ${pre_ram_total}MB"

    # Parse monitor
    local samples avg_ram peak_ram avg_cpu
    samples=$(wc -l < "$mon_file")
    samples=$((samples - 1))
    if [ "$samples" -gt 0 ]; then
        avg_ram=$(awk -F',' 'NR>1 {sum+=$2; n++} END {if(n>0) printf "%.0f", sum/n; else print "0"}' "$mon_file")
        peak_ram=$(awk -F',' 'NR>1 {if($2>max) max=$2} END {print max+0}' "$mon_file")
        avg_cpu=$(awk -F',' 'NR>1 {sum+=$3; n++} END {if(n>0) printf "%.1f", sum/n; else print "0"}' "$mon_file")
    else
        avg_ram=0; peak_ram=0; avg_cpu=0
    fi

    # Write report
    cat > "$OUT_DIR/skynet_${name}.txt" <<REPORT
=== TORCH BENCHMARK: $name ===
Machine: Skynet (Dell Mini PC, Ubuntu)
CPU: $(grep 'model name' /proc/cpuinfo | head -1 | cut -d':' -f2 | xargs) ($(nproc) threads)
GPU: Intel UHD 630 (iGPU, NO CUDA)
RAM: $(free -h | awk '/Mem:/ {print $2}')
Model: Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf (~4.9GB)
Library: $ITAK_TORCH_LIB
GPU Layers: 0 (CPU only)
Threads: $nthreads (0 = auto)
Context: 4096

--- BENCHMARK OUTPUT ---
$result

--- RESOURCES BEFORE ---
RAM: ${pre_ram}MB / ${pre_ram_total}MB

--- RESOURCES AFTER ---
RAM: ${post_ram}MB / ${pre_ram_total}MB

--- RESOURCES DURING (sampled every 3s) ---
Avg RAM Used:   ${avg_ram} MB
Peak RAM Used:  ${peak_ram} MB
Avg CPU Usage:  ${avg_cpu}%
Samples:        $samples
REPORT

    echo "$result"
    echo "  Resources: RAM avg=${avg_ram}MB peak=${peak_ram}MB | CPU avg=${avg_cpu}% ($samples samples)"
    echo ""
}

echo ""
echo "===================================================="
echo "  TORCH BENCHMARK SUITE - SKYNET"
echo "  Model: Llama 3.1 8B Q4_K_M (~4.9GB)"
echo "  CPU: $(grep 'model name' /proc/cpuinfo | head -1 | cut -d':' -f2 | xargs)"
echo "  Cores: $(nproc)"
echo "===================================================="
echo ""

# Test 1: All threads (auto)
run_bench "cpu_auto" 0

# Test 2: Physical cores only (6 for i7-8700T)
run_bench "cpu_6threads" 6

# Test 3: Half cores
run_bench "cpu_3threads" 3

echo "=== ALL SKYNET BENCHMARKS COMPLETE ==="
echo "Results in: $OUT_DIR"
