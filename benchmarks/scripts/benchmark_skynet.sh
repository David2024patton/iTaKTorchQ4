#!/usr/bin/env bash
# iTaK Torch Benchmark Suite - Skynet (Linux)
# Runs benchmarks on the Skynet mini PC (i7-8700T, Ubuntu)
#
# Usage: bash benchmark_skynet.sh [category]
# Categories: boot, ttft, throughput, resources, threads, unit, all

set -euo pipefail

CATEGORY="${1:-all}"
MODEL_PATH="${GGUF_MODEL:-$HOME/models/qwen2.5-0.5b-instruct-q4_k_m.gguf}"
OLLAMA_MODEL="${OLLAMA_MODEL:-qwen2.5:0.5b}"
ITERATIONS="${ITERATIONS:-3}"
TIMESTAMP=$(date +%Y-%m-%d_%H%M%S)
OUTPUT_DIR="${OUTPUT_DIR:-$HOME/itak-torch-benchmark/results}"
PROJECT_DIR="${PROJECT_DIR:-$HOME/GOAgent}"
REPORT_FILE="${OUTPUT_DIR}/benchmark_${TIMESTAMP}_skynet.md"

# Auto-set LD_LIBRARY_PATH so GGML backend .so files are found at runtime.
if [ -n "${ITAK_TORCH_LIB:-}" ]; then
    export LD_LIBRARY_PATH="${ITAK_TORCH_LIB}:${LD_LIBRARY_PATH:-}"
fi

mkdir -p "$OUTPUT_DIR"

TEST_PROMPT="Explain the concept of transformer attention mechanisms in neural networks. Include details about query, key, and value matrices."

# ============================================================
# Helpers
# ============================================================

write_report() {
    echo "$1" >> "$REPORT_FILE"
}

get_cpu_info() {
    local model=$(grep "model name" /proc/cpuinfo | head -1 | cut -d: -f2 | xargs)
    local cores=$(grep "cpu cores" /proc/cpuinfo | head -1 | cut -d: -f2 | xargs)
    local threads=$(nproc)
    local ram=$(free -g | awk '/Mem:/{print $2}')
    echo "${model}|${cores}|${threads}|${ram}GB"
}

get_gpu_info() {
    # Check for intel iGPU.
    if command -v intel_gpu_top &>/dev/null; then
        echo "Intel UHD (integrated)"
    elif command -v nvidia-smi &>/dev/null; then
        nvidia-smi --query-gpu=name --format=csv,noheader,nounits 2>/dev/null || echo "None"
    else
        echo "None"
    fi
}

get_ram_usage_mb() {
    free -m | awk '/Mem:/{print $3}'
}

run_ollama_inference() {
    local model="$1"
    local prompt="$2"

    # Run with verbose flag and capture timing.
    local start_ms=$(date +%s%N)
    local output
    output=$(echo "$prompt" | ollama run "$model" --verbose 2>&1)
    local end_ms=$(date +%s%N)

    local elapsed_ms=$(( (end_ms - start_ms) / 1000000 ))

    # Parse tok/s from verbose output.
    local tok_per_sec=$(echo "$output" | grep -oP 'eval rate:\s*\K[\d.]+' || echo "0")
    local eval_count=$(echo "$output" | grep -oP 'eval count:\s*\K\d+' || echo "0")

    echo "${tok_per_sec}|${eval_count}|${elapsed_ms}"
}

# ============================================================
# Report Header
# ============================================================

write_header() {
    IFS='|' read -r cpu cores threads ram <<< "$(get_cpu_info)"
    local gpu=$(get_gpu_info)

    write_report "# iTaK Torch Benchmark Report - Skynet"
    write_report ""
    write_report "**Date:** $(date '+%Y-%m-%d %H:%M:%S')"
    write_report "**Machine:** Skynet (Ubuntu)"
    write_report "**CPU:** $cpu"
    write_report "**Cores/Threads:** $cores/$threads"
    write_report "**RAM:** $ram"
    write_report "**GPU:** $gpu"
    write_report "**Model:** $(basename "$MODEL_PATH")"
    write_report "**Ollama Model:** $OLLAMA_MODEL"
    write_report ""
    write_report "---"
    write_report ""
}

# ============================================================
# Cat 7: Unit Tests
# ============================================================

run_unit_tests() {
    echo -e "\n=== Category 7: Unit Tests ==="

    write_report "## Category 7: Unit Tests"
    write_report ""

    if [ -d "$PROJECT_DIR" ]; then
        local result
        result=$(cd "$PROJECT_DIR" && go test ./pkg/torch/ -v -count=1 -short 2>&1) || true
        local passed=$(echo "$result" | grep -c "PASS" || true)
        local failed=$(echo "$result" | grep -c "FAIL" || true)

        write_report "| Package | Passed | Failed |"
        write_report "|---------|--------|--------|"
        write_report "| pkg/torch/ | $passed | $failed |"
        write_report ""

        if [ "$failed" -gt 0 ]; then
            write_report "> [!CAUTION]"
            write_report "> Unit tests failed."
            echo "WARNING: Unit tests failed!"
        else
            write_report "> [!NOTE]"
            write_report "> All unit tests passed."
            echo "All unit tests passed."
        fi
    else
        write_report "Project directory not found at $PROJECT_DIR"
    fi
    write_report ""
}

# ============================================================
# Cat 1: Cold Boot Time
# ============================================================

run_boot_time() {
    echo -e "\n=== Category 1: Cold Boot Time ==="

    write_report "## Category 1: Cold Boot Time"
    write_report ""

    # Stop Ollama model to get cold boot.
    ollama stop "$OLLAMA_MODEL" 2>/dev/null || true
    sleep 2

    local ram_before=$(get_ram_usage_mb)
    local start_ms=$(date +%s%N)
    echo "hi" | ollama run "$OLLAMA_MODEL" >/dev/null 2>&1
    local end_ms=$(date +%s%N)
    local boot_ms=$(( (end_ms - start_ms) / 1000000 ))
    local ram_after=$(get_ram_usage_mb)

    write_report "| Engine | Boot Time (ms) | RAM Before (MB) | RAM After (MB) |"
    write_report "|--------|---------------|----------------|----------------|"
    write_report "| Ollama ($OLLAMA_MODEL) | $boot_ms | $ram_before | $ram_after |"
    write_report ""
}

# ============================================================
# Cat 3: Generation Throughput
# ============================================================

run_throughput() {
    echo -e "\n=== Category 3: Generation Throughput ==="

    write_report "## Category 3: Generation Throughput (tok/s)"
    write_report ""

    # Ollama tests.
    echo "Running Ollama ($OLLAMA_MODEL) x$ITERATIONS..."
    write_report "### Ollama ($OLLAMA_MODEL)"
    write_report ""
    write_report "| Run | tok/s | Tokens | Time (ms) |"
    write_report "|-----|-------|--------|-----------|"

    local sum=0
    for i in $(seq 1 "$ITERATIONS"); do
        echo "  Iteration $i/$ITERATIONS..."
        IFS='|' read -r tps tokens elapsed <<< "$(run_ollama_inference "$OLLAMA_MODEL" "$TEST_PROMPT")"
        write_report "| $i | $tps | $tokens | $elapsed |"
        sum=$(echo "$sum + $tps" | bc)
    done
    local mean=$(echo "scale=1; $sum / $ITERATIONS" | bc)
    write_report "| **Mean** | **$mean** | - | - |"
    write_report ""

    # Go benchmark (if project exists and lib is available).
    if [ -d "$PROJECT_DIR" ] && [ -n "${ITAK_TORCH_LIB:-}" ]; then
        echo "Running iTaK Torch Go benchmark..."
        write_report "### iTaK Torch (Go Benchmark)"
        write_report ""
        write_report '```'
        local bench_result
        bench_result=$(cd "$PROJECT_DIR" && YZMA_BENCHMARK_MODEL="$MODEL_PATH" go test ./pkg/torch/llama/ -bench=BenchmarkInference -benchtime=10s -count="$ITERATIONS" 2>&1) || true
        write_report "$bench_result"
        write_report '```'
        write_report ""
    else
        write_report "### iTaK Torch (Go Benchmark)"
        write_report ""
        write_report "Skipped: ITAK_TORCH_LIB not set or project not found."
        write_report ""
    fi
}

# ============================================================
# Cat 4: Resource Monitoring
# ============================================================

run_resource_monitoring() {
    echo -e "\n=== Category 4: Resource Monitoring ==="

    write_report "## Category 4: System Resources During Inference"
    write_report ""

    local ram_before=$(get_ram_usage_mb)

    write_report "### Baseline (idle)"
    write_report ""
    write_report "| Metric | Value |"
    write_report "|--------|-------|"
    write_report "| System RAM Used | ${ram_before} MB |"
    write_report ""

    # Monitor during inference.
    local monitor_file="/tmp/bench_monitor_$$.log"

    # Start background monitor.
    (
        for i in $(seq 1 60); do
            local cpu_pct=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}')
            local ram_used=$(free -m | awk '/Mem:/{print $3}')
            echo "$(date +%H:%M:%S.%N)|$cpu_pct|$ram_used" >> "$monitor_file"
            sleep 0.5
        done
    ) &
    local monitor_pid=$!

    # Run inference.
    IFS='|' read -r tps tokens elapsed <<< "$(run_ollama_inference "$OLLAMA_MODEL" "$TEST_PROMPT")"

    sleep 2
    kill "$monitor_pid" 2>/dev/null || true
    wait "$monitor_pid" 2>/dev/null || true

    if [ -f "$monitor_file" ]; then
        local avg_cpu=$(awk -F'|' '{sum+=$2; n++} END{printf "%.1f", sum/n}' "$monitor_file")
        local peak_cpu=$(awk -F'|' 'BEGIN{max=0}{if($2>max)max=$2}END{printf "%.1f", max}' "$monitor_file")
        local peak_ram=$(awk -F'|' 'BEGIN{max=0}{if($3>max)max=$3}END{print max}' "$monitor_file")

        write_report "### During Ollama Inference"
        write_report ""
        write_report "| Metric | Value |"
        write_report "|--------|-------|"
        write_report "| Mean CPU % | ${avg_cpu}% |"
        write_report "| Peak CPU % | ${peak_cpu}% |"
        write_report "| Peak RAM | ${peak_ram} MB |"
        write_report "| Generation tok/s | $tps |"
        write_report ""

        rm -f "$monitor_file"
    fi
}

# ============================================================
# Cat 5: Thread Scaling
# ============================================================

run_thread_scaling() {
    echo -e "\n=== Category 5: Thread Scaling ==="

    write_report "## Category 5: Thread Scaling"
    write_report ""

    if [ -d "$PROJECT_DIR" ]; then
        local result
        result=$(cd "$PROJECT_DIR" && go test ./pkg/torch/ -run TestDetectOptimal -v -count=1 2>&1) || true
        write_report '```'
        write_report "$result"
        write_report '```'
    else
        write_report "Skipped: project not found."
    fi
    write_report ""
}

# ============================================================
# Main
# ============================================================

echo "================================================"
echo "  iTaK Torch Benchmark Suite - Skynet"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================"

write_header

case "$CATEGORY" in
    all)
        run_unit_tests
        run_boot_time
        run_throughput
        run_resource_monitoring
        run_thread_scaling
        ;;
    unit)       run_unit_tests ;;
    boot)       run_boot_time ;;
    throughput) run_throughput ;;
    resources)  run_resource_monitoring ;;
    threads)    run_thread_scaling ;;
    *)
        echo "Unknown category: $CATEGORY"
        echo "Valid: boot, ttft, throughput, resources, threads, unit, all"
        exit 1
        ;;
esac

write_report "---"
write_report ""
write_report "*Report generated by iTaK Torch Benchmark Skill v1.0*"

echo ""
echo "================================================"
echo "  Benchmark complete!"
echo "  Report: $REPORT_FILE"
echo "================================================"
