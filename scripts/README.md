# iTaK Torch Scripts

Organized collection of scripts for benchmarking, testing, and utilities.

## Structure

```
scripts/
  benchmark/          # Performance benchmarking suites
    README.md         # Benchmark documentation
    bench_h2h.py      # Head-to-head iTaK Torch vs Ollama (base template)
    benchmark.ps1     # PowerShell resource-tracking benchmark
    skynet_bench.py   # Lightweight TTFT benchmark for remote Skynet node
```

## Usage

All benchmark scripts are designed to be run from the **repo root** (`iTaKAgent/`).
Models should be present in `models/` (gitignored).

### Quick Start
```bash
# Head-to-head CPU comparison (5 runs, 100 tokens)
python scripts/benchmark/bench_h2h.py --runs 5

# PowerShell resource-tracking benchmark
.\scripts\benchmark\benchmark.ps1 -Engine itaktorch -Mode cpu

# Skynet remote TTFT test (run via SSH)
ssh skynet@192.168.0.217 "cd ~/iTaKAgent && python3 scripts/benchmark/skynet_bench.py"
```

## Creating New Tests

For new test variations:
1. Copy the base script that matches your test type
2. Rename it to reflect the specific test (e.g., `bench_h2h_igpu_override.py`)
3. Modify only the parameters/config needed
4. Keep the original base script untouched

This preserves a clean audit trail for future database ingestion.
