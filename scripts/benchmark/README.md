# Benchmark Scripts

Performance benchmarking suite for iTaK Torch inference engine.

## Scripts

| Script | Language | Platform | Purpose |
|--------|----------|----------|---------|
| `bench_h2h.py` | Python | Any | **Base template.** Head-to-head iTaK Torch vs Ollama comparison. Uses Ollama's native `/api/chat` endpoint for accurate eval metrics. Forces CPU with `num_gpu: 0`. |
| `benchmark.ps1` | PowerShell | Windows | Resource-tracking benchmark. Measures tok/s, CPU%, RAM, GPU%, VRAM. Supports both iTaK Torch and Ollama engines. |
| `skynet_bench.py` | Python | Linux | Lightweight TTFT (Time-To-First-Token) benchmark for the Skynet edge node. Fires uncached + cached requests. |

## Running

All scripts should be run from the **repo root** directory.

### Head-to-Head (Recommended)
```bash
# Default: 5 runs, 100 max tokens, iTaK on 8086, Ollama on 11434
python scripts/benchmark/bench_h2h.py

# Custom ports
python scripts/benchmark/bench_h2h.py --itaktorch-port 8086 --ollama-port 11434 --runs 5
```

### PowerShell Resource Tracker
```powershell
# iTaK Torch CPU
.\scripts\benchmark\benchmark.ps1 -Engine itaktorch -Mode cpu -Port 8086

# Ollama CPU
.\scripts\benchmark\benchmark.ps1 -Engine ollama -Mode cpu

# iTaK Torch GPU (Vulkan)
.\scripts\benchmark\benchmark.ps1 -Engine itaktorch -Mode gpu -Port 8086
```

### Skynet Remote
```bash
# SSH into Skynet and run the TTFT benchmark
ssh skynet@192.168.0.217 "cd ~/iTaKAgent && python3 scripts/benchmark/skynet_bench.py"
```

## Creating Variant Tests

When testing a new feature or configuration:
1. **Copy** the base script (e.g., `bench_h2h.py`)
2. **Rename** to reflect the test: `bench_h2h_phase7_zerocgo.py`
3. **Modify** only what's needed for the specific test
4. **Never modify** the base template scripts

This keeps the base scripts stable and creates a record of each test variant for future database ingestion.

## Models

Models are stored in `models/` at the repo root (gitignored). Currently used:
- `qwen2.5-0.5b-instruct-q4_k_m.gguf` (468MB) - Standard benchmark model

Models must be present on both Beast and Skynet. Transfer via:
```bash
scp models/qwen2.5-0.5b-instruct-q4_k_m.gguf skynet@192.168.0.217:~/iTaKAgent/models/
```
