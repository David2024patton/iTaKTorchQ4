# iTaK Torch Benchmarks

Comprehensive benchmarking suite for the iTaK Torch inference engine. Measures throughput (tok/s) and system resources (VRAM, GPU power, temperature, RAM) across multiple engines and hardware backends.

## Engines Tested

### GPU Engines
| Engine | Description |
|--------|-------------|
| **Ollama GPU** | Ollama with default GPU acceleration |
| **iTaK Torch Vulkan** | iTaK Torch using Vulkan dGPU backend |
| **iTaK Torch CUDA** | iTaK Torch using CUDA dGPU backend |

### CPU Engines
| Engine | Description | How CPU is Forced |
|--------|-------------|-------------------|
| **Ollama CPU** | Ollama forced to CPU | REST API with `num_gpu: 0` |
| **iTaK Torch CPU** | iTaK Torch CPU-only backend | `lib/windows_amd64/` (no GPU libs) |

> **Note on Ollama CPU**: Setting `CUDA_VISIBLE_DEVICES=""` does NOT work because Ollama runs as a background service. The benchmark uses the `/api/generate` endpoint with `"num_gpu": 0` to force zero GPU layers for true CPU inference.

## Metrics Captured Per Engine

Every engine gets the same treatment:
- **tok/s**: Mean, min, max across N iterations (default 3)
- **VRAM**: Before/after snapshots (MB)
- **GPU Temperature**: Before/after (C)
- **GPU Power Draw**: Before/after (W)
- **System RAM**: Before/after (GB)

## Machines

### Beast (Windows)
- **CPU**: Intel i9-14900K (24C/32T)
- **RAM**: 128GB DDR5
- **GPU**: NVIDIA RTX 4070 Ti SUPER (16GB VRAM)
- Tests all GPU + CPU engines

### Skynet (Ubuntu)
- **CPU**: Intel i7-8700T (6C/12T)
- **RAM**: 32GB DDR4
- **GPU**: Intel UHD 630 (iGPU only)
- Tests CPU + iGPU engines only

## Quick Start

### Beast

```powershell
# Full engine matrix (GPU + CPU)
.\benchmarks\scripts\benchmark_beast.ps1 -Category matrix

# Everything (unit tests + boot + matrix + thread scaling)
.\benchmarks\scripts\benchmark_beast.ps1 -Category all

# Just unit tests
.\benchmarks\scripts\benchmark_beast.ps1 -Category unit
```

### Skynet

```bash
ssh skynet@192.168.0.217
cd ~/iTaK-Torch
git pull
bash benchmarks/scripts/benchmark_skynet.sh all
```

## Script Parameters (Beast)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `-Category` | `all` | `boot`, `matrix`, `throughput`, `resources`, `threads`, `unit`, `all` |
| `-Model` | `models/qwen2.5-0.5b-instruct-q4_k_m.gguf` | GGUF model for iTaK Torch |
| `-OllamaModel` | `qwen2.5:0.5b` | Ollama model tag |
| `-Iterations` | `3` | Runs per engine |
| `-OutputDir` | `benchmarks/results/` | Report output directory |
| `-ProjectDir` | `E:\.agent\GOAgent` | iTaK Torch Go project |

## Environment Variables

| Variable | Set By | Purpose |
|----------|--------|---------|
| `ITAK_TORCH_LIB` | Script | Points to backend lib directory |
| `YZMA_BENCHMARK_MODEL` | Script | GGUF model path for Go benchmarks |

## Output

Reports saved as timestamped markdown files in `benchmarks/results/`:

```
results/
  benchmark_2026-03-08_013534_beast.md
```

## Latest Baseline (March 8, 2026 - Beast)

| Engine | tok/s | vs Ollama GPU | VRAM (MB) | Power (W) | Temp (C) |
|--------|-------|---------------|-----------|-----------|----------|
| **iTaK Torch Vulkan** | **705.8** | **+12.6%** | 11,511 | 192.8 | 49 |
| iTaK Torch CUDA | 661.2 | +5.5% | 11,500 | 163.5 | 48 |
| Ollama GPU | 626.7 | baseline | 11,597 | 140.8 | 40 |
| iTaK Torch CPU | 108.8 | -82.6% | 10,501 | 17.9 | 33 |
| Ollama CPU | 90.7 | -85.5% | 10,501 | 18.0 | 33 |

## Prerequisites

### Both Machines
- Go 1.26+ (for iTaK Torch Go benchmarks)
- Ollama installed and running
- GGUF model file

### Beast Only
- `nvidia-smi` (ships with NVIDIA drivers)
- iTaK Torch lib directories in GOAgent:
  ```
  lib/windows_amd64_vulkan/   # Vulkan backend
  lib/windows_amd64_cuda/     # CUDA backend
  lib/windows_amd64/          # CPU-only backend
  ```
