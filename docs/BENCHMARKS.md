# iTaK Torch Benchmarks

Performance metrics on various hardware configurations using the Qwen3-8B (4-bit) model.

## Hardware: NVIDIA RTX 4070 Ti SUPER (Windows)

- **Engine**: WGPU/CUDA
- **TTFT**: 0.82 seconds
- **TPS**: 25.4 tokens/sec
- **VRAM**: 5.2 GB

## Hardware: Apple M2 Pro (macOS)

- **Engine**: Metal (via WGPU)
- **TTFT**: 1.10 seconds
- **TPS**: 18.2 tokens/sec
- **Unified Memory**: 5.8 GB

## Hardware: Intel Core i9 (Linux)

- **Engine**: AVX2 (CPU Only)
- **TTFT**: 2.45 seconds
- **TPS**: 2.3 tokens/sec
- **RAM**: 6.1 GB
