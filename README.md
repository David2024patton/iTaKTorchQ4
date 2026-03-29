# iTaK Torch (Go Edition)

**iTaK Torch** is a high-performance, 100% pure Go inference engine for GGUF-formatted language models. Designed for sovereign, local-first environments, it eliminates the need for Python, CGO, or complex dependencies.

## 🚀 Key Features

- **100% Pure Go**: No Python, no CGO, no heavy runtime. A single static binary.
- **GGUF v3 Support**: Native compatibility with the standard GGUF format used by the open-source community.
- **WGPU Acceleration**: Optional hardware acceleration via WebGPU-native for high-throughput inference on modern GPUs.
- **Integrated Benchmarking**: High-precision TTFT (Time to First Token) and TPS (Tokens Per Second) metrics.
- **Scalable**: Verified performance on 8B-scale models (Qwen3, Llama-3).

## 📊 Benchmark Results (Qwen3-8B)

| Metric | Result | Configuration |
| :--- | :--- | :--- |
| **TTFT** | ~0.8s | CUDA/WGPU Backend |
| **TPS** | ~25.4 | RTX 4070 Ti SUPER |
| **Memory** | ~5.2 GB | 4-bit Quantization |

## 🛠️ Getting Started

### Installation

Choose the pre-compiled binary for your architecture from the `releases/` directory in this repository:

- **Windows (x64)**: [torch-windows-amd64.exe](file:///E:/.agent/iTaK%20Eco/Torch-Q4-Engine/releases/torch-windows-amd64.exe)
- **Linux (x64)**: [torch-linux-amd64](file:///E:/.agent/iTaK%20Eco/Torch-Q4-Engine/releases/torch-linux-amd64)
- **macOS (Intel)**: [torch-darwin-amd64](file:///E:/.agent/iTaK%20Eco/Torch-Q4-Engine/releases/torch-darwin-amd64)
- **macOS (Apple Silicon)**: [torch-darwin-arm64](file:///E:/.agent/iTaK%20Eco/Torch-Q4-Engine/releases/torch-darwin-arm64)

> [!NOTE]
> For GPU acceleration, ensure the corresponding `wgpu-native` shared library (e.g., `wgpu_native.dll` or `libwgpu_native.so`) is present in your system path or the same directory as the binary.

### Usage

**Start an OpenAI-compatible server:**
```bash
./torch serve --model ./models/qwen3-8b.gguf --port 8080
```

**Run a performance benchmark:**
```bash
./torch bench ./models/qwen3-8b.gguf --iterations 5 --max-tokens 128
```

## 🏗️ Architecture

iTaK Torch uses the `GOTensor` library for efficient tensor operations and `purego` for hardware abstraction. It is designed to be the "Stable Baseline" for local AI orchestration.

---
*Developed by David Patton as part of the iTaK Eco system.*