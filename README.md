# 🔥 iTaK Torch

**A pure Go inference engine that replaces Ollama.** No Python, no CUDA dependency, no Docker required. Just a single binary that runs LLMs on any machine.

> Built for the [iTaK Ecosystem](https://itak.live) - works standalone or as the AI backbone for iTaK agents.

---

## Quick Start

```bash
# Download a model and chat with it
itaktorch pull qwen3.5:0.8b
itaktorch chat model.gguf --bench

# Or run as an API server (OpenAI-compatible)
itaktorch serve --port 39271
```

That's it. No setup, no dependencies, no GPU drivers needed.

---

<details>
<summary><b>📦 What is iTaK Torch?</b></summary>

iTaK Torch is a local LLM inference engine written in **100% Go**. It loads GGUF model files and runs them directly on your hardware - CPU, GPU, or both.

**Why not just use Ollama?**
- Ollama processes **one request at a time** by default. Torch has a built-in swarm scheduler.
- Ollama is a separate service to manage. Torch is a **single binary** (12 MB) that embeds into your app.
- Torch auto-detects your hardware and picks the fastest strategy automatically.

**Who is this for?**
- Developers building AI agents that need local inference
- Anyone who wants to run LLMs without cloud APIs
- Teams building multi-agent swarms that need parallel inference

</details>

<details>
<summary><b>⚡ Features</b></summary>

### Inference Engine
- **GOTensor** - Pure Go transformer engine (no C dependencies)
- **GGUF loader** - Reads any GGUF model file (Llama, Qwen, Mistral, etc.)
- **GQA attention** - Grouped-Query Attention for modern architectures
- **Sparse FFN** - PowerInfer-style neuron activation for faster inference
- **Speculative decoding** - Draft model acceleration (2-3x speedup)

### GPU Acceleration (Vulkan)
- **Tiled MatMul** - 16x16 workgroup shared memory (5-10x over naive)
- **Fused SiLU+Mul** - Gate operation in one dispatch per FFN layer
- **Vec4 MatVec** - 4-element unrolled loads for 4x bandwidth
- **Q8 Dequant shader** - Int8 weights dequantized on-GPU (4x less PCIe)
- **FP16 compute** - Half-precision dot products on compatible hardware (2x throughput)
- **Multi-GPU** - Splits work across dGPU + iGPU via tensor parallelism
- **CPU fallback** - Always works without GPU

### GPU Acceleration (CUDA)
- **cuBLAS tensor cores** - Native NVIDIA acceleration via `cublasSgemm`/`cublasSgemv`
- **Persistent weight cache** - Model weights uploaded to GPU once, reused across calls
- **Zero CGO** - Uses `ebitengine/purego` for runtime library loading (no build deps)
- **No CUDA toolkit required** - Loads from NVIDIA driver DLLs/SOs already on the system
- **Linux, WSL, and Windows** - Cross-platform with a single binary
- **Build:** `go build -tags cuda ./...`

### GPU Acceleration (Metal / Apple Silicon)
- **AMX coprocessor** - Apple's dedicated matrix math hardware on M1/M2/M3/M4
- **Accelerate framework** - cblas_sgemm/sgemv loaded via purego (zero CGO)
- **Zero-copy weights** - Unified memory means Go slices are directly accessible by AMX
- **Intel Mac support** - Falls back to Accelerate's SSE/AVX path on Intel Macs
- **Build:** `go build -tags metal ./...`

### Swarm Inference
- **`/v1/swarm` endpoint** - Send N tasks, get N results back in parallel
- **SSE streaming** - `"stream": true` sends each result as it completes
- **Auto-detection** - Probes your CPU, RAM, and GPUs to pick the best strategy
- **4 strategies** - Sequential, Batch, Parallel, GPU-Batch
- **`/v1/capabilities`** - Query what this machine can handle before sending work

### Distributed Cluster
- **Multi-node** - Distribute swarm tasks across multiple machines on your LAN
- **Capability-weighted** - Faster machines get more tasks automatically
- **Auto-discovery** - UDP broadcast finds peers on the LAN (zero config)
- **Health cleanup** - Dead peers removed after 2 minutes of no response
- **Auto-fallback** - If a peer fails, its tasks run locally
- **Distributed streaming** - `"stream": true` works across cluster with SSE

### API Compatibility
- **OpenAI-compatible** - `POST /v1/chat/completions` works with any OpenAI client
- **Ollama-compatible** - Drop-in replacement (`/api/generate`, `/api/chat`, `/api/tags`)
- **SSE streaming** - Real-time token streaming
- **Response caching** - Identical prompts return cached results instantly

### Model Management
- **HuggingFace pull** - Download models from HF Hub
- **Ollama pull** - Pull models from Ollama's registry
- **Model registry** - Hot-swap models without restarting the server
- **LoRA adapters** - Load/unload fine-tuned adapters at runtime
- **SafeTensors import** - Load HuggingFace models directly (F32/F16/BF16)
- **GGUF export** - Save fine-tuned models back to GGUF for sharing

### Advanced Inference
- **Flash Attention** - Tiled O(N) memory attention for 32K+ context
- **Rotary Position Embeddings (RoPE)** - NTK-aware scaling, YaRN context extension
- **Sliding window attention** - Mistral-style O(N*W) memory for long contexts
- **BPE tokenizer** - Real tokenization from GGUF vocab (SentencePiece compatible)
- **Mixture of Experts (MoE)** - Top-K gating for Mixtral/DeepSeek/Qwen-MoE
- **GBNF grammar decoding** - Constrained output for guaranteed JSON/schema conformance
- **Attention Residuals (AttnRes)** - Learned depth-wise attention replacing fixed residuals

### Performance
- **Continuous batching** - Dynamic concurrent request serving
- **Prompt caching** - KV prefix sharing for common system prompts
- **KV cache compression** - INT8 quantized cache (4x memory reduction)
- **Quantize on load** - Auto Q4/Q8 at load time (4-8x memory savings)
- **Multi-GPU tensor parallel** - Shard weights across GPUs
- **Tensor pool** - 80-90% fewer GC allocations
- **SIMD kernels** - 2-3x faster CPU math operations

### Training and Fine-Tuning
- **LoRA training** - Low-rank adapter fine-tuning (rank 4-16)
- **AttnRes training** - Fine-tune attention residual queries
- **Autograd engine** - Reverse-mode autodiff with 8 differentiable ops
- **AdamW optimizer** - With gradient clipping, warmup, cosine decay
- **Training API** - `POST /v1/training/start` to launch background jobs
- **GGUF export** - Save fine-tuned weights for sharing with Ollama/llama.cpp

### Observability
- **Prometheus metrics** - `GET /metrics` for Grafana dashboards
- **Health endpoint** - `GET /health` with GPU, memory, scheduler stats
- **Benchmark suite** - Automated perf testing with JSON output
- **Feature status** - `GET /v1/features` to inspect enabled capabilities

### Streaming and Long Context
- **Attention Sinks** - StreamingLLM-style infinite context (sink + recent window)
- **Context Manager** - Auto-truncation with 4 strategies (truncate, sliding, summarize, sink)
- **Beam Search** - K-beam decoding for higher quality output

### Training and Fine-Tuning
- **LoRA training** - Low-rank adapter fine-tuning (rank 4-16)
- **AttnRes training** - Fine-tune attention residual queries
- **Autograd engine** - Reverse-mode autodiff with 8 differentiable ops
- **AdamW optimizer** - Gradient clipping, warmup, cosine decay
- **Mixed precision** - FP16/BF16 with dynamic loss scaling (2x throughput)
- **Activation checkpointing** - 8x memory savings at 33% compute cost
- **Weight merging** - Merge LoRA adapters into base model for deployment
- **Training API** - `POST /v1/training/start` for background fine-tuning jobs
- **GGUF export** - Save fine-tuned weights for Ollama/llama.cpp

### Agent Integration
- **Tool/function calling** - Parse Hermes, Qwen, Llama, raw JSON tool calls
- **GBNF grammar decoding** - Guaranteed JSON/schema output
- **Embedding pooling** - Mean/CLS/last token/max for vector search
- **Tokenize API** - `POST /v1/tokenize` + `POST /v1/detokenize`
- **Model info** - `GET /v1/models/info` with full metadata + capabilities

### Sampling and Decoding
- **Logit processor chain** - Composable pipeline: temperature, top-k, top-p, min-p, Mirostat v2
- **Repetition penalty** - Penalize repeated n-grams with presence/frequency penalties
- **Token healing** - Fix broken tokens at prompt/completion boundaries
- **Beam search** - K-beam decoding with length penalty and early stopping
- **Text watermarking** - Green/red list invisible watermarks with statistical detection

### Security and Metering
- **Token budget** - Per-API-key usage tracking with daily/monthly limits
- **Priority queue** - 5-level request scheduling with anti-starvation
- **Dynamic quantization** - Auto-switch FP32/Q8/Q4 based on memory pressure

### Production Infrastructure
- **Structured logging** - JSON logs for Loki/ELK/CloudWatch
- **Retry with fallback** - Exponential backoff, progressive context reduction
- **Rate limiting** - Per-endpoint with configurable burst

### Multimodal
- **Vision input pipeline** - Image load, resize, normalize, ViT patch extraction (LLaVA/Qwen-VL configs)
- **JSON Schema validation** - Validate structured output against schema with auto-fix

### Safety
- **Guardrails** - Input/output filtering: prompt injection (9 patterns), PII detection/redaction, topic blocking
- **Graceful shutdown** - Drain active requests, run cleanup hooks, release GPU cleanly

### Operational Maturity
- **Request tracing** - OpenTelemetry-style trace/span hierarchy with latency breakdown
- **Model warmup** - Pre-compute KV cache for system prompts at startup
- **A/B routing** - Weighted traffic split between models with per-route metrics
- **Model versioning** - Deploy/rollback/compare versions with rolling quality metrics
- **Semantic cache** - Cache responses by embedding cosine similarity (avoid re-inference)
- **Tensor memoization** - Content-addressable cache for RoPE tables, masks, embeddings

### Monitoring
- **Built-in benchmarks** - `--bench` flag shows tok/s, TTFT, RAM, per-layer timing
- **Prometheus metrics** - `/metrics` endpoint for Grafana dashboards
- **Health endpoint** - `/health` with GPU info, scheduler stats, resource usage
- **Debug system** - `ITAK_DEBUG=1` for live structured logs

</details>

<details>
<summary><b>🖥️ Hardware Auto-Detection</b></summary>

When you run `itaktorch serve`, it probes your machine and picks the best strategy:

| Your Hardware | Strategy | What Happens |
|---|---|---|
| Laptop (8 GB, no GPU) | `batch` | 1 model, 2 parallel slots |
| Desktop (16 GB, no GPU) | `batch` | 1 model, 4 parallel slots |
| Workstation (32+ GB) | `parallel` | Multiple model instances |
| Any machine + GPU 6+ GB | `gpu-batch` | Model on GPU, batched inference |
| Multi-GPU setup | `gpu-batch` | Tensor parallelism across GPUs |

Check what your machine supports:
```bash
curl http://localhost:39271/v1/capabilities
```

```json
{
  "strategy": "gpu-batch",
  "max_parallel": 6,
  "cpu_cores": 24,
  "ram_gb": 128,
  "gpus": [
    {"name": "NVIDIA GeForce RTX 4070 Ti SUPER", "vram_mb": 16384},
    {"name": "Intel UHD Graphics 770", "vram_mb": 2048, "is_shared": true}
  ]
}
```

</details>

<details>
<summary><b>🐝 Swarm Endpoint</b></summary>

The `/v1/swarm` endpoint accepts multiple tasks and runs them in parallel. The agent doesn't need to know about your hardware - Torch figures it out.

**Request:**
```bash
curl -X POST http://localhost:39271/v1/swarm \
  -H "Content-Type: application/json" \
  -d '{
    "tasks": [
      {"id": "index", "messages": [{"role": "user", "content": "Generate index.html"}]},
      {"id": "about", "messages": [{"role": "user", "content": "Generate about.html"}]},
      {"id": "pricing", "messages": [{"role": "user", "content": "Generate pricing.html"}]}
    ],
    "model": "qwen3:1.7b"
  }'
```

**Response (includes bench metrics):**
```json
{
  "results": [
    {"id": "index", "text": "<html>...", "metrics": {"tokens_per_second": 45.2}},
    {"id": "about", "text": "<html>...", "metrics": {"tokens_per_second": 43.8}},
    {"id": "pricing", "text": "<html>...", "metrics": {"tokens_per_second": 44.1}}
  ],
  "bench": {
    "strategy": "gpu-batch",
    "task_count": 3,
    "max_parallel": 3,
    "total_duration_ms": 2340,
    "total_tokens": 1847,
    "avg_tok_per_sec": 789.3,
    "gpus": ["NVIDIA GeForce RTX 4070 Ti SUPER"]
  }
}
```

</details>

<details>
<summary><b>🌐 Distributed Cluster</b></summary>

Run Torch on multiple machines and distribute work across your LAN. Each node runs a Torch server. A coordinator distributes tasks proportionally by hardware capability.

**Setup:**
```bash
# On each machine
itaktorch serve --port 39271

# Register peers with the coordinator (Beast)
curl -X POST http://beast:39271/v1/cluster/join \
  -d '{"address":"192.168.0.100:39271","name":"daughter-pc"}'

curl -X POST http://beast:39271/v1/cluster/join \
  -d '{"address":"192.168.0.101:39271","name":"kid-pc"}'
```

**Check cluster:**
```bash
curl http://beast:39271/v1/cluster/peers
```

Now when you send `/v1/swarm` to Beast, tasks distribute automatically:
```
18 tasks received at Beast
  Beast (RTX 4070 Ti, max_parallel=8): 8 tasks
  daughter-pc (RTX 3060, max_parallel=6): 6 tasks
  kid-pc (GTX Titan, max_parallel=4): 4 tasks
```

If a peer goes offline, its tasks automatically fall back to the local node.

**Auto-Discovery:**
Nodes also auto-discover each other via UDP broadcast on port 39272. Just start Torch on each machine - they find each other automatically.

</details>

<details>
<summary><b>🎮 GPU Build Modes</b></summary>

| Build Tag | Backend | Acceleration | Requirements |
|-----------|---------|-------------|-------------|
| (none) | CPU only | SIMD | Nothing |
| `-tags cuda` | cuBLAS | Tensor cores | NVIDIA GPU driver |
| `-tags metal` | Accelerate | AMX coprocessor | macOS (any version) |
| `-tags wgpu` | Vulkan/Metal | Compute shaders | Vulkan or Metal driver |

**CUDA backend** (best for NVIDIA GPUs):
```bash
go build -tags cuda -o itaktorch ./cmd/itaktorch
```
The CUDA build loads `libcublas.so` (Linux/WSL) or `cublas64_12.dll` (Windows) from the NVIDIA driver at runtime. Users do **not** need the CUDA Toolkit installed.

**Metal backend** (best for Apple Silicon Macs):
```bash
go build -tags metal -o itaktorch ./cmd/itaktorch
```
Uses Apple's Accelerate framework for AMX-accelerated matrix math. Zero-copy weights via unified memory. Also works on Intel Macs (uses AVX path).

**Vulkan backend** (any GPU vendor):
```bash
go build -tags wgpu -o itaktorch ./cmd/itaktorch
```

**CPU-only** (always works):
```bash
go build -o itaktorch ./cmd/itaktorch
```

<details>
<summary><b>🔧 CLI Commands</b></summary>

```bash
# Run as API server
itaktorch serve --port 39271

# Interactive chat with benchmarks
itaktorch chat model.gguf --bench --gpu

# Pull models
itaktorch pull qwen3.5:0.8b          # from Ollama registry
itaktorch pull hf:Qwen/Qwen3.5-0.8B  # from HuggingFace

# List available models
itaktorch models

# Run benchmark suite
itaktorch bench model.gguf dense
```

</details>

<details>
<summary><b>🐳 Docker</b></summary>

```bash
# Build the image
docker build -t itaktorch .

# Run with a model directory mounted
docker run -d \
  --name itaktorch \
  -p 39271:39271 \
  -v ~/.itaktorch/models:/models \
  itaktorch serve --port 39271

# Check health
curl http://localhost:39271/health
```

For GPU support (NVIDIA):
```bash
docker run -d \
  --gpus all \
  --name itaktorch-gpu \
  -p 39271:39271 \
  -v ~/.itaktorch/models:/models \
  itaktorch serve --port 39271 --gpu
```

</details>

<details>
<summary><b>📡 API Reference</b></summary>

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/v1/chat/completions` | Standard chat (OpenAI-compatible) |
| POST | `/v1/swarm` | Parallel batch inference |
| GET | `/v1/capabilities` | Hardware profile and strategy |
| GET | `/v1/models` | List available models |
| GET | `/health` | Server health + metrics |
| POST | `/v1/models/pull` | Download model from HuggingFace |
| POST | `/v1/models/pull/ollama` | Download model from Ollama |
| POST | `/v1/models/load` | Load a model into memory |
| POST | `/v1/models/unload` | Unload a model |
| GET | `/v1/models/loaded` | List currently loaded models |
| POST | `/v1/generate_raw` | Raw token generation |
| GET | `/v1/embeddings` | Generate embeddings |
| POST | `/v1/cluster/join` | Register a cluster peer |
| GET | `/v1/cluster/peers` | List cluster peers |
| GET | `/v1/scheduler/stats` | Scheduler queue metrics |
| GET | `/v1/cache/stats` | Response cache stats |
| GET | `/metrics` | Prometheus metrics |

</details>

---

## License

MIT
