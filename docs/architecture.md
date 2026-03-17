# iTaK Torch Architecture

## Overview

iTaK Torch is a Go-native LLM inference engine that loads GGUF models and runs inference through Vulkan, CUDA, or CPU backends. It uses zero-CGO FFI via `purego` to call llama.cpp's shared libraries directly from Go, producing a single static binary with no Python or CGO dependencies.

## Core Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    CLI (cmd/itaktorch)                        │
│  serve | recommend | catalog | pull | models                 │
└───────────────────────────┬──────────────────────────────────┘
                            │
┌───────────────────────────▼──────────────────────────────────┐
│                   HTTP Server (server.go)                     │
│  OpenAI-compatible + Ollama-compatible API routes             │
│  Rate Limiting | Response Cache | Debug Endpoints             │
├──────────────────────────────────────────────────────────────┤
│                   Scheduler (scheduler.go)                    │
│  Dual-priority queues (critical/normal)                       │
│  Backpressure at queue capacity (503)                         │
├──────────────────────────────────────────────────────────────┤
│                   Engine Layer                                │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────────┐         │
│  │ TorchEngine  │  │ MockEngine  │  │ GOTensor     │         │
│  │ (llama.cpp)  │  │ (testing)   │  │ (pure Go)    │         │
│  └──────┬──────┘  └─────────────┘  └──────────────┘         │
│         │                                                     │
│  ┌──────▼───────────────────────────────────────────┐        │
│  │ llama/ package (purego FFI bindings)               │        │
│  │ Model, Context, Sampler, Batch, Vocab, Memory      │        │
│  └──────┬────────────────────────────────────────────┘        │
│         │                                                     │
│  ┌──────▼──────┐                                              │
│  │ loader/     │  Dynamic lib loading per platform             │
│  │ lib/        │  windows_amd64, linux_amd64, darwin_arm64     │
│  └─────────────┘                                              │
└──────────────────────────────────────────────────────────────┘
```

## Key Packages

### `pkg/torch/` - Core Engine

| File | Purpose |
|------|---------|
| `server.go` | HTTP server with OpenAI-compatible routes, middleware, SSE streaming |
| `llama_engine.go` | Main inference engine wrapping llama.cpp FFI calls |
| `engine.go` | `Engine` interface definition (Infer, ModelName, GetStats, Close) |
| `scheduler.go` | Dual-priority request scheduler with backpressure |
| `model_registry.go` | Multi-model LRU registry for hot-swapping models |
| `models.go` | Curated model catalog with hardware-aware recommendations |
| `slot_manager.go` | KV cache slot allocation for concurrent requests |
| `continuous_batch.go` | Pipeline batching for throughput |
| `prefix_cache.go` | Prompt prefix KV cache reuse |
| `response_cache.go` | Prompt-level LRU response cache |
| `cpu_topology.go` | CPU core count, HT detection, thread tuning |
| `gpu_detect.go` | GPU vendor and VRAM detection (auto_config.go uses this) |
| `auto_config.go` | Auto-selects backend and GPU layers based on hardware |
| `vision_engine.go` | CLIP encoder for multimodal image+text inference |
| `spec_decode.go` | Speculative decoding (draft+target 2-model acceleration) |
| `multi_gpu.go` | Multi-GPU tensor parallelism and layer splitting |
| `json_mode.go` | JSON schema to GBNF grammar for structured output |
| `grammar.go` | GBNF grammar templates and validation |
| `hf_pull.go` | HuggingFace Hub model downloader with resumable downloads |
| `ollama_pull.go` | Ollama registry model downloader |
| `handlers_ollama.go` | Ollama-compatible API handlers (drop-in replacement) |
| `handlers_models.go` | Model management endpoints (load, unload, search, pull) |
| `handlers_tools.go` | Tool calling support |
| `handlers_lora.go` | LoRA adapter management |
| `handlers_embeddings.go` | OpenAI-compatible embeddings endpoint |
| `handlers_ops.go` | Operations endpoints (cache, scheduler stats) |
| `ratelimit.go` | Per-IP token bucket rate limiter |
| `validate.go` | Path traversal validation and file extension enforcement |
| `metrics.go` | Inference performance tracking |
| `metrics_export.go` | Prometheus-compatible metrics export |
| `kv_metrics.go` | KV cache utilization tracking |

### `pkg/torch/llama/` - FFI Bindings

Pure Go bindings to llama.cpp via `purego`. No CGO required. Platform-specific files (`*_windows.go`, `*_other.go`) handle struct-by-value calling conventions.

Key types: `Model`, `Context`, `Vocab`, `Sampler`, `Batch`, `Token`

### `pkg/torch/native/` - GOTensor

A pure Go transformer engine for tiny models (<1B params). Uses 32x32 tiled matrix multiplication for L1 cache residency. No external dependencies.

### `pkg/torch/loader/` - Dynamic Library Loading

Loads platform-specific shared libraries (`llama.dll`, `libllama.so`, `libllama.dylib`) at runtime using `purego`.

## Data Flow

1. **Request** arrives at HTTP server (OpenAI or Ollama format)
2. **Rate limiter** checks per-IP token bucket
3. **Cache check** looks up prompt hash in LRU cache
4. **Scheduler** queues the request (critical or normal priority)
5. **Engine** runs inference through llama.cpp FFI
6. **Response** is built, cached, and returned with performance headers

## Backend Selection Priority

1. Vulkan (cross-platform GPU)
2. CUDA (NVIDIA)
3. Metal (Apple Silicon)
4. HIP (AMD)
5. SYCL (Intel Arc)
6. CPU (universal fallback)

Selection is automatic via `auto_config.go` unless overridden by `--backend` flag or `GGML_BACKEND` env var.
