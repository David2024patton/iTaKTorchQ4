# iTaK Torch Configuration

## CLI Flags

### `serve` Command

```
itaktorch serve [flags]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | (required) | Path to GGUF model file |
| `--port` | `28080` | HTTP server port |
| `--mock` | `false` | Run with mock engine (no model needed) |
| `--gpu-layers` | `-1` (auto) | Layers to offload to GPU (-1 = all, 0 = CPU only) |
| `--ctx` | `4096` | Context window size in tokens |
| `--batch` | `512` | Batch size for prompt processing |
| `--threads` | auto | CPU threads (auto-tuned by `cpu_topology.go`) |
| `--backend` | auto | Force backend: `vulkan`, `cuda`, `cpu` |
| `--ollama` | `false` | Enable Ollama-compatible API routes |
| `--max-slots` | `1` | Concurrent inference slots for batching |
| `--kv-cache-type` | `f16` | KV cache quantization: `f16`, `q8_0`, `q4_0` |
| `--no-flash-attn` | `false` | Disable flash attention |
| `--no-mmap` | `false` | Disable memory-mapped model loading |
| `--draft-model` | (none) | Path to draft model for speculative decoding |
| `--draft-n` | `5` | Speculative tokens per step |

### `recommend` Command

```
itaktorch recommend
```

Detects hardware (RAM, GPU, VRAM) and recommends compatible models from the curated catalog.

### `catalog` Command

```
itaktorch catalog
```

Lists all models in the curated catalog with size and hardware requirements.

### `pull` Command

```
itaktorch pull <model-name>
```

Downloads a model from HuggingFace Hub to the local cache.

### `models` Command

```
itaktorch models
```

Lists downloaded models in the cache directory.

---

## Environment Variables

### Runtime

| Variable | Values | Effect |
|----------|--------|--------|
| `ITAK_DEBUG` | `1`, `true` | Enable debug endpoints and DEBUG-level logging |
| `ITAK_DEBUG_COLOR` | `false`, `0` | Disable ANSI colors in logs |
| `ITAK_TORCH_LIB` | path | Override library directory (for development) |
| `GGML_BACKEND` | `vulkan`, `cuda`, `cpu` | Force a specific compute backend |
| `GGML_VK_DEVICE` | `0`, `1`, etc. | Select Vulkan GPU device index |
| `GGML_N_THREADS` | number | Override CPU thread count |
| `GGML_N_GPU_LAYERS` | number | Override GPU layer count |

### Testing

| Variable | Description |
|----------|-------------|
| `ITAK_TORCH_LIB` | Path to llama.cpp shared libraries (required for `llama/` tests) |
| `YZMA_TEST_MODEL` | Path to GGUF model for integration tests |
| `YZMA_TEST_SPLIT_MODELS` | Comma-separated paths for split model tests |
| `YZMA_TEST_ENCODER_MODEL` | Path to encoder model for embedding tests |
| `YZMA_TEST_MMMODEL` | Path to multimodal model for vision tests |
| `YZMA_TEST_LORA_MODEL` | Path to LoRA base model |
| `YZMA_TEST_LORA_ADAPTER` | Path to LoRA adapter file |
| `YZMA_BENCHMARK_MODEL` | Path to model for benchmark tests |

---

## Performance Tuning

### GPU Offload

```bash
# Full GPU offload (recommended for models that fit in VRAM)
--gpu-layers -1

# Partial offload (when model is larger than VRAM)
--gpu-layers 20

# CPU only
--gpu-layers 0
```

### KV Cache Quantization

Reduces VRAM usage by quantizing the key-value cache:

```bash
# Default (full precision, most accurate)
--kv-cache-type f16

# 8-bit quantized (50% VRAM savings, minimal quality loss)
--kv-cache-type q8_0

# 4-bit quantized (75% VRAM savings, some quality loss)
--kv-cache-type q4_0
```

### Thread Tuning

- Auto-tuned by `cpu_topology.go` based on physical cores, HT status, and model size
- Small models: uses physical core count (avoids HT contention)
- Large models with no GPU: uses all logical cores
- Models with GPU offload: reduces threads (GPU does heavy lifting)

### Speculative Decoding

For 2-3x speedup on predictable tasks (code, JSON, templates):

```bash
itaktorch serve \
  --model large-model.gguf \
  --draft-model small-model.gguf \
  --draft-n 5
```

Requirements:
- Both models must share the same tokenizer (same vocabulary)
- Best with same-family models (e.g., Qwen-0.5B drafting for Qwen-14B)
- VRAM must fit both models simultaneously

### Multi-GPU

Automatically detected when 2+ discrete GPUs are present:
- Same-vendor GPUs: tensor parallelism (row split) for max throughput
- Mixed-vendor GPUs: layer split (safer, always works)
- VRAM-proportional auto-balancing

---

## Model Storage

Default cache directory: `~/.itaktorch/models/`

Models are stored as GGUF files with SHA256 checksum verification after download.
