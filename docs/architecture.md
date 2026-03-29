# iTaK Torch Architecture (Go Edition)

iTaK Torch is designed as a modular, local-first inference engine.

## Core Modules

### 1. GGUF Parser

- **Location**: `pkg/torch/native/gguf.go`
- **Function**: Parses GGUF v3 files, extracting metadata, tensor info, and KV pairs.

### 2. Inference Kernel

- **Technique**: Uses standard Transformer logic implemented in pure Go.
- **Acceleration**: Leverages `ebitengine/purego` for hardware-native instructions (AVX2, NEON) where available.

### 3. Server Layer

- **API**: OpenAI-compatible HTTP endpoints.
- **Concurrency**: Go-routines for parallel prompt processing and stream handling.

## Dependency Stack

- **WebGPU**: `github.com/cogentcore/webgpu` for GPU-accelerated tensor math.
- **FFI**: `github.com/jupiterrider/ffi` for cross-platform hardware abstraction.
