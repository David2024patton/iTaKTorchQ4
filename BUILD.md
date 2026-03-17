# iTaK Torch - Standalone Build & Test Guide

## Status: ✅ Migration Complete

All 67 Go source files have been successfully migrated from GOAgent to iTaK Torch with updated module path:
- **Old module**: `github.com/David2024patton/iTaKAgent`
- **New module**: `github.com/David2024patton/iTaKTorch`

## Build Instructions

### Prerequisites
```powershell
# Install Go 1.26.0+ (currently configured for 1.26.0)
# Verify Go installation
go version

# Navigate to project root
cd "e:\.agent\iTaK Torch"
```

### Build Binary

```powershell
# Tidy dependencies
go mod tidy

# Build standalone executable (Windows)
go build -o bin/itaktorch.exe ./cmd/itaktorch

# Verify binary created
ls -lh bin/itaktorch.exe

# Expected output: ~5-8 MB executable with all backends embedded
```

### Alternative Builds by Backend

```powershell
# CPU-only build (smallest)
go build -ldflags="-X main.Backend=cpu" -o bin/itaktorch-cpu.exe ./cmd/itaktorch

# Vulkan-enabled build (cross-GPU support)
$env:GGML_VK_DEVICE = "0"
go build -ldflags="-X main.Backend=vulkan" -o bin/itaktorch-vulkan.exe ./cmd/itaktorch

# NVIDIA CUDA build
go build -ldflags="-X main.Backend=cuda" -o bin/itaktorch-cuda.exe ./cmd/itaktorch
```

## Testing

### 1. Mock Mode Test (No Model Required)
```powershell
cd "e:\.agent\iTaK Torch"

# Start mock server
.\bin\itaktorch.exe serve --mock --port 41934

# In another terminal, test the API
$response = Invoke-RestMethod -Uri "http://localhost:41934/v1/chat/completions" `
  -Method POST `
  -ContentType "application/json" `
  -Body '{"messages":[{"role":"user","content":"Hello"}],"model":"mock"}'

$response | ConvertTo-Json
```

**Expected output**: Mock engine returns `[iTaKTorch Mock / mock] Received: "Hello"`

### 2. Recommend Hardware
```powershell
# Detect your system specs and recommended models
.\bin\itaktorch.exe recommend

# Output example:
# iTaKTorch Hardware Detection
# =========================
#   Estimated RAM:  16384 MB
#   GPU:            Not detected (CPU-only mode)
# 
# Models that fit your hardware (8 available):
#   qwen2.5-0.5b-q4_k_m ... CPU friendly
#   phi-2-q5_k_m ... Fast inference
#   ...
```

### 3. Model Catalog
```powershell
# List all available models
.\bin\itaktorch.exe catalog

# Pull a model
.\bin\itaktorch.exe pull qwen2.5-0.5b-q4_k_m

# Check downloaded models
.\bin\itaktorch.exe models
```

### 4. Full Integration Test
```powershell
# Download a small model (if not already cached)
.\bin\itaktorch.exe pull qwen2.5-0.5b-q4_k_m

# Start the actual server with GPU acceleration
.\bin\itaktorch.exe serve `
  --model ~/.itaktorch/models/qwen2.5-0.5b-q4_k_m.gguf `
  --gpu-layers 10 `
  --ctx 512 `
  --batch 128 `
  --port 41934

# In another terminal, send an inference request
$request = @{
    model = "qwen2.5-0.5b-q4_k_m"
    messages = @(
        @{ role = "user"; content = "What is 2+2?" }
    )
    temperature = 0.7
    max_tokens = 100
} | ConvertTo-Json

$response = Invoke-RestMethod -Uri "http://localhost:41934/v1/chat/completions" `
  -Method POST `
  -ContentType "application/json" `
  -Body $request

$response.choices[0].message.content
```

## Module Structure

```
iTaK Torch/
├── go.mod                          # Module: github.com/David2024patton/iTaKTorch
├── cmd/
│   └── itaktorch/
│       └── main.go                 # CLI entry point
├── pkg/torch/
│   ├── continuous_batch.go         # Pipeline batching
│   ├── cpu_topology.go             # CPU core detection
│   ├── engine.go                   # Engine interface
│   ├── kv_metrics.go               # KV cache tracking
│   ├── llama_engine.go             # Main inference (2200+ lines)
│   ├── logger.go                   # Logging utilities
│   ├── metrics.go                  # Performance tracking
│   ├── models.go                   # Model management
│   ├── model_registry.go           # Multi-model LRU registry
│   ├── prefix_cache.go             # Prompt caching
│   ├── scheduler.go                # Request scheduling
│   ├── server.go                   # HTTP API (OpenAI-compatible)
│   ├── slot_manager.go             # Slot allocation
│   ├── tokenizer.go                # Token encoding/decoding
│   ├── types.go                    # Core types
│   ├── vision_engine.go            # Multimodal support
│   ├── llama/                      # FFI bindings (35 files)
│   │   ├── backend.go              # Vulkan/CUDA/Metal auto-detection
│   │   ├── batch.go, context.go, ggml.go, model.go, etc.
│   ├── tokenizer/                  # BPE tokenizer (2 files)
│   ├── loader/                     # Dynamic lib loading (5 files)
│   └── utils/                      # Platform-specific utils (4 files)
├── lib/
│   ├── windows_amd64/              # CPU-only
│   ├── windows_amd64_cuda/         # NVIDIA CUDA
│   ├── windows_amd64_vulkan/       # Cross-GPU Vulkan
│   ├── darwin_arm64/               # Apple Silicon
│   └── linux_amd64/                # Linux
├── models/                         # GGUF model storage
├── docs/                           # Documentation
├── config/                         # Configuration files
├── scripts/                        # Utility scripts
└── benchmarks/                     # Performance benchmarks
```

## Verification Commands

### Verify Module Path
```powershell
Get-Content go.mod | Select-String "^module"
# Should output: module github.com/David2024patton/iTaKTorch
```

### Verify Import Updates
```powershell
# Check for any remaining old imports (should be 0)
Select-String -Path "pkg\torch\**\*.go" -Pattern "iTaKAgent" -Recurse
# Should return nothing

# Verify new imports are present
Select-String -Path "pkg\torch\**\*.go" -Pattern "iTaKTorch" -Recurse | Measure-Object
# Should show >50 matches
```

### Test Import Resolution
```powershell
# This will verify all imports resolve correctly
go mod tidy
go build -o /dev/null ./cmd/itaktorch 2>&1
# Should show no import errors
```

## GPU Backend Details

### Auto-Detection Priority (llama/backend.go)
1. **Vulkan** - Cross-platform (Windows/Linux/macOS)
2. **CUDA** - NVIDIA GPUs (throughput specialist)
3. **Metal** - Apple Silicon native
4. **HIP** - AMD GPUs
5. **SYCL** - Intel Arc
6. **CPU** - Universal fallback

### Configuration Environment Variables
```powershell
# Force specific backend
$env:GGML_BACKEND = "vulkan"  # or: cuda, metal, cpu

# Vulkan device selection
$env:GGML_VK_DEVICE = "0"  # Select GPU device

# Performance tuning
$env:GGML_N_THREADS = "4"  # CPU thread count
$env:GGML_N_GPU_LAYERS = "10"  # Layers to GPU
```

## API Endpoints (OpenAI-Compatible)

```
POST /v1/chat/completions           # Standard chat inference
POST /v1/completions                # Text completion
POST /v1/embeddings                 # Text embeddings
GET  /health                        # Server health check
GET  /v1/models                     # List loaded models
GET  /metrics                       # Performance metrics
```

## Performance Optimization Tips

1. **Batch Size**: Increase from 128 (default) to 512+ for throughput
2. **KV Cache**: Use `--kv-cache-type q8_0` to save 50% VRAM
3. **GPU Layers**: Use `--gpu-layers -1` to offload everything
4. **Multi-Model**: Use `--max-slots 4` for parallel requests
5. **Flash Attention**: Enabled by default, disable with `--no-flash-attn` if issues

## Troubleshooting

### Build Fails with Import Errors
```powershell
# Clear Go cache and retry
go clean -cache
go mod tidy
go build ./cmd/itaktorch
```

### Runtime Backend Not Found
```powershell
# Verify lib files exist
ls lib/windows_amd64*

# Check environment
$env:ITAK_TORCH_LIB = ".\lib"
.\bin\itaktorch.exe serve --mock --port 41934
```

### Port Already in Use
```powershell
# Use different port
.\bin\itaktorch.exe serve --mock --port 11434

# Or kill existing process
Get-Process itaktorch | Stop-Process -Force
```

## Next Steps

1. ✅ Binary built successfully
2. ✅ Mock mode tested
3. ⏭️ Download a model: `.\bin\itaktorch recommend`
4. ⏭️ Run against real model
5. ⏭️ Benchmark performance: `go test -bench ./pkg/torch/...`
6. ⏭️ Deploy to production with Docker

## Project Links

- **Repository**: `github.com/David2024patton/iTaKTorch`
- **Previously**: Part of `github.com/David2024patton/iTaKAgent`
- **Benchmarks**: See `benchmarks/` directory for performance data
- **Documentation**: See `docs/` directory for architecture deep-dives
