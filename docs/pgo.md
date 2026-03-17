# Profile-Guided Optimization (PGO) for iTaK Torch

PGO is the same technique Google uses (AutoFDO) to optimize the Android kernel. It works by:

1. **Recording** how your code actually runs (which functions are hot, which branches are taken)
2. **Feeding** that data back to the compiler
3. **Rebuilding** -- the compiler uses the profile to make better optimization decisions

Go has built-in PGO since Go 1.21. Google reports **2-14% performance improvements** on real workloads.

## Set It and Forget It (Recommended)

Run this once on Linux/WSL or any Unix system:

```bash
make pgo-default
```

This generates `cmd/itaktorch/default.pgo`. Once that file exists, **every** `go build ./cmd/itaktorch/` is automatically PGO-optimized. No flags, no extra steps. Commit it to the repo and every build is faster from that point forward.

To regenerate (after major code changes or on different hardware):

```bash
rm cmd/itaktorch/default.pgo
make pgo-default
```

## Quick Start

### Option A: Fully Automatic (30 seconds)

```bash
make pgo-auto
```

This starts a mock server, sends 50 test requests, captures the profile, and rebuilds with it. One command.

### Option B: Profile with Real Inference

For best results, profile with your actual model and workload:

```bash
# Step 1: Start server with profiling enabled.
itaktorch serve --model ./my-model.gguf --pgo-capture cpu.pprof

# Step 2: Send real requests (use your actual prompts).
# Let it run for 1-5 minutes under typical load.

# Step 3: Press Ctrl+C. Profile is saved automatically.

# Step 4: Rebuild with the profile.
make pgo-build
# Or: go build -pgo=cpu.pprof -ldflags="-s -w" -o itaktorch ./cmd/itaktorch/
```

### Option C: Live Profiling via HTTP

While the server is running, you can grab a profile anytime:

```bash
# 30-second CPU profile
curl -o cpu.pprof http://localhost:41934/debug/pprof/profile?seconds=30

# Then rebuild
go build -pgo=cpu.pprof ./cmd/itaktorch/
```

## What Gets Faster

PGO optimizes:
- **Hot function inlining** -- frequently called functions get inlined more aggressively
- **Branch prediction** -- the compiler learns which `if` branches are taken most
- **Register allocation** -- hot variables stay in registers instead of memory
- **Function layout** -- hot code paths are placed together for better CPU cache usage

For Torch specifically, this means:
- Scheduler dispatch gets faster
- HTTP handler routing improves
- JSON marshaling/parsing speeds up
- Response cache lookups accelerate

## Verifying PGO is Active

```bash
# Build with PGO and check the binary
go build -pgo=cpu.pprof -gcflags="-m=2" ./cmd/itaktorch/ 2>&1 | grep "PGO"
```

The compiler will print messages about PGO-guided decisions.

## Re-profiling

PGO profiles should be refreshed when:
- You add major new features
- Request patterns change significantly
- You upgrade Go versions

The profile doesn't need to be from the exact same binary -- Go's PGO is robust to code changes.

## Android PGO

For Android builds, profile on the target device:

```bash
# Push the binary to the phone
adb push itaktorch-android-arm64 /data/local/tmp/itaktorch

# Run with profiling
adb shell /data/local/tmp/itaktorch serve --mock --pgo-capture /data/local/tmp/cpu.pprof &

# Send requests...

# Pull the profile
adb pull /data/local/tmp/cpu.pprof

# Rebuild with mobile-specific profile
GOOS=android GOARCH=arm64 go build -pgo=cpu.pprof -ldflags="-s -w" -o itaktorch-android-arm64 ./cmd/itaktorch/
```
