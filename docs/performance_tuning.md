# iTaK Torch Performance Tuning

Torch is built in Go and optimized for low-latency inference. Here are the "free speed" toggles available to you.

## 1. PGO (Profile-Guided Optimization)

By default, the compiler guesses how your code runs. PGO gives it actual profiles.

**Enable it forever (Recommended):**
```bash
make pgo-default
```
This takes 30 seconds to generate `cmd/itaktorch/default.pgo`. After that, **every** build is 2-14% faster automatically.

## 2. AVX2 SIMD Instructions (`GOAMD64=v3`)

The Makefile is pre-configured to use `GOAMD64=v3` for all `x86_64` (AMD/Intel) builds. This tells Go it can use AVX2 vector instructions that have been standard on CPUs for the last decade.

**Benefit:** 5-15% faster standard library operations (hashing, memory copies, math).

If you are running on an ancient CPU (pre-2015) and get an "illegal instruction" crash, build manually with `GOAMD64=v1`:
```bash
GOAMD64=v1 make build
```

## 3. Tune Garbage Collection (`--gogc`)

Go's garbage collector pauses execution to clean up memory. By default, it runs when heap memory doubles (`GOGC=100`).

Inference servers allocate a lot of short-lived memory per token. You can trade RAM for throughput by telling Go to run the GC less often.

```bash
# Default (good balance)
./itaktorch serve --model x.gguf

# Less GC pauses, uses ~2x more RAM than default (Recommended for VPS/Docker)
./itaktorch serve --model x.gguf --gogc 200

# Even less GC pauses, uses ~5x more RAM
./itaktorch serve --model x.gguf --gogc 500
```

## 4. Zero-Allocation Hot Paths

Under the hood, Torch uses `sync.Pool` extensively in the streaming handlers. 
- Bytes buffers for JSON encoding
- Token slice arrays
- Strings builders

This eliminates ~80% of heap allocations on the hot path compared to standard HTTP REST servers, minimizing GC pressure regardless of your `--gogc` setting.
