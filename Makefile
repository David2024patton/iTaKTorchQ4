# iTaK Torch Makefile
#
# Standard builds, PGO optimization, and cross-compilation targets.
# Run `make help` for a full list of targets.

BINARY      = itaktorch
MODULE      = github.com/David2024patton/iTaKTorch
CMD         = ./cmd/itaktorch/
PROFILE     = cpu.pprof
LDFLAGS     = -s -w
PORT        = 41934

# Default: build for current platform.
# GOAMD64=v3 enables AVX2 instructions for 5-15% faster standard library ops.
.PHONY: build
build:
	GOAMD64=v3 go build -ldflags="$(LDFLAGS)" -o $(BINARY) $(CMD)

# ============================================================================
# PGO (Profile-Guided Optimization)
# Same technique as Google's AutoFDO but for Go binaries.
# Yields 2-14% faster inference depending on workload.
# ============================================================================

# Step 1: Build and run server with CPU profiling enabled.
# Hit it with real requests, then Ctrl+C to save the profile.
.PHONY: pgo-profile
pgo-profile: build
	@echo "=== PGO Step 1: Recording CPU profile ==="
	@echo "  1. Server will start with profiling enabled"
	@echo "  2. Send real inference requests to it"
	@echo "  3. Press Ctrl+C to save profile and stop"
	@echo ""
	./$(BINARY) serve --mock --port $(PORT) --pgo-capture $(PROFILE)

# Step 2: Rebuild with the captured profile for optimized binary.
.PHONY: pgo-build
pgo-build:
	@if [ ! -f $(PROFILE) ]; then \
		echo "Error: $(PROFILE) not found. Run 'make pgo-profile' first."; \
		exit 1; \
	fi
	@echo "=== PGO Step 2: Building with profile ==="
	go build -pgo=$(PROFILE) -ldflags="$(LDFLAGS)" -o $(BINARY) $(CMD)
	@echo "PGO-optimized binary built: ./$(BINARY)"

# One-shot: profile for 30s with mock load, then rebuild.
.PHONY: pgo-auto
pgo-auto: build
	@echo "=== PGO Auto: 30s profile capture + rebuild ==="
	./$(BINARY) serve --mock --port $(PORT) --pgo-capture $(PROFILE) &
	sleep 2
	@echo "Sending test requests..."
	@for i in $$(seq 1 50); do \
		curl -s -X POST http://localhost:$(PORT)/v1/chat/completions \
			-H "Content-Type: application/json" \
			-d '{"model":"mock","messages":[{"role":"user","content":"hello"}]}' > /dev/null; \
	done
	@echo "Stopping server..."
	kill %1 2>/dev/null || true
	sleep 1
	go build -pgo=$(PROFILE) -ldflags="$(LDFLAGS)" -o $(BINARY) $(CMD)
	@echo "PGO-optimized binary built: ./$(BINARY)"

# Generate default.pgo so every future build is PGO-optimized automatically.
# This is the "set it and forget it" option. Run once, commit, done.
.PHONY: pgo-default
pgo-default:
	bash scripts/generate_pgo.sh
	@echo ""
	@echo "default.pgo is now in cmd/itaktorch/."
	@echo "Every 'go build ./cmd/itaktorch/' is now PGO-optimized."
	@echo "Commit cmd/itaktorch/default.pgo to lock it in."

# ============================================================================
# Cross-Compilation Targets
# ============================================================================

# Linux AMD64 (servers, VPS, Docker).
.PHONY: linux-amd64
linux-amd64:
	CGO_ENABLED=0 GOOS=linux GOARCH=amd64 GOAMD64=v3 go build -ldflags="$(LDFLAGS)" -o $(BINARY)-linux-amd64 $(CMD)

# Linux ARM64 (Raspberry Pi, ARM servers).
.PHONY: linux-arm64
linux-arm64:
	CGO_ENABLED=0 GOOS=linux GOARCH=arm64 go build -ldflags="$(LDFLAGS)" -o $(BINARY)-linux-arm64 $(CMD)

# Android ARM64 (phones, tablets).
# Uses the GOTensor pure-Go engine (no llama.cpp FFI on Android).
.PHONY: android-arm64
android-arm64:
	CGO_ENABLED=0 GOOS=android GOARCH=arm64 go build -ldflags="$(LDFLAGS)" -o $(BINARY)-android-arm64 $(CMD)

# Android x86_64 (emulators, Chromebooks).
.PHONY: android-amd64
android-amd64:
	CGO_ENABLED=0 GOOS=android GOARCH=amd64 GOAMD64=v3 go build -ldflags="$(LDFLAGS)" -o $(BINARY)-android-amd64 $(CMD)

# macOS ARM64 (Apple Silicon).
.PHONY: darwin-arm64
darwin-arm64:
	CGO_ENABLED=0 GOOS=darwin GOARCH=arm64 go build -ldflags="$(LDFLAGS)" -o $(BINARY)-darwin-arm64 $(CMD)

# macOS AMD64 (Intel Macs).
.PHONY: darwin-amd64
darwin-amd64:
	CGO_ENABLED=0 GOOS=darwin GOARCH=amd64 GOAMD64=v3 go build -ldflags="$(LDFLAGS)" -o $(BINARY)-darwin-amd64 $(CMD)

# Windows AMD64.
.PHONY: windows-amd64
windows-amd64:
	CGO_ENABLED=0 GOOS=windows GOARCH=amd64 GOAMD64=v3 go build -ldflags="$(LDFLAGS)" -o $(BINARY)-windows-amd64.exe $(CMD)

# Build all platforms.
.PHONY: all
all: linux-amd64 linux-arm64 android-arm64 android-amd64 darwin-arm64 darwin-amd64 windows-amd64
	@echo "All platform binaries built."

# ============================================================================
# Testing & Benchmarks
# ============================================================================

.PHONY: test
test:
	go test ./... -v

.PHONY: bench
bench:
	go test ./pkg/torch/ -bench=. -benchmem -count=3

.PHONY: vet
vet:
	go vet ./...

# ============================================================================
# Docker
# ============================================================================

.PHONY: docker
docker:
	docker build -t itaktorch:latest .

.PHONY: docker-run
docker-run:
	docker run -p $(PORT):$(PORT) -v ./models:/models itaktorch:latest serve --mock --port $(PORT)

# ============================================================================
# Utilities
# ============================================================================

.PHONY: clean
clean:
	rm -f $(BINARY) $(BINARY)-* $(PROFILE)

.PHONY: help
help:
	@echo "iTaK Torch Build Targets"
	@echo ""
	@echo "  build          Build for current platform"
	@echo ""
	@echo "  PGO (Profile-Guided Optimization):"
	@echo "  pgo-profile    Start server with CPU profiling (Ctrl+C to save)"
	@echo "  pgo-build      Rebuild using captured CPU profile"
	@echo "  pgo-auto       Auto-profile for 30s + rebuild (one command)"
	@echo "  pgo-default    Generate default.pgo (auto-PGO on every build forever)"
	@echo ""
	@echo "  Cross-Compilation:"
	@echo "  linux-amd64    Linux x86_64"
	@echo "  linux-arm64    Linux ARM64 (RPi, ARM servers)"
	@echo "  android-arm64  Android ARM64 (phones)"
	@echo "  android-amd64  Android x86_64 (emulators)"
	@echo "  darwin-arm64   macOS Apple Silicon"
	@echo "  darwin-amd64   macOS Intel"
	@echo "  windows-amd64  Windows x86_64"
	@echo "  all            Build all platforms"
	@echo ""
	@echo "  Testing:"
	@echo "  test           Run all tests"
	@echo "  bench          Run benchmarks"
	@echo "  vet            Run go vet"
	@echo ""
	@echo "  Docker:"
	@echo "  docker         Build Docker image"
	@echo "  docker-run     Run Docker container"
	@echo ""
	@echo "  clean          Remove build artifacts"
