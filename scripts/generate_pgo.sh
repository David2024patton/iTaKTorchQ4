#!/bin/bash
# generate_pgo.sh - Generate a default PGO profile for iTaK Torch.
#
# This script:
#   1. Builds Torch normally
#   2. Starts a mock server with CPU profiling
#   3. Hammers it with representative requests for 30 seconds
#   4. Saves the profile as cmd/itaktorch/default.pgo
#
# Once default.pgo exists, every `go build ./cmd/itaktorch/` automatically
# uses PGO. No -pgo= flag needed. This is a Go 1.21+ convention.
#
# Usage:
#   ./scripts/generate_pgo.sh           # Use default port
#   ./scripts/generate_pgo.sh 55123     # Use custom port
#
# After running this, commit default.pgo to the repo so every build is faster.

set -e

PORT=${1:-55199}
PROFILE="cmd/itaktorch/default.pgo"
BINARY="itaktorch-pgo-gen"
URL="http://localhost:${PORT}"
DURATION=30

echo "=== iTaK Torch PGO Profile Generator ==="
echo "PORT:     ${PORT}"
echo "PROFILE:  ${PROFILE}"
echo "DURATION: ${DURATION}s"
echo ""

# Step 1: Build.
echo "[1/4] Building Torch..."
go build -o "${BINARY}" ./cmd/itaktorch/

# Step 2: Start server with profiling.
echo "[2/4] Starting mock server on port ${PORT} with CPU profiling..."
./"${BINARY}" serve --mock --port "${PORT}" --pgo-capture "${PROFILE}" &
SERVER_PID=$!

# Wait for server to be ready.
for i in $(seq 1 10); do
    if curl -s "${URL}/health" > /dev/null 2>&1; then
        break
    fi
    sleep 0.5
done

echo "[3/4] Sending ${DURATION}s of representative requests..."

# Mix of request patterns to capture realistic code paths.
END=$((SECONDS + DURATION))
COUNT=0
while [ $SECONDS -lt $END ]; do
    # Chat completions (most common endpoint).
    curl -s -X POST "${URL}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{"model":"mock","messages":[{"role":"user","content":"Hello, how are you?"}],"max_tokens":50}' > /dev/null &

    # Streaming chat.
    curl -s -X POST "${URL}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{"model":"mock","messages":[{"role":"system","content":"You are helpful."},{"role":"user","content":"Tell me about Go."}],"stream":true,"max_tokens":100}' > /dev/null &

    # Ollama compat endpoints.
    curl -s -X POST "${URL}/api/generate" \
        -H "Content-Type: application/json" \
        -d '{"model":"mock","prompt":"What is PGO?"}' > /dev/null &

    curl -s -X POST "${URL}/api/chat" \
        -H "Content-Type: application/json" \
        -d '{"model":"mock","messages":[{"role":"user","content":"Explain inference."}]}' > /dev/null &

    # Model listing.
    curl -s "${URL}/v1/models" > /dev/null &
    curl -s "${URL}/api/tags" > /dev/null &

    # Health check.
    curl -s "${URL}/health" > /dev/null &

    COUNT=$((COUNT + 7))
    wait
    sleep 0.1
done

echo "    Sent ~${COUNT} requests across all endpoint types."

# Step 4: Stop server (triggers profile save).
echo "[4/4] Stopping server and saving profile..."
kill "${SERVER_PID}" 2>/dev/null || true
wait "${SERVER_PID}" 2>/dev/null || true

# Clean up temp binary.
rm -f "${BINARY}"

# Verify profile was created.
if [ -f "${PROFILE}" ]; then
    SIZE=$(wc -c < "${PROFILE}" | tr -d ' ')
    echo ""
    echo "=== PGO profile generated ==="
    echo "  File: ${PROFILE}"
    echo "  Size: ${SIZE} bytes"
    echo ""
    echo "Every 'go build ./cmd/itaktorch/' will now use PGO automatically."
    echo "Commit ${PROFILE} to the repo so all builds are optimized."
else
    echo "ERROR: Profile was not created. Check server output above."
    exit 1
fi
