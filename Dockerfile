# iTaK Torch - Lightweight Docker image
# Uses a pre-built static binary (no Go compile needed).
#
# Build binary first: GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go build -o itaktorch_linux_amd64 ./cmd/itaktorch
# Then:  docker build -t itaktorch .
# Run:   docker run --gpus all -p 39271:39271 -v /path/to/models:/models itaktorch

FROM ubuntu:24.04

# Vulkan ICD + NVIDIA driver support
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates wget libvulkan1 mesa-vulkan-drivers \
    && rm -rf /var/lib/apt/lists/*

# NVIDIA GPU support (NVIDIA Container Toolkit)
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics

# Copy pre-built binary and libs
COPY itaktorch_linux_amd64 /usr/local/bin/itaktorch
RUN chmod +x /usr/local/bin/itaktorch
COPY lib/linux_amd64/ /app/lib/linux_amd64/

# Set lib path for llama.cpp backend
ENV ITAK_TORCH_LIB=/app/lib/linux_amd64
ENV LD_LIBRARY_PATH=/app/lib/linux_amd64:/usr/local/nvidia/lib64

WORKDIR /app
RUN mkdir -p /models
VOLUME /models

EXPOSE 39271

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD wget -q -O /dev/null http://localhost:39271/health || exit 1

ENTRYPOINT ["itaktorch"]
CMD ["serve", "--port", "39271", "--models-dir", "/models"]
