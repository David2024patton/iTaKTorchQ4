# iTaK Torch - Lightweight Docker image
# Uses a pre-built static binary (cross-compiled with CGO_ENABLED=0).
# Build the binary first: go build -ldflags="-s -w" -o itaktorch ./cmd/itaktorch
# Then: docker build -t itaktorch .

FROM ubuntu:24.04

# Install ca-certificates for HTTPS model downloads
RUN apt-get update && apt-get install -y ca-certificates wget && rm -rf /var/lib/apt/lists/*

# NVIDIA GPU support (matches Ollama's container pattern).
# These env vars tell the NVIDIA Container Toolkit to expose GPUs.
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64

# Copy the pre-built static binary
COPY itaktorch /usr/local/bin/itaktorch
RUN chmod +x /usr/local/bin/itaktorch

# Model directory - mount your models here
RUN mkdir -p /root/.itaktorch/models
VOLUME /root/.itaktorch/models

# Default port
EXPOSE 39271

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD wget -q -O /dev/null http://localhost:39271/health || exit 1

ENTRYPOINT ["itaktorch"]
CMD ["serve", "--port", "39271", "--models-dir", "/root/.itaktorch/models"]
