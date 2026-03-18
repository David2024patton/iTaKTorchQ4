# iTaK Torch - GPU-accelerated Docker image
# Uses pre-built binary + CUDA libs for full GPU inference.
#
# Build:  docker build -t itaktorch .
# Run:    docker run --gpus all -p 39271:39271 -v /path/to/models:/models itaktorch

FROM nvidia/cuda:12.8.0-runtime-ubuntu24.04

# Runtime deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates wget libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# NVIDIA GPU support
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Copy pre-built binary and CUDA libs
COPY itaktorch_linux_amd64 /usr/local/bin/itaktorch
RUN chmod +x /usr/local/bin/itaktorch
COPY lib/linux_amd64/ /app/lib/linux_amd64/

# Create .so.0 symlinks and register with ldconfig
RUN cd /app/lib/linux_amd64 && \
    ln -sf libllama.so libllama.so.0 && \
    ln -sf libggml.so libggml.so.0 && \
    ln -sf libggml-base.so libggml-base.so.0 && \
    ln -sf libggml-cuda.so libggml-cuda.so.0 && \
    echo "/app/lib/linux_amd64" > /etc/ld.so.conf.d/itaktorch.conf && \
    ldconfig

# Set lib path
ENV ITAK_TORCH_LIB=/app/lib/linux_amd64
ENV LD_LIBRARY_PATH=/app/lib/linux_amd64

WORKDIR /app
RUN mkdir -p /models
VOLUME /models

EXPOSE 39271

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD wget -q -O /dev/null http://localhost:39271/health || exit 1

ENTRYPOINT ["itaktorch"]
CMD ["serve", "--port", "39271", "--models-dir", "/models"]
