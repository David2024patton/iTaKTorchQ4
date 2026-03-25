# iTaK Torch - Vulkan GPU-accelerated Docker image
# Stage 1: Compile llama.cpp with Vulkan support from source.
# Stage 2: Runtime image with pre-built Torch binary + Vulkan libs.
#
# Build:  docker build -t itaktorch .
# Run:    docker run --gpus all --device /dev/dri -p 39271:39271 -v /path/to/models:/models itaktorch

# ====== Stage 1: Build llama.cpp with Vulkan ======
FROM ubuntu:24.04 AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git ca-certificates \
    libvulkan-dev vulkan-tools glslang-tools glslc \
    && rm -rf /var/lib/apt/lists/*

# Clone and build llama.cpp with Vulkan backend as shared libs
RUN git clone --depth 1 https://github.com/ggml-org/llama.cpp.git /build/llama.cpp

WORKDIR /build/llama.cpp
RUN cmake -B build \
    -DGGML_VULKAN=ON \
    -DBUILD_SHARED_LIBS=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLAMA_BUILD_EXAMPLES=OFF \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_SERVER=OFF \
    && cmake --build build --config Release -j$(nproc)

# Collect all .so files into a staging directory so we don't need to guess paths
RUN mkdir -p /staging && \
    find /build/llama.cpp/build -name "*.so" -exec cp {} /staging/ \; && \
    ls -la /staging/

# ====== Stage 2: Runtime ======
FROM ubuntu:24.04

# Runtime deps: Vulkan ICD loader + NVIDIA Vulkan driver support + mesa for fallback
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates wget libgomp1 \
    libvulkan1 mesa-vulkan-drivers vulkan-tools \
    && rm -rf /var/lib/apt/lists/*

# NVIDIA GPU support (Vulkan uses the NVIDIA ICD via the driver)
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics

# Copy all compiled Vulkan llama.cpp shared libs from staging
COPY --from=builder /staging/ /app/lib/linux_amd64/

# Copy pre-built Torch binary
COPY torch_linux_amd64 /usr/local/bin/torch
RUN chmod +x /usr/local/bin/torch

# Create symlinks and register with ldconfig
RUN cd /app/lib/linux_amd64 && \
    for f in *.so; do ln -sf "$f" "${f}.0" 2>/dev/null; done && \
    echo "/app/lib/linux_amd64" > /etc/ld.so.conf.d/torch.conf && \
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

ENTRYPOINT ["torch"]
CMD ["serve", "--port", "39271", "--models-dir", "/models", "--backend", "vulkan"]
