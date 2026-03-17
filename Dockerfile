# ── CRI Benchmark — Contextual Resonance Index ──────────────────
# Multi-stage build for minimal production image
#
# Usage:
#   docker build -t cri-benchmark .
#   docker run --rm cri-benchmark --help
#   docker run --rm -v $(pwd)/results:/app/results cri-benchmark run ...

# ── Stage 1: Build ──────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /build

# Install build deps
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy only packaging files first for layer caching
COPY pyproject.toml README.md LICENSE CHANGELOG.md ./
COPY src/ src/

# Build the wheel
RUN pip wheel --no-deps --wheel-dir /build/wheels .

# ── Stage 2: Runtime ───────────────────────────────────────────
FROM python:3.12-slim AS runtime

LABEL maintainer="CRI Benchmark Contributors"
LABEL description="CRI Benchmark — Contextual Resonance Index"
LABEL org.opencontainers.image.source="https://github.com/Contextually-AI/cri-benchmark"
LABEL org.opencontainers.image.licenses="MIT"

# Create non-root user
RUN groupadd --gid 1000 cri && \
    useradd --uid 1000 --gid cri --create-home cri

WORKDIR /app

# Install the wheel from builder stage
COPY --from=builder /build/wheels/*.whl /tmp/wheels/
RUN pip install --no-cache-dir /tmp/wheels/*.whl && \
    pip install --no-cache-dir chromadb>=0.4 && \
    rm -rf /tmp/wheels

# Copy datasets, docs, and example adapters
COPY --chown=cri:cri datasets/ datasets/
COPY --chown=cri:cri docs/ docs/
COPY --chown=cri:cri examples/ examples/

# Ensure examples/ (adapters) is importable
ENV PYTHONPATH=/app

# Create results directory and cache for ChromaDB ONNX model
RUN mkdir -p /app/results && chown cri:cri /app/results && \
    mkdir -p /.cache && chmod 777 /.cache

# Pre-download ChromaDB ONNX embedding model so it doesn't download at runtime
RUN apt-get update && apt-get install -y --no-install-recommends curl && \
    mkdir -p /root/.cache/chroma/onnx_models/all-MiniLM-L6-v2 && \
    curl -SL --retry 3 --retry-delay 5 -o /root/.cache/chroma/onnx_models/all-MiniLM-L6-v2/onnx.tar.gz \
        https://chroma-onnx-models.s3.amazonaws.com/all-MiniLM-L6-v2/onnx.tar.gz && \
    tar -xzf /root/.cache/chroma/onnx_models/all-MiniLM-L6-v2/onnx.tar.gz \
        -C /root/.cache/chroma/onnx_models/all-MiniLM-L6-v2/ && \
    cp -r /root/.cache/chroma /.cache/chroma && \
    chmod -R 777 /.cache/chroma && \
    apt-get purge -y curl && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

# Switch to non-root user
USER cri

# Default entrypoint
ENTRYPOINT ["cri"]
CMD ["--help"]
