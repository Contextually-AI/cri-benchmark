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
LABEL org.opencontainers.image.source="https://github.com/cri-benchmark/cri"
LABEL org.opencontainers.image.licenses="MIT"

# Create non-root user
RUN groupadd --gid 1000 cri && \
    useradd --uid 1000 --gid cri --create-home cri

WORKDIR /app

# Install the wheel from builder stage
COPY --from=builder /build/wheels/*.whl /tmp/wheels/
RUN pip install --no-cache-dir /tmp/wheels/*.whl && \
    rm -rf /tmp/wheels

# Copy datasets and docs (read-only reference data)
COPY --chown=cri:cri datasets/ datasets/
COPY --chown=cri:cri docs/ docs/

# Create results directory
RUN mkdir -p /app/results && chown cri:cri /app/results

# Switch to non-root user
USER cri

# Default entrypoint
ENTRYPOINT ["cri"]
CMD ["--help"]
