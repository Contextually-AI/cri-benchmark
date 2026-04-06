# Quick Start Guide

Get from zero to your first CRI Benchmark result in under 10 minutes.

## Prerequisites

- Python 3.11 or later
- An Anthropic OAuth subscription token for the default judge model (or a custom LangChain LLM)

## Step 1 — Install

### From PyPI

```bash
pip install cri-benchmark
```

### From Source

```bash
git clone https://github.com/Contextually-AI/cri-benchmark.git
cd cri-benchmark
pip install -e ".[dev]"
```

## Step 2 — Configure the LLM Judge

The CRI judge uses Claude Haiku as the default model via an Anthropic OAuth subscription token. Place your token in a file one level above the project root:

```bash
echo "sk-ant-oat01-..." > ../.auth_token
```

To use a different LLM provider, pass a custom `llm_factory` to `run_benchmark()` programmatically.
See the [integration guide](integration.md) for details.

## Step 3 — Explore Available Datasets

```bash
cri list-datasets
```

Expected output:

```
═══ Canonical Datasets ═══

 Name                     Messages   Ground Truth   Path
 persona-1-base               2862   ✓              src/cri/datasets/persona-1-base
```

| Dataset | Persona | Messages | Profile Dims | Belief Changes | Conflicts |
|---------|---------|----------|-------------|----------------|-----------|
| `persona-1-base` | Marcus Rivera | 2862 | 18 | 7 | 8 |

## Step 4 — Explore Available Adapters

```bash
cri list-adapters
```

Expected output:

```
═══ Registered Adapters ═══

 Name            Description                                                          Available
 full-context    Stores every user message; returns all on query. Upper-bound recall   ✓
                 baseline.
 no-memory       Discards all input; returns nothing. Lower-bound baseline.            ✓
 rag             Simple ChromaDB vector-store RAG adapter. Requires                    ✗
                 'pip install cri-benchmark[rag]'.
```

## Step 5 — Run Your First Benchmark

Start with the **no-memory** baseline:

```bash
cri run \
  --adapter no-memory \
  --dataset src/cri/datasets/persona-1-base \
  --verbose
```

> **Tip:** The no-memory adapter discards all input and scores zero on every dimension. This establishes the lower bound — any real memory system should score higher.

## Step 6 — Try the Full-Context Baseline

```bash
cri run \
  --adapter full-context \
  --dataset src/cri/datasets/persona-1-base \
  --verbose
```

The full-context adapter stores every user message verbatim and returns all of them on every query. It establishes an upper-bound for recall (but not precision).

## Step 7 — Save Results

Write structured results to a directory:

```bash
cri run \
  --adapter full-context \
  --dataset src/cri/datasets/persona-1-base \
  --output results/full-context-base \
  --format json \
  --verbose
```

This creates:

```
results/full-context-base/
├── result.json        # Complete BenchmarkResult (CRI scores + performance)
└── judge_log.json     # Full log of every judge evaluation
```

## Step 8 — Run with Docker

The `run.sh` script builds a Docker container and runs the benchmark. All parameters are optional:

```bash
# Run all adapters against all datasets
./run.sh

# Smoke test with message limit
./run.sh --limit 50

# Single adapter
./run.sh --adapter rag --limit 50
```

See the [README](../../README.md#run-with-docker-runsh) for the full parameter reference.

## Step 9 — Validate a Dataset

Before running a benchmark, you can validate a dataset's structure:

```bash
cri validate-dataset src/cri/datasets/persona-1-base
```

```
Validating dataset: src/cri/datasets/persona-1-base

✓ Dataset is valid.
  Messages:           2862
  Profile dimensions: 18
  Belief changes:     7
  Conflicts:          8
  Temporal facts:     12
```

## Benchmark Pipeline

```mermaid
graph LR
    A[Load Dataset] --> B[Instantiate Adapter]
    B --> C[Ingest Messages]
    C --> D[Run Scoring Engine]
    D --> E[LLM Judge Evaluations]
    E --> F[Compute CRI Score]
    F --> G[Generate Report]
```

## Next Steps

| Goal | Guide |
|------|-------|
| Integrate your own memory system | [Integration Guide](integration.md) |
| Understand the evaluation dimensions | [README — Evaluation Dimensions](../../README.md#evaluation-dimensions) |
| Understand the methodology | [Methodology Overview](../../METHODOLOGY.md) |
