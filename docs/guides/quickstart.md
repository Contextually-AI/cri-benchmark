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
 persona-1-basic              1000   ✓              datasets/canonical/persona-1-basic
 persona-2-intermediate       2000   ✓              datasets/canonical/persona-2-intermediate
 persona-3-advanced           3000   ✓              datasets/canonical/persona-3-advanced
```

Each dataset represents a different complexity level:

| Dataset | Persona | Profile Dims | Belief Changes | Conflicts |
|---------|---------|-------------|----------------|-----------|
| `persona-1-basic` | Alex Chen | 10 | 3 | 3 |
| `persona-2-intermediate` | Sarah Miller | 14 | 5 | 5 |
| `persona-3-advanced` | Marcus Rivera | 18 | 7 | 7 |

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

Start with the **no-memory** baseline on the simplest dataset:

```bash
cri run \
  --adapter no-memory \
  --dataset datasets/canonical/persona-1-basic \
  --verbose
```

> **Tip:** The no-memory adapter discards all input and scores zero on every dimension. This establishes the lower bound — any real memory system should score higher.

## Step 6 — Try the Full-Context Baseline

```bash
cri run \
  --adapter full-context \
  --dataset datasets/canonical/persona-1-basic \
  --verbose
```

The full-context adapter stores every user message verbatim and returns all of them on every query. It establishes an upper-bound for recall (but not precision).

## Step 7 — Save Results

Write structured results to a directory:

```bash
cri run \
  --adapter full-context \
  --dataset datasets/canonical/persona-1-basic \
  --output results/full-context-basic \
  --format json \
  --verbose
```

This creates:

```
results/full-context-basic/
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
cri validate-dataset datasets/canonical/persona-1-basic
```

```
Validating dataset: datasets/canonical/persona-1-basic

✓ Dataset is valid.
  Messages:           1000
  Profile dimensions: 10
  Belief changes:     3
  Conflicts:          3
  Temporal facts:     5
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
