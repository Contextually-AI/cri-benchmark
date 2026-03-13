# Quick Start Guide

Get from zero to your first CRI Benchmark result in under 10 minutes.

## Prerequisites

- Python 3.10 or later
- An LLM API key (Anthropic or OpenAI) for the judge model

## Step 1 — Install

### From PyPI

```bash
pip install cri-benchmark
```

### From Source

```bash
git clone https://github.com/cri-benchmark/cri.git
cd cri
pip install -e ".[dev]"
```

## Step 2 — Set Your API Key

The CRI judge uses [LiteLLM](https://litellm.ai) under the hood, so any supported provider works:

```bash
# Anthropic (recommended — default judge model is claude-haiku-4-5)
export ANTHROPIC_API_KEY="sk-ant-..."

# or OpenAI
export OPENAI_API_KEY="sk-..."
```

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
 upp             UPP ontology-based memory adapter. Requires                           ✗
                 'pip install cri-benchmark[upp]'.
```

## Step 5 — Run Your First Benchmark

Start with the **no-memory** baseline on the simplest dataset:

```bash
cri run \
  --adapter no-memory \
  --dataset datasets/canonical/persona-1-basic \
  --verbose
```

Expected output:

```
═══ CRI Benchmark ═══
  Adapter:     no-memory
  Dataset:     datasets/canonical/persona-1-basic
  Judge model: claude-haiku-4-5-20250315
  Judge runs:  3
  Format:      console

✓ Dataset loaded: 1000 messages, 10 profile dimensions
Ingesting messages...
✓ Ingested 1000 messages
Running evaluation across all dimensions...

═══════════════════════════════════════════════════
  CRI Benchmark Results — no-memory
═══════════════════════════════════════════════════

  Composite CRI:  0.0000

  Dimension Scores:
  ┌────────────┬─────────┬────────┬───────┐
  │ Dimension  │ Score   │ Passed │ Total │
  ├────────────┼─────────┼────────┼───────┤
  │ PAS        │ 0.0000  │ 0      │ 10    │
  │ DBU        │ 0.0000  │ 0      │ 3     │
  │ MEI        │ 0.0000  │ 0      │ 10    │
  │ TC         │ 0.0000  │ 0      │ 5     │
  │ CRQ        │ 0.0000  │ 0      │ 3     │
  │ QRP        │ 0.0000  │ 0      │ 10    │
  └────────────┴─────────┴────────┴───────┘
```

> **This is expected!** The no-memory adapter discards all input, so it scores zero on every dimension. This establishes the lower bound. Any real memory system should score higher.

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

## Step 8 — Validate a Dataset

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

## Understanding the Scores

The CRI Benchmark evaluates memory systems across six dimensions:

| Dimension | Code | Weight | What It Measures |
|-----------|------|--------|------------------|
| Persona Accuracy Score | PAS | 25% | Can the system recall stored profile facts? |
| Dynamic Belief Updating | DBU | 20% | Does the system update beliefs when facts change? |
| Memory Efficiency Index | MEI | 20% | Does the system store knowledge efficiently with good coverage? |
| Temporal Coherence | TC | 15% | Does the system track temporal validity of facts? |
| Conflict Resolution Quality | CRQ | 10% | Can the system resolve contradictory information? |
| Query Response Precision | QRP | 10% | Does the system return relevant facts without noise? |

The **composite CRI score** is the weighted average of all dimension scores (0.0 – 1.0).

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
| Add a new evaluation metric | [Adding New Metrics](new-metrics.md) |
| Create custom benchmark datasets | [Adding New Datasets](new-datasets.md) |
| Ensure reproducible results | [Reproducibility Guide](reproducibility.md) |
| Understand the methodology | [Methodology Overview](../methodology/overview.md) |
| Explore metric definitions | [Metric Documentation](../methodology/metrics/composite-cri.md) |
