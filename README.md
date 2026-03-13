<p align="center">
  <h1 align="center">CRI Benchmark — Contextual Resonance Index</h1>
  <p align="center">
    <strong>The open-source standard for evaluating AI long-term memory systems</strong>
  </p>
</p>

<p align="center">
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+"></a>
  <a href="https://github.com/cri-benchmark/cri/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/cri-benchmark/cri/ci.yml?label=CI" alt="CI Status"></a>
  <a href="https://pypi.org/project/cri-benchmark/"><img src="https://img.shields.io/pypi/v/cri-benchmark?color=green&label=PyPI" alt="PyPI Version"></a>
</p>

---

The **Contextual Resonance Index (CRI)** is a benchmark framework designed to evaluate how well AI systems maintain, update, and utilize contextual knowledge about users and entities over time. It measures the quality of long-term memory — not just what a system can retrieve, but how accurately it captures evolving facts, resolves contradictions, handles temporal knowledge, and maintains coherent representations across hundreds of interactions. CRI provides a transparent, reproducible, and scientifically grounded evaluation methodology that any memory system can adopt through a minimal adapter interface.

---

## What CRI Measures

Existing AI benchmarks focus on retrieval accuracy or downstream task performance. **CRI evaluates the knowledge model itself** — what was stored, what was updated, what was correctly rejected, and how coherently knowledge evolves over time.

This is critical for memory systems that go beyond naive RAG or append-only logs: ontology-based architectures, knowledge graphs, user profiling engines, and any system where structured understanding matters more than raw recall.

> 📏 Read the full [Evaluation Methodology →](docs/methodology/overview.md)

## Key Features

- 🎯 **Seven scored dimensions** — each measuring a distinct property of memory behavior
- ⚖️ **Transparent composite score** — weighted formula with published justification for every weight
- 🤖 **Hybrid scoring** — deterministic checks where possible, LLM-as-judge with majority voting for semantic evaluation
- 🔌 **3-method adapter interface** — integrate any memory system with minimal effort
- 📊 **Canonical datasets** — pre-generated personas at basic, intermediate, and advanced complexity
- 🛠️ **Dataset generator** — create custom scenarios for your specific use cases
- 📈 **Performance profiling** — latency and memory growth reported alongside quality scores
- 🔬 **Fully reproducible** — seeded randomness, logged prompts, deterministic pipeline
- 🧩 **Extensible** — add new metrics, datasets, and adapters without modifying the core engine

## Architecture

```mermaid
graph LR
    subgraph Input
        D[📁 Dataset<br><i>Events + Ground Truth</i>]
    end

    subgraph "System Under Test"
        A[🔌 Adapter<br><i>Your Memory System</i>]
    end

    subgraph Evaluation
        S[📐 Scoring Engine<br><i>7 Dimensions + Judge</i>]
    end

    subgraph Output
        R[📊 Reporter<br><i>JSON · Markdown · Console</i>]
    end

    D -- "events" --> A
    A -- "stored facts<br>+ query responses" --> S
    D -- "ground truth" --> S
    S -- "dimension scores<br>+ composite CRI" --> R
```

The benchmark pipeline is simple: **Dataset → Adapter → Scorer → Reporter**. Your memory system only needs to implement the Adapter — everything else is handled by CRI.

## Evaluation Dimensions

CRI evaluates memory systems across **seven scored dimensions**, each targeting a distinct property of long-term knowledge management:

| Code | Dimension | What It Measures |
|------|-----------|-----------------|
| **PAS** | Profile Accuracy Score | Does the system accurately capture and recall entity facts? |
| **DBU** | Dynamic Belief Updating | When facts change, does the system update its beliefs? |
| **MEI** | Memory Efficiency Index | Does the system store exactly what it should — no more, no less? |
| **TC** | Temporal Consistency | Does the system handle time-bounded and expiring knowledge? |
| **CRQ** | Conflict Resolution Quality | When contradictory information arrives, is it resolved correctly? |
| **QRP** | Query Relevance Precision | Are retrieved facts relevant to the query, and irrelevant facts excluded? |
| **SFC** | Selective Forgetting Capability | Does the system appropriately discard ephemeral or superseded information? |

> 📐 Detailed definitions: [PAS](docs/methodology/metrics/pas.md) · [DBU](docs/methodology/metrics/dbu.md) · [MEI](docs/methodology/metrics/mei.md) · [TC](docs/methodology/metrics/tc.md) · [CRQ](docs/methodology/metrics/crq.md) · [QRP](docs/methodology/metrics/qrp.md) · [SFC](docs/methodology/metrics/sfc.md)

### Composite Score

The CRI composite score combines dimensions with published, configurable weights (shown here for the CORE profile):

```
CRI = 0.25 × PAS + 0.20 × DBU + 0.20 × MEI + 0.15 × TC + 0.10 × CRQ + 0.10 × QRP
```

All dimension scores are normalized to **[0.0, 1.0]**. The composite weights reflect the relative importance of each capability for real-world memory systems: accurate knowledge capture (PAS), belief evolution (DBU), and storage efficiency (MEI) are weighted highest, while conflict resolution (CRQ) and query precision (QRP) serve as important secondary indicators.

> 📊 Full formula justification: [Composite CRI →](docs/methodology/metrics/composite-cri.md)

**Non-scored performance profiles** — latency, memory growth, and cost — are reported alongside the CRI score but intentionally kept separate. Quality and performance are different concerns and should not be conflated.

## Quick Start

### Install

```bash
pip install cri-benchmark
```

Or from source:

```bash
git clone https://github.com/cri-benchmark/cri.git
cd cri
pip install -e ".[dev]"
```

#### Full setup (includes private dependencies)

For a complete setup including private dependencies (e.g., UPP adapter), use the init script:

```bash
./init.sh
```

This script authenticates with AWS CodeArtifact and installs all dependencies in one step.

> **Prerequisites:** [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html) configured with credentials that have access to the CodeArtifact repository.

### Configure the LLM Judge

CRI uses an LLM as judge for semantic evaluation. Set your API key:

```bash
export OPENAI_API_KEY="sk-..."
# Or any provider supported by litellm:
# export ANTHROPIC_API_KEY="sk-ant-..."
```

### Run Your First Benchmark

```bash
# Run against the built-in baseline adapter
cri run --adapter examples/adapters/full_context_adapter.py --dataset persona-1-basic

# Compare multiple systems
cri run \
  --adapter examples/adapters/full_context_adapter.py \
  --adapter examples/adapters/rag_adapter.py \
  --dataset persona-1-basic

# List available datasets
cri list-datasets
```

**Expected output:**

```
CRI Benchmark v0.1.0 — persona-1-basic
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Adapter: full-context-adapter

  PAS  ██████████████████░░  0.89
  DBU  ████████████████░░░░  0.81
  MEI  █████████████████░░░  0.85
  TC   ██████████████░░░░░░  0.72
  CRQ  ████████████████░░░░  0.78
  QRP  ███████████████████░  0.91

  ─────────────────────────────────
  CRI Composite Score:  0.83
```

### Run Programmatically

```python
from cri.adapter import MemoryAdapter
from cri.models import Message, StoredFact
from cri.runner import BenchmarkRunner

class MyMemoryAdapter:
    """Adapter for my memory system — satisfies MemoryAdapter via structural subtyping."""

    def ingest(self, messages: list[Message]) -> None:
        # Feed the messages into your memory system
        for msg in messages:
            self.my_system.process(msg.content, role=msg.role, ts=msg.timestamp)

    def query(self, topic: str) -> list[StoredFact]:
        # Retrieve facts relevant to the topic
        results = self.my_system.search(topic)
        return [StoredFact(text=r.text, metadata={"score": r.score}) for r in results]

    def get_all_facts(self) -> list[StoredFact]:
        # Return every stored fact for memory-hygiene auditing
        return [StoredFact(text=f.text, metadata=f.meta) for f in self.my_system.dump_all()]

runner = BenchmarkRunner(adapter=MyMemoryAdapter())
result = runner.run(dataset="persona-1-basic")
print(f"CRI Score: {result.cri_result.cri:.2f}")
```

## Implement Your Own Adapter

Connecting your memory system to CRI requires implementing **three methods**:

| Method | Signature | Purpose |
|--------|-----------|---------|
| `ingest` | `(messages: list[Message]) -> None` | Process and store conversation messages into your memory system |
| `query` | `(topic: str) -> list[StoredFact]` | Retrieve facts relevant to a given topic |
| `get_all_facts` | `() -> list[StoredFact]` | Return every stored fact for memory-hygiene auditing |

Because `MemoryAdapter` uses structural subtyping (a `typing.Protocol`), your class does **not** need to inherit from it — just implement the three methods with compatible signatures.

That's it. No complex protocols, no proprietary formats, no infrastructure requirements.

> 🔌 Full integration walkthrough: [Integration Guide →](docs/guides/integration.md)

## Documentation

### Concepts
- 📖 [Project Vision](docs/vision.md) — why CRI exists and where it's going
- 🧠 [What is AI Memory?](docs/concepts/ai-memory.md) — the problem space
- 🌐 [Ontology-Based Memory](docs/concepts/ontology-memory.md) — why structured memory matters
- 💡 [Benchmark Philosophy](docs/concepts/benchmark-philosophy.md) — design principles behind CRI

### Methodology
- 📏 [Evaluation Overview](docs/methodology/overview.md) — how CRI evaluates memory systems
- 📐 [Metric Definitions](docs/methodology/metrics/) — detailed specification of each dimension
- ⚖️ [LLM Judge Design](docs/methodology/judge.md) — how semantic evaluation works
- 📁 [Dataset Design](docs/methodology/datasets.md) — canonical scenarios and generation

### Architecture
- 🏗️ [Architecture Overview](docs/architecture/overview.md) — system design and data flow
- 🔌 [Adapter Interface](docs/architecture/adapter-interface.md) — the integration contract
- ⚙️ [Scoring Engine](docs/architecture/scoring-engine.md) — how scores are computed
- 🔄 [Data Flow](docs/architecture/data-flow.md) — end-to-end pipeline walkthrough

### Guides
- 🚀 [Quick Start](docs/guides/quickstart.md) — run your first benchmark in under 10 minutes
- 🔌 [Integration Guide](docs/guides/integration.md) — implement an adapter step by step
- 📐 [Adding New Metrics](docs/guides/new-metrics.md) — extend CRI with custom dimensions
- 📁 [Creating Datasets](docs/guides/new-datasets.md) — build scenarios for your domain
- 🔬 [Reproducibility](docs/guides/reproducibility.md) — ensuring consistent results

### Research
- 📄 [Literature Review](docs/research/) — analysis of related work and prior art

## Example Adapters

The repository includes reference adapter implementations to help you get started:

| Adapter | Description |
|---------|-------------|
| [`full_context`](examples/adapters/) | Sends all events as LLM context — strong but expensive baseline |
| [`rag`](examples/adapters/) | ChromaDB-backed retrieval — standard vector store approach |
| [`no_memory`](examples/adapters/) | Answers with no context — useful lower bound |
| [`upp`](examples/adapters/) | UPP ontology-based memory — structured knowledge via the UPP protocol |

## Contributing

We welcome contributions! Whether it's new evaluation dimensions, datasets, adapter implementations, documentation improvements, or bug fixes — all contributions help CRI become a better standard.

Please read our [Contributing Guide](CONTRIBUTING.md) before submitting a pull request.

Areas where contributions are especially welcome:
- 🔌 Adapter implementations for popular memory systems
- 📐 New evaluation dimensions with published justification
- 📁 Domain-specific datasets
- 📖 Documentation improvements and translations
- 🧪 Test coverage

## Project Status

CRI is in **active early development** (v0.1.0). The core framework, seven evaluation dimensions, and canonical datasets are being established. We aim to release a stable v1.0 once the methodology has been validated through community feedback and real-world usage.

## License

[MIT](LICENSE) — use it, extend it, contribute back.

---

<p align="center">
  <i>CRI Benchmark is an open-source project aiming to become the industry-standard reference for evaluating AI long-term memory systems.</i>
</p>
