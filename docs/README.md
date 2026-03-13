# CRI Benchmark Documentation

Welcome to the CRI Benchmark documentation. This documentation is organized from high-level concepts to technical details.

## Navigation

### 🔭 Vision & Concepts
- [Project Vision](vision.md) — Why CRI exists and what it aims to achieve
- [What is AI Memory?](concepts/ai-memory.md) — Understanding AI memory systems
- [Ontology-Based Memory](concepts/ontology-memory.md) — Why ontology matters for memory
- [Benchmark Philosophy](concepts/benchmark-philosophy.md) — Design principles behind CRI

### 📏 Methodology
- [Methodology Overview](methodology/overview.md) — How CRI evaluation works
- [LLM-as-Judge](methodology/judge.md) — The judge-based evaluation approach
- [Datasets](methodology/datasets.md) — Dataset design and structure
- **Metrics:**
  - [PAS — Persona Accuracy Score](methodology/metrics/pas.md)
  - [DBU — Dynamic Belief Updating](methodology/metrics/dbu.md)
  - [MEI — Memory Efficiency Index](methodology/metrics/mei.md)
  - [TC — Temporal Coherence](methodology/metrics/tc.md)
  - [CRQ — Conflict Resolution Quality](methodology/metrics/crq.md)
  - [QRP — Query Response Precision](methodology/metrics/qrp.md)
  - [Composite CRI Score](methodology/metrics/composite-cri.md)

### 🏗️ Architecture
- [Architecture Overview](architecture/overview.md) — System design and components
- [Adapter Interface](architecture/adapter-interface.md) — How to connect memory systems
- [Scoring Engine](architecture/scoring-engine.md) — How scores are calculated
- [Data Flow](architecture/data-flow.md) — How data moves through the pipeline

### 📚 Guides
- [Quick Start](guides/quickstart.md) — Get running in 5 minutes
- [Integration Guide](guides/integration.md) — Connect your memory system
- [Adding New Metrics](guides/new-metrics.md) — Extend CRI with new dimensions
- [Creating New Datasets](guides/new-datasets.md) — Build custom benchmark datasets
- [Reproducibility](guides/reproducibility.md) — Ensuring reproducible results

### 🔬 Research
- [Ontology as Memory Analysis](research/ontology-as-memory-analysis.md)
- [AMA-Bench Analysis](research/ama-bench-analysis.md)
- [UPP Protocol Analysis](research/upp-protocol-analysis.md)
- [Contextually SDK Analysis](research/contextually-sdk-analysis.md)
- [Literature Review](research/literature-review.md)
