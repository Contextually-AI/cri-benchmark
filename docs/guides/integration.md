# Integration Guide

This guide walks you through integrating your memory system with the CRI Benchmark, from implementing the adapter interface to running your first evaluation.

## Overview

The CRI Benchmark communicates with memory systems through a simple **adapter protocol**. Your system needs to implement three methods — that's it.

```mermaid
graph LR
    subgraph Your System
        A[Memory Engine]
    end
    subgraph CRI Benchmark
        B[Runner] --> C[Scoring Engine]
        C --> D[LLM Judge]
    end
    B -- "ingest(messages)" --> A
    B -- "query(topic)" --> A
    B -- "get_all_facts()" --> A
    A -- "list[StoredFact]" --> B
```

## The MemoryAdapter Protocol

The `MemoryAdapter` is defined as a Python `Protocol` with structural subtyping. This means:

- **No inheritance required** — your class doesn't need to import or subclass anything from CRI
- **No dependency on CRI** — you can implement the interface in your own package
- **Runtime verification** — the runner checks compliance with `isinstance()` before starting

### The Three Methods

| Method | Purpose | Input | Output |
|--------|---------|-------|--------|
| `ingest(messages)` | Process conversation messages and store extracted facts | `list[Message]` | `None` |
| `query(topic)` | Retrieve facts relevant to a topic string | `str` | `list[StoredFact]` |
| `get_all_facts()` | Dump the entire fact store for auditing | — | `list[StoredFact]` |

### Data Models

```python
from pydantic import BaseModel

class Message(BaseModel):
    message_id: int          # Sequential identifier
    role: str                # "user" or "assistant"
    content: str             # Message text
    timestamp: str           # ISO-8601 timestamp
    session_id: str | None   # Optional session grouping
    day: int | None          # Simulation day number

class StoredFact(BaseModel):
    text: str                # Textual content of the fact
    metadata: dict           # Arbitrary metadata (scores, timestamps, etc.)
```

## Step-by-Step Integration

### Step 1 — Create Your Adapter Class

```python
# my_adapter.py
from cri.models import Message, StoredFact


class MyMemoryAdapter:
    """Adapter for the Acme Memory Engine."""

    def __init__(self):
        self._facts: list[StoredFact] = []

    def ingest(self, messages: list[Message]) -> None:
        """Process messages and extract facts."""
        for msg in messages:
            if msg.role == "user":
                # Your fact extraction logic here
                extracted = self._extract_facts(msg.content)
                for fact_text in extracted:
                    self._facts.append(
                        StoredFact(
                            text=fact_text,
                            metadata={
                                "source_msg": msg.message_id,
                                "timestamp": msg.timestamp,
                            },
                        )
                    )

    def query(self, topic: str) -> list[StoredFact]:
        """Retrieve facts relevant to the given topic."""
        relevant = []
        for fact in self._facts:
            if self._is_relevant(fact.text, topic):
                relevant.append(fact)
        return relevant

    def get_all_facts(self) -> list[StoredFact]:
        """Return all stored facts."""
        return list(self._facts)

    # -- Your internal logic --

    def _extract_facts(self, content: str) -> list[str]:
        """Extract factual statements from message content."""
        # Replace with your extraction pipeline
        # (NLP, LLM, regex, ontology builder, etc.)
        return [content]  # naive: treat entire message as a fact

    def _is_relevant(self, fact_text: str, topic: str) -> bool:
        """Check if a fact is relevant to a topic."""
        # Replace with your relevance logic
        # (embedding similarity, keyword match, ontology traversal, etc.)
        return topic.lower() in fact_text.lower()
```

> **Key insight**: The benchmark doesn't prescribe _how_ your system extracts facts, determines relevance, or structures its internal knowledge. It only cares about the inputs and outputs of these three methods.

### Step 2 — Verify Protocol Compliance

```python
from cri.adapter import MemoryAdapter
from my_adapter import MyMemoryAdapter

adapter = MyMemoryAdapter()
assert isinstance(adapter, MemoryAdapter), "Adapter doesn't satisfy protocol!"
print("✓ Adapter is protocol-compliant")
```

### Step 3 — Run via CLI

The simplest way to run your adapter is through the CLI with a dotted import path:

```bash
cri run \
  --adapter my_adapter:MyMemoryAdapter \
  --dataset datasets/canonical/persona-1-basic \
  --verbose
```

The `--adapter` flag accepts either:
- A **registry name**: `no-memory`, `full-context`, `rag`, `upp`
- A **dotted path**: `my_package.adapters:MyMemoryAdapter` or `my_package.adapters.MyMemoryAdapter`

### Step 4 — Run Programmatically

For more control, use the Python API:

```python
import asyncio
from pathlib import Path

from cri.runner import run_benchmark
from my_adapter import MyMemoryAdapter


async def main():
    result = await run_benchmark(
        adapter_name="my_adapter:MyMemoryAdapter",
        dataset_path="datasets/canonical/persona-1-basic",
        judge_model="claude-haiku-4-5-20250315",
        judge_runs=3,
        output_dir="results/my-system",
        output_format="json",
        verbose=True,
    )

    print(f"Composite CRI: {result.cri_result.cri}")
    print(f"PAS: {result.cri_result.pas}")
    print(f"DBU: {result.cri_result.dbu}")


asyncio.run(main())
```

## Advanced Integration Patterns

### Wrapping an Existing System

If your memory system already has its own API, create a thin adapter wrapper:

```python
from cri.models import Message, StoredFact


class OntologyMemoryAdapter:
    """Adapter wrapping an ontology-based memory system."""

    def __init__(self, endpoint: str = "http://localhost:8080"):
        import requests
        self._session = requests.Session()
        self._endpoint = endpoint

    def ingest(self, messages: list[Message]) -> None:
        # Convert CRI messages to your system's event format
        events = [
            {
                "text": msg.content,
                "role": msg.role,
                "timestamp": msg.timestamp,
                "session": msg.session_id,
            }
            for msg in messages
        ]
        self._session.post(
            f"{self._endpoint}/ingest",
            json={"events": events},
        )

    def query(self, topic: str) -> list[StoredFact]:
        resp = self._session.get(
            f"{self._endpoint}/query",
            params={"topic": topic},
        )
        results = resp.json()["results"]
        return [
            StoredFact(
                text=r["text"],
                metadata={"confidence": r.get("confidence", 1.0)},
            )
            for r in results
        ]

    def get_all_facts(self) -> list[StoredFact]:
        resp = self._session.get(f"{self._endpoint}/facts")
        facts = resp.json()["facts"]
        return [
            StoredFact(text=f["text"], metadata=f.get("metadata", {}))
            for f in facts
        ]
```

### Adapter with Setup and Teardown

If your adapter needs initialization (loading models, connecting to databases), handle it in `__init__`:

```python
class EmbeddingMemoryAdapter:
    def __init__(self):
        # Initialize your system here
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer("all-MiniLM-L6-v2")
        self._facts: list[StoredFact] = []
        self._embeddings = []

    def ingest(self, messages: list[Message]) -> None:
        for msg in messages:
            if msg.role == "user":
                self._facts.append(StoredFact(text=msg.content))
                self._embeddings.append(
                    self._model.encode(msg.content)
                )

    def query(self, topic: str) -> list[StoredFact]:
        import numpy as np
        topic_emb = self._model.encode(topic)
        similarities = [
            np.dot(topic_emb, emb) / (np.linalg.norm(topic_emb) * np.linalg.norm(emb))
            for emb in self._embeddings
        ]
        # Return top-5 most relevant facts
        top_indices = np.argsort(similarities)[-5:][::-1]
        return [self._facts[i] for i in top_indices if similarities[i] > 0.3]

    def get_all_facts(self) -> list[StoredFact]:
        return list(self._facts)
```

## What the Benchmark Evaluates

When your adapter runs, the benchmark:

1. **Ingests** all conversation messages in chronological order via `ingest()`
2. **Queries** your system with topic strings derived from the ground truth via `query()`
3. **Audits** your fact store via `get_all_facts()` for noise, duplication, and completeness
4. **Judges** each check using an LLM that compares your system's output against expected answers

### What Makes a Good Adapter

| Property | Why It Matters |
|----------|---------------|
| **Fact extraction** | Filter noise (greetings, filler) from signal (facts about the user) |
| **Knowledge updates** | When a user says "I moved to Denver," update the stored location |
| **Conflict resolution** | When contradictory info appears, resolve to the most recent/authoritative |
| **Temporal tracking** | Know which facts are current vs. historical |
| **Precise retrieval** | Return only relevant facts for a query, not everything |

## Troubleshooting

### "Adapter does not satisfy MemoryAdapter protocol"

Ensure your class has all three methods with the correct signatures:
- `ingest(self, messages: list[Message]) -> None`
- `query(self, topic: str) -> list[StoredFact]`
- `get_all_facts(self) -> list[StoredFact]`

### "Cannot resolve adapter"

If using a dotted path, ensure:
- The module is importable (on `sys.path` or installed)
- The path uses either `module.path:ClassName` or `module.path.ClassName` format
- The class name is spelled correctly

### Low CRI Scores

- **Low PAS**: Your system isn't extracting profile facts from messages
- **Low DBU**: Your system isn't updating facts when new information arrives
- **Low MEI**: Your system has poor memory efficiency — low coverage or too much noise
- **Low TC**: Your system doesn't track temporal validity of facts
- **Low CRQ**: Your system can't resolve conflicting information correctly
- **Low QRP**: Your system returns too many irrelevant facts or misses relevant ones

## Next Steps

- [Quick Start Guide](quickstart.md) — Run built-in baselines
- [Adding New Metrics](new-metrics.md) — Define custom evaluation dimensions
- [Adding New Datasets](new-datasets.md) — Create custom benchmark scenarios
- [Reproducibility Guide](reproducibility.md) — Ensure consistent results
