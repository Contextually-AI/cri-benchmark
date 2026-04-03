# CRI Benchmark — Examples

This directory contains example adapter implementations that demonstrate how to integrate different memory system architectures with the CRI Benchmark.

## Adapters

### [`no_memory_adapter.py`](adapters/no_memory_adapter.py)

A **baseline adapter with no memory**. It returns empty responses for all queries, establishing the lower bound for benchmark scores. Use this to verify that your benchmark setup is working and to understand what a "zero memory" score looks like.

### [`full_context_adapter.py`](adapters/full_context_adapter.py)

A **full-context window adapter** that stores all conversation messages and passes them to an LLM when answering queries. This represents the "brute force" approach — no structured memory, just raw context. It serves as an upper bound baseline for simple memory tasks.

### [`rag_adapter.py`](adapters/rag_adapter.py)

A **Retrieval-Augmented Generation (RAG) adapter** that uses TF-IDF vectorization and cosine similarity to retrieve relevant conversation snippets before answering queries. This demonstrates a common real-world memory architecture and typically scores between the no-memory and full-context baselines.

### [`upp_adapter.py`](adapters/upp_adapter.py)

A **UPP (Universal Personalization Protocol) adapter** that connects an ontology-based memory system to the CRI Benchmark. It uses the UPP SDK to store and retrieve structured knowledge. The `upp-python` package is a core dependency and is installed automatically.

## Running an Example

```bash
# Run with the no-memory baseline
cri run --adapter no-memory --dataset datasets/canonical/persona-1-base/ --format json --output results/

# Run with the full-context adapter
cri run --adapter full-context --dataset datasets/canonical/persona-1-base/ --format json --output results/

# Run with the RAG adapter
cri run --adapter rag --dataset datasets/canonical/persona-1-base/ --format json --output results/
```

## Creating Your Own Adapter

See the [Integration Guide](../docs/guides/integration.md) for step-by-step instructions on implementing the `MemoryAdapter` protocol for your own memory system.
