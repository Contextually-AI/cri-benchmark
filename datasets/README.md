# CRI Benchmark — Canonical Datasets

This directory contains the canonical benchmark datasets for the
**Contextual Resonance Index (CRI)** benchmark. Each dataset represents
a simulated multi-session conversation with a fictional persona, designed
to exercise all six CRI evaluation dimensions.

## Dataset Format

Each dataset directory contains:

| File | Format | Description |
|------|--------|-------------|
| `conversations.jsonl` | JSONL | One `Message` JSON object per line — the conversation stream |
| `ground_truth.json` | JSON | Complete `GroundTruth` object with expected outcomes |
| `metadata.json` | JSON | `DatasetMetadata` with provenance info (persona, seed, counts) |

### Message Schema (conversations.jsonl)

```json
{
  "message_id": 1,
  "role": "user",
  "content": "I work as a data analyst at a fintech startup here in Denver.",
  "timestamp": "2026-01-01T08:15:00",
  "session_id": "session-001",
  "day": 1
}
```

### Ground Truth Schema (ground_truth.json)

The ground truth file contains:

- **final_profile** — Expected profile dimensions the memory system should capture
- **changes** — Belief changes (old → new value) the system should track
- **noise_examples** — Messages that should NOT produce stored facts
- **signal_examples** — Messages that SHOULD produce stored facts
- **conflicts** — Contradictory statements the system must resolve
- **temporal_facts** — Facts with time-bounded validity
- **query_relevance_pairs** — Queries with expected relevant/irrelevant facts

## Canonical Personas

### Marcus Rivera (`persona-1-base`)

- **Messages**: 2862
- **Simulated Days**: 289
- **Profile Dimensions**: 18
- **Belief Changes**: 7
- **Conflicts**: 8
- **Temporal Facts**: 12
- **Query-Relevance Pairs**: 20
- **Signal Examples**: 20
- **Noise Examples**: 20

Marcus Rivera is a 41-year-old Puerto Rican-American architect who undergoes significant life changes over ~290 days: a career shift from architecture to sustainable building consulting, a move from Manhattan to Austin to rural Vermont, a divorce from Diana and new relationship with Elena, dietary evolution from standard to vegan to flexitarian, and a political awakening from apolitical to engaged.

## CRI Evaluation Dimensions Covered

Each dataset exercises all six dimensions:

| Dimension | Code | Exercised By |
|-----------|------|-------------|
| Persona Accuracy Score | PAS | Signal messages establishing profile facts |
| Dynamic Belief Updating | DBU | Belief change sequences (old → new value) |
| Memory Efficiency Index | MEI | Comparison of stored facts against ground-truth facts |
| Temporal Coherence | TC | Temporal facts with valid_from/valid_until |
| Conflict Resolution Quality | CRQ | Conflicting statements at specific points |
| Query Response Precision | QRP | Query-relevance pairs with expected results |

## Loading Datasets

```python
from cri.datasets.loader import load_dataset, list_canonical_datasets

# List all canonical datasets
datasets = list_canonical_datasets()
for ds in datasets:
    print(f"{ds.name}: {ds.message_count} messages, GT={ds.has_ground_truth}")

# Load a specific dataset
dataset = load_dataset("datasets/canonical/persona-1-base")
print(f"Messages: {len(dataset.messages)}")
print(f"Profile dims: {len(dataset.ground_truth.final_profile)}")
```

## Reproducibility

The persona-1-base dataset is hand-crafted (not procedurally generated)
to ensure high narrative quality and realistic conversation patterns.

## Extending

To add new datasets, see [docs/guides/new-datasets.md](../docs/guides/new-datasets.md).
