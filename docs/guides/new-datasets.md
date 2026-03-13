# Adding New Datasets

This guide explains how to create custom benchmark datasets for the CRI Benchmark, from defining a persona specification to generating and validating the final dataset.

## Dataset Structure

Every CRI dataset is a directory containing three files:

```
my-dataset/
├── conversations.jsonl    # One Message JSON per line (chronological)
├── ground_truth.json      # Single GroundTruth JSON object
└── metadata.json          # Single DatasetMetadata JSON object
```

### conversations.jsonl

Each line is a JSON-serialized `Message`:

```json
{"message_id": 1, "role": "user", "content": "Hi there! I just moved to Berlin.", "timestamp": "2026-01-01T09:00:00+00:00", "session_id": "session-001", "day": 1}
{"message_id": 2, "role": "assistant", "content": "Welcome to Berlin! How are you settling in?", "timestamp": "2026-01-01T09:00:15+00:00", "session_id": "session-001", "day": 1}
```

### ground_truth.json

Contains all expected outcomes that the scoring engine evaluates against:

```json
{
  "final_profile": {
    "city": {
      "dimension_name": "city",
      "value": "Berlin, Germany",
      "query_topic": "current city",
      "category": "where"
    }
  },
  "changes": [],
  "noise_examples": [],
  "signal_examples": [],
  "conflicts": [],
  "temporal_facts": [],
  "query_relevance_pairs": []
}
```

### metadata.json

Dataset provenance and configuration:

```json
{
  "dataset_id": "my-custom-dataset",
  "persona_id": "custom-persona",
  "message_count": 500,
  "simulated_days": 30,
  "version": "1.0.0",
  "seed": 42
}
```

## Approach 1 — Using the Dataset Generator

The recommended approach is to define a `RichPersonaSpec` and use the `DatasetGenerator` to synthesize the full dataset.

### Step 1 — Define a PersonaSpec

```python
# my_persona.py
from cri.datasets.personas.specs import RichPersonaSpec
from cri.models import (
    BeliefChange,
    ConflictScenario,
    NoiseExample,
    ProfileDimension,
    QueryRelevancePair,
    SignalExample,
    TemporalFact,
)

MY_PERSONA = RichPersonaSpec(
    persona_id="persona-dev-ops-engineer",
    name="Jamie Park",
    description=(
        "A 32-year-old DevOps engineer at a SaaS company in Seattle. "
        "Jamie recently transitioned from backend development, has a "
        "rescue dog named Pixel, enjoys homebrewing and hiking, and "
        "is learning Rust on the side."
    ),
    complexity_level="basic",
    simulated_days=30,
    target_message_count=500,

    # Define the expected final profile
    profile_dimensions={
        "name": ProfileDimension(
            dimension_name="name",
            value="Jamie Park",
            query_topic="name",
            category="who",
        ),
        "age": ProfileDimension(
            dimension_name="age",
            value="32",
            query_topic="age",
            category="who",
        ),
        "occupation": ProfileDimension(
            dimension_name="occupation",
            value="DevOps Engineer at a SaaS company",
            query_topic="occupation",
            category="what",
        ),
        "city": ProfileDimension(
            dimension_name="city",
            value="Seattle, Washington",
            query_topic="current city",
            category="where",
        ),
        "hobbies": ProfileDimension(
            dimension_name="hobbies",
            value=["homebrewing", "hiking", "learning Rust"],
            query_topic="hobbies",
            category="what",
        ),
        "pet": ProfileDimension(
            dimension_name="pet",
            value="a rescue dog named Pixel",
            query_topic="pet",
            category="what",
        ),
    },

    # Define belief changes (things that evolve over time)
    belief_changes=[
        BeliefChange(
            fact="occupation",
            old_value="Backend Developer",
            new_value="DevOps Engineer at a SaaS company",
            query_topic="occupation",
            changed_around_msg=150,
            key_messages=[140, 150, 160],
        ),
    ],

    # Noise: messages with no factual persona content
    noise_examples=[
        NoiseExample(
            text="Hey, how's it going?",
            reason="Generic greeting",
        ),
        NoiseExample(
            text="Can you explain Kubernetes pods?",
            reason="Technical question unrelated to persona",
        ),
        NoiseExample(
            text="Thanks, that helped!",
            reason="Gratitude with no persona data",
        ),
    ],

    # Signal: messages that reveal persona facts
    signal_examples=[
        SignalExample(
            text="I work as a DevOps engineer at a SaaS company here in Seattle.",
            target_fact="occupation: DevOps Engineer, city: Seattle",
        ),
        SignalExample(
            text="Pixel, my rescue dog, loves our weekend hikes in the Cascades.",
            target_fact="pet: rescue dog named Pixel, hobby: hiking",
        ),
        SignalExample(
            text="I've been homebrewing IPAs lately. My latest batch turned out great!",
            target_fact="hobby: homebrewing",
        ),
    ],

    # Conflicts: contradictory statements to test resolution
    conflicts=[
        ConflictScenario(
            conflict_id="conflict-jamie-01",
            topic="role",
            conflicting_statements=[
                "I'm a backend developer. I write Go services all day.",
                "I transitioned to DevOps last month. Now I manage CI/CD pipelines.",
            ],
            correct_resolution="Jamie transitioned from backend to DevOps. DevOps is current.",
            resolution_type="recency",
            introduced_at_messages=[50, 180],
        ),
    ],

    # Temporal facts: facts with time-bounded validity
    temporal_facts=[
        TemporalFact(
            fact_id="tf-jamie-01",
            description="Jamie was a Backend Developer",
            value="Backend Developer",
            valid_from="2020-01-01",
            valid_until="2025-12-01",
            query_topic="career history",
            should_be_current=False,
        ),
        TemporalFact(
            fact_id="tf-jamie-02",
            description="Jamie is a DevOps Engineer",
            value="DevOps Engineer at a SaaS company",
            valid_from="2025-12-01",
            valid_until=None,
            query_topic="current occupation",
            should_be_current=True,
        ),
    ],

    # Query-relevance pairs: test retrieval precision
    query_relevance_pairs=[
        QueryRelevancePair(
            query_id="qrp-jamie-01",
            query="What does Jamie do for work?",
            expected_relevant_facts=["DevOps Engineer", "SaaS company"],
            expected_irrelevant_facts=["homebrewing", "Pixel"],
        ),
        QueryRelevancePair(
            query_id="qrp-jamie-02",
            query="Does Jamie have any pets?",
            expected_relevant_facts=["rescue dog named Pixel"],
            expected_irrelevant_facts=["DevOps", "Seattle"],
        ),
    ],
)
```

### Step 2 — Generate the Dataset

```python
from pathlib import Path

from cri.models import GeneratorConfig
from cri.datasets.generator import DatasetGenerator
from my_persona import MY_PERSONA

# Create generator with a fixed seed for reproducibility
config = GeneratorConfig(seed=42)
generator = DatasetGenerator(config)

# Generate the dataset
dataset = generator.generate(MY_PERSONA)

# Save to disk
generator.save_dataset(
    dataset,
    Path("datasets/custom/devops-engineer"),
)

print(f"Generated {len(dataset.messages)} messages")
print(f"Profile dimensions: {len(dataset.ground_truth.final_profile)}")
print(f"Belief changes: {len(dataset.ground_truth.changes)}")
print(f"Conflicts: {len(dataset.ground_truth.conflicts)}")
```

### Step 3 — Validate the Dataset

```bash
cri validate-dataset datasets/custom/devops-engineer
```

Expected output:

```
Validating dataset: datasets/custom/devops-engineer

✓ Dataset is valid.
  Messages:           500
  Profile dimensions: 6
  Belief changes:     1
  Conflicts:          1
  Temporal facts:     2
```

### Step 4 — Run the Benchmark

```bash
cri run \
  --adapter no-memory \
  --dataset datasets/custom/devops-engineer \
  --verbose
```

## Approach 2 — Manual Dataset Creation

You can also create datasets manually without the generator. This is useful when:
- You have real conversation data to convert
- You want hand-crafted evaluation scenarios
- You need specific edge cases not covered by the generator

### Step 1 — Create conversations.jsonl

```python
import json
from cri.models import Message

messages = [
    Message(
        message_id=1,
        role="user",
        content="Hi! I'm a teacher in Portland.",
        timestamp="2026-01-01T10:00:00+00:00",
        session_id="session-001",
        day=1,
    ),
    Message(
        message_id=2,
        role="assistant",
        content="Hello! That's great. What do you teach?",
        timestamp="2026-01-01T10:00:15+00:00",
        session_id="session-001",
        day=1,
    ),
    Message(
        message_id=3,
        role="user",
        content="I teach high school physics. Love it!",
        timestamp="2026-01-01T10:01:00+00:00",
        session_id="session-001",
        day=1,
    ),
    # ... more messages
]

with open("my-dataset/conversations.jsonl", "w") as f:
    for msg in messages:
        f.write(msg.model_dump_json() + "\n")
```

### Step 2 — Create ground_truth.json

```python
from cri.models import GroundTruth, ProfileDimension

ground_truth = GroundTruth(
    final_profile={
        "occupation": ProfileDimension(
            dimension_name="occupation",
            value="high school physics teacher",
            query_topic="occupation",
            category="what",
        ),
        "city": ProfileDimension(
            dimension_name="city",
            value="Portland",
            query_topic="current city",
            category="where",
        ),
    },
    changes=[],
    noise_examples=[],
    signal_examples=[],
    conflicts=[],
    temporal_facts=[],
    query_relevance_pairs=[],
)

with open("my-dataset/ground_truth.json", "w") as f:
    f.write(ground_truth.model_dump_json(indent=2))
```

### Step 3 — Create metadata.json

```python
from cri.models import DatasetMetadata

metadata = DatasetMetadata(
    dataset_id="manual-teacher-dataset",
    persona_id="teacher-persona",
    message_count=len(messages),
    simulated_days=1,
    version="1.0.0",
    seed=None,  # No seed for manually created datasets
)

with open("my-dataset/metadata.json", "w") as f:
    f.write(metadata.model_dump_json(indent=2))
```

## Dataset Design Guidelines

### Ground Truth Coverage

Each ground truth component feeds a specific CRI dimension:

| Component | CRI Dimension | Purpose |
|-----------|---------------|---------|
| `final_profile` | PAS | Expected profile facts the system should store |
| `changes` | DBU | Belief updates the system should track |
| `noise_examples` | MEI, QRP | Messages that should NOT become facts |
| `signal_examples` | PAS, MEI | Messages that SHOULD become facts |
| `conflicts` | CRQ | Contradictions the system must resolve |
| `temporal_facts` | TC | Time-bounded facts the system must track |
| `query_relevance_pairs` | QRP | Retrieval precision test cases |

### Complexity Levels

| Level | Messages | Profile Dims | Changes | Conflicts | Use Case |
|-------|----------|-------------|---------|-----------|----------|
| Basic | 500–1000 | 8–12 | 2–3 | 2–3 | Core functionality testing |
| Intermediate | 1500–2500 | 12–16 | 4–6 | 4–6 | Real-world simulation |
| Advanced | 2500–4000 | 16–20 | 6–10 | 6–10 | Stress testing and edge cases |

### Best Practices

1. **Balance signal and noise** — Real conversations are ~30% signal, ~70% noise
2. **Space out belief changes** — Changes clustered together are harder to evaluate
3. **Make conflicts unambiguous** — The correct resolution should be clear from context
4. **Include temporal context** — Reference dates and time periods in messages
5. **Use realistic language** — Avoid overly formal or structured messages
6. **Set a fixed seed** — Always use `GeneratorConfig(seed=N)` for reproducibility

## Adding to the Canonical Suite

To contribute a dataset to the canonical benchmark suite:

1. Create a `RichPersonaSpec` in `src/cri/datasets/personas/specs.py`
2. Add it to the `ALL_PERSONAS` list
3. Run `python scripts/generate_canonical_datasets.py` to regenerate all datasets
4. Validate with `cri validate-dataset datasets/canonical/<your-dataset>`
5. Update `datasets/README.md` with a description

```python
# In src/cri/datasets/personas/specs.py

MY_NEW_PERSONA = RichPersonaSpec(
    persona_id="persona-4-specialized",
    name="...",
    # ...
)

# Add to the canonical list
ALL_PERSONAS = [
    PERSONA_BASIC,
    PERSONA_INTERMEDIATE,
    PERSONA_ADVANCED,
    MY_NEW_PERSONA,  # ← Add here
]
```

## Troubleshooting

### "Dataset validation failed"

Common issues:
- `conversations.jsonl` has malformed JSON lines
- `ground_truth.json` references message IDs that don't exist
- Missing required fields in `ProfileDimension` or `BeliefChange`
- `metadata.json` message count doesn't match actual messages

### Generator produces too few messages

Increase `target_message_count` in your `RichPersonaSpec`. The generator aims for approximately this count but may vary ±20%.

### Generator produces unrealistic conversations

The generator uses template-based synthesis. For more realistic conversations, consider:
- Adding more `signal_examples` with varied phrasing
- Adding more `noise_examples` to increase conversation naturalness
- Adjusting `simulated_days` to spread content over a longer timeline

## Next Steps

- [Quick Start Guide](quickstart.md) — Run your first benchmark
- [Integration Guide](integration.md) — Connect your memory system
- [Adding New Metrics](new-metrics.md) — Create custom evaluation dimensions
- [Reproducibility Guide](reproducibility.md) — Ensure consistent results
