#!/usr/bin/env python3
"""Generate canonical CRI benchmark datasets from persona specifications.

This script creates structurally complete datasets from persona specifications.
Each dataset directory gets:
  - conversations.jsonl  — One Message JSON per line
  - ground_truth.json    — Complete GroundTruth JSON object
  - metadata.json        — DatasetMetadata JSON object

The conversations are deterministically generated from persona specs without
requiring an LLM. They exercise all metric dimensions:
  - PAS (Persona Accuracy Score): via signal messages establishing profile facts
  - DBU (Dynamic Belief Updating): via messages that update beliefs
  - MEI (Memory Efficiency Index): via coverage and efficiency evaluation
  - TC (Temporal Coherence): via temporal fact introductions with timestamps
  - CRQ (Conflict Resolution Quality): via conflicting statements at specific points
  - QRP (Query Response Precision): ground truth pairs for retrieval eval

Usage:
    python scripts/generate_canonical_datasets.py
"""

from __future__ import annotations

import json
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cri.datasets.personas.specs import (  # noqa: E402
    ALL_PERSONAS,
    RichPersonaSpec,
)
from cri.models import (  # noqa: E402
    DatasetMetadata,
    GroundTruth,
    Message,
)


def generate_ground_truth(persona: RichPersonaSpec) -> GroundTruth:
    """Convert a RichPersonaSpec into a GroundTruth object."""
    return GroundTruth(
        final_profile=persona.profile_dimensions,
        changes=persona.belief_changes,
        noise_examples=persona.noise_examples,
        signal_examples=persona.signal_examples,
        conflicts=persona.conflicts,
        temporal_facts=persona.temporal_facts,
        query_relevance_pairs=persona.query_relevance_pairs,
    )


def _assistant_response(signal_text: str, persona_name: str) -> str:
    """Generate a plausible assistant response to a signal message."""
    # Short acknowledging responses that reference key info
    templates = [
        f"Got it! Thanks for sharing that, {persona_name}.",
        "That's really interesting. I'll keep that in mind.",
        "Thanks for letting me know!",
        "Noted! That's helpful context.",
        "I appreciate you sharing that with me.",
        "That's great to know about you.",
        "Understood, thanks for the update!",
        "Interesting! I'll remember that.",
    ]
    return random.choice(templates)


def _assistant_response_noise(noise_text: str) -> str:
    """Generate assistant response to a noise message."""
    if "?" in noise_text:
        templates = [
            "I'd be happy to help with that!",
            "Let me look into that for you.",
            "Good question! Here's what I think...",
            "Sure, I can help with that.",
        ]
    else:
        templates = [
            "Sounds good!",
            "No problem at all!",
            "Of course!",
            "Glad I could help!",
            "You're welcome!",
        ]
    return random.choice(templates)


def _generate_belief_change_messages(
    persona: RichPersonaSpec,
    change_idx: int,
) -> list[tuple[str, str]]:
    """Generate user+assistant message pairs for a belief change."""
    change = persona.belief_changes[change_idx]
    pairs = []

    # Message showing old value
    old_msg = f"You know, I've been {change.old_value} for a while now. It's been working out."
    pairs.append((old_msg, f"That's great to hear, {persona.name}!"))

    # Transition message
    transition_msg = f"Actually, things have changed. I'm no longer {change.old_value}. I've switched to {change.new_value}."
    pairs.append((transition_msg, "Thanks for the update! I'll keep that in mind."))

    # Confirmation message
    confirm_msg = f"Yeah, {change.new_value} is definitely where I'm at now regarding {change.fact}."
    pairs.append((confirm_msg, "Noted! Glad you're happy with the change."))

    return pairs


def _generate_conflict_messages(
    persona: RichPersonaSpec,
    conflict_idx: int,
) -> list[tuple[str, str, int]]:
    """Generate messages for a conflict scenario. Returns (user_msg, asst_msg, target_msg_id)."""
    conflict = persona.conflicts[conflict_idx]
    results = []

    for i, statement in enumerate(conflict.conflicting_statements):
        target_id = conflict.introduced_at_messages[i] if i < len(conflict.introduced_at_messages) else 0
        asst = f"I see, thanks for sharing that about {conflict.topic}."
        results.append((statement, asst, target_id))

    return results


def _generate_temporal_messages(
    persona: RichPersonaSpec,
) -> list[tuple[str, str]]:
    """Generate messages that establish temporal facts."""
    pairs = []
    for tf in persona.temporal_facts:
        if tf.should_be_current:
            msg = f"Currently, {tf.description.lower()} — specifically, {tf.value}."
            pairs.append((msg, "Got it, thanks for the update!"))
        else:
            msg = f"Back when {tf.description.lower()}, it was {tf.value}. That's changed now though."
            pairs.append((msg, "Interesting to hear about that history."))
    return pairs


def generate_conversations(persona: RichPersonaSpec, seed: int = 42) -> list[Message]:
    """Generate a deterministic conversation from a persona spec.

    Strategy:
    1. Start with introductory messages
    2. Weave in signal examples (profile facts)
    3. Insert belief changes at appropriate points
    4. Insert conflict statements at their target positions
    5. Sprinkle noise throughout
    6. Add temporal fact references
    7. Pad to approximate target_message_count
    """
    rng = random.Random(seed)
    base_date = datetime(2026, 1, 1, 8, 0, 0)

    # Pre-collect all content
    signals = list(persona.signal_examples)
    noises = list(persona.noise_examples)
    belief_changes = list(persona.belief_changes)
    conflicts = list(persona.conflicts)
    # Build message slots: list of (priority, target_position, user_text, asst_text)
    # priority: lower = placed first if same position
    slots: list[tuple[int, int, str, str]] = []

    # --- Introduction ---
    intro_msgs = [
        (
            f"Hey there! I'm {persona.name}. {persona.description.split('.')[0]}.",
            f"Nice to meet you, {persona.name}! I'd love to learn more about you.",
        ),
        (
            "Thanks! Happy to chat. I've got a lot going on these days.",
            "Tell me anything you'd like — I'm here to help and listen.",
        ),
    ]
    for i, (u, a) in enumerate(intro_msgs):
        slots.append((0, i * 2, u, a))

    # --- Signal examples (spread across the conversation) ---
    target = persona.target_message_count
    sample_range = range(10, max(target // 2, len(signals) + 20))
    sample_count = min(len(signals), target // 4)
    signal_positions = sorted(rng.sample(sample_range, sample_count))
    for i, sig in enumerate(signals):
        pos = signal_positions[i] if i < len(signal_positions) else 10 + i * 10
        asst = _assistant_response(sig.text, persona.name)
        slots.append((1, pos, sig.text, asst))

    # --- Belief change messages ---
    for i, change in enumerate(belief_changes):
        change_pairs = _generate_belief_change_messages(persona, i)
        base_pos = change.changed_around_msg
        for j, (u, a) in enumerate(change_pairs):
            slots.append((2, base_pos - 20 + j * 10, u, a))

    # --- Conflict messages ---
    for i, _conflict in enumerate(conflicts):
        conflict_msgs = _generate_conflict_messages(persona, i)
        for u, a, target_pos in conflict_msgs:
            slots.append((3, target_pos, u, a))

    # --- Temporal fact messages ---
    temporal_pairs = _generate_temporal_messages(persona)
    temporal_base = target // 3
    for i, (u, a) in enumerate(temporal_pairs):
        slots.append((4, temporal_base + i * 15, u, a))

    # --- Additional signal repetitions to reinforce facts ---
    reinforcement_base = target * 2 // 3
    for i, sig in enumerate(signals[:5]):  # Repeat first 5 signals
        asst_resp = _assistant_response(sig.text, persona.name)
        slots.append((5, reinforcement_base + i * 8, sig.text, asst_resp))

    # --- Noise messages to fill gaps ---
    # Calculate how many noise messages we need to reach target
    current_msg_count = len(slots) * 2  # each slot = user + assistant
    needed_noise = max(0, (target - current_msg_count) // 2)
    noise_cycle = noises * ((needed_noise // len(noises)) + 2) if noises else []
    rng.shuffle(noise_cycle)

    noise_positions = sorted(rng.sample(range(4, target), min(needed_noise, target - 10)))
    for i in range(min(needed_noise, len(noise_cycle), len(noise_positions))):
        n = noise_cycle[i]
        slots.append((10, noise_positions[i], n.text, _assistant_response_noise(n.text)))

    # --- Sort by target position, then priority ---
    slots.sort(key=lambda x: (x[1], x[0]))

    # --- Build messages ---
    messages: list[Message] = []
    msg_id = 1
    current_time = base_date
    day_num = 1
    msgs_today = 0
    session_id = f"session-{day_num:03d}"

    for _, _, user_text, asst_text in slots:
        # Advance time
        gap_minutes = rng.randint(1, 45)
        current_time += timedelta(minutes=gap_minutes)
        msgs_today += 2

        # New day every N messages
        msgs_per_day = rng.randint(8, 20)
        if msgs_today >= msgs_per_day:
            day_num += 1
            msgs_today = 0
            current_time = current_time.replace(hour=8, minute=0, second=0) + timedelta(days=1)
            session_id = f"session-{day_num:03d}"

        # User message
        messages.append(
            Message(
                message_id=msg_id,
                role="user",
                content=user_text,
                timestamp=current_time.isoformat(),
                session_id=session_id,
                day=day_num,
            )
        )
        msg_id += 1

        # Assistant message
        current_time += timedelta(seconds=rng.randint(2, 30))
        messages.append(
            Message(
                message_id=msg_id,
                role="assistant",
                content=asst_text,
                timestamp=current_time.isoformat(),
                session_id=session_id,
                day=day_num,
            )
        )
        msg_id += 1

    return messages


def save_dataset(persona: RichPersonaSpec, output_dir: Path, seed: int = 42) -> dict:
    """Generate and save a complete dataset for a persona."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate conversations
    messages = generate_conversations(persona, seed=seed)

    # Generate ground truth
    ground_truth = generate_ground_truth(persona)

    # Compute actual simulated days
    if messages:
        days = set()
        for m in messages:
            if m.day is not None:
                days.add(m.day)
        simulated_days = len(days) if days else 1
    else:
        simulated_days = 0

    # Create metadata
    metadata = DatasetMetadata(
        dataset_id=persona.persona_id,
        persona_id=persona.persona_id,
        message_count=len(messages),
        simulated_days=simulated_days,
        version="1.0.0",
        seed=seed,
    )

    # Write conversations.jsonl
    conversations_path = output_dir / "conversations.jsonl"
    with open(conversations_path, "w", encoding="utf-8") as f:
        for msg in messages:
            f.write(msg.model_dump_json() + "\n")

    # Write ground_truth.json
    gt_path = output_dir / "ground_truth.json"
    with open(gt_path, "w", encoding="utf-8") as f:
        f.write(ground_truth.model_dump_json(indent=2))

    # Write metadata.json
    meta_path = output_dir / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write(metadata.model_dump_json(indent=2))

    stats = {
        "persona_id": persona.persona_id,
        "persona_name": persona.name,
        "complexity": persona.complexity_level,
        "message_count": len(messages),
        "simulated_days": simulated_days,
        "profile_dimensions": len(ground_truth.final_profile),
        "belief_changes": len(ground_truth.changes),
        "conflicts": len(ground_truth.conflicts),
        "temporal_facts": len(ground_truth.temporal_facts),
        "query_relevance_pairs": len(ground_truth.query_relevance_pairs),
        "noise_examples": len(ground_truth.noise_examples),
        "signal_examples": len(ground_truth.signal_examples),
    }

    print(
        f"  ✓ {persona.persona_id}: {len(messages)} messages, "
        f"{simulated_days} days, "
        f"{len(ground_truth.final_profile)} profile dims, "
        f"{len(ground_truth.changes)} changes, "
        f"{len(ground_truth.conflicts)} conflicts"
    )

    return stats


def generate_readme(datasets_dir: Path, all_stats: list[dict]) -> None:
    """Generate datasets/README.md documenting the canonical datasets."""
    readme = """# CRI Benchmark — Canonical Datasets

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

"""

    for stats in all_stats:
        readme += f"""### {stats["persona_name"]} (`{stats["persona_id"]}`)

- **Complexity**: {stats["complexity"]}
- **Messages**: {stats["message_count"]}
- **Simulated Days**: {stats["simulated_days"]}
- **Profile Dimensions**: {stats["profile_dimensions"]}
- **Belief Changes**: {stats["belief_changes"]}
- **Conflicts**: {stats["conflicts"]}
- **Temporal Facts**: {stats["temporal_facts"]}
- **Query-Relevance Pairs**: {stats["query_relevance_pairs"]}
- **Signal Examples**: {stats["signal_examples"]}
- **Noise Examples**: {stats["noise_examples"]}

"""

    readme += """## CRI Evaluation Dimensions Covered

Each dataset exercises all six dimensions:

| Dimension | Code | Exercised By |
|-----------|------|-------------|
| Persona Accuracy Score | PAS | Signal messages establishing profile facts |
| Dynamic Belief Updating | DBU | Belief change sequences (old → new value) |
| Memory Efficiency Index | MEI | Coverage and efficiency evaluation |
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

All datasets are generated with a fixed random seed (42) for full
reproducibility. The generation script is at `scripts/generate_canonical_datasets.py`.

To regenerate:

```bash
python scripts/generate_canonical_datasets.py
```

## Extending

To add new datasets, see [docs/guides/new-datasets.md](../docs/guides/new-datasets.md).
"""

    readme_path = datasets_dir / "README.md"
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme)
    print(f"\n  ✓ Generated {readme_path}")


def main() -> None:
    datasets_dir = PROJECT_ROOT / "datasets" / "canonical"
    print("CRI Benchmark — Canonical Dataset Generator")
    print("=" * 50)
    print(f"Output directory: {datasets_dir}")
    print()

    all_stats = []
    for persona in ALL_PERSONAS:
        output_dir = datasets_dir / persona.persona_id
        stats = save_dataset(persona, output_dir, seed=42)
        all_stats.append(stats)

    # Generate README
    generate_readme(datasets_dir.parent, all_stats)

    # Validate each dataset
    print("\nValidation:")
    from cri.datasets.loader import load_dataset, validate_dataset

    all_valid = True
    for persona in ALL_PERSONAS:
        ds_dir = datasets_dir / persona.persona_id
        try:
            dataset = load_dataset(ds_dir)
            errors = validate_dataset(dataset)
            if errors:
                print(f"  ✗ {persona.persona_id}: {len(errors)} errors")
                for err in errors[:5]:
                    print(f"    - {err}")
                all_valid = False
            else:
                print(f"  ✓ {persona.persona_id}: VALID ({len(dataset.messages)} messages)")
        except Exception as e:
            print(f"  ✗ {persona.persona_id}: Load error: {e}")
            all_valid = False

    print()
    if all_valid:
        print("All datasets generated and validated successfully! ✓")
    else:
        print("Some datasets have validation issues. See above.")

    # Print summary
    print("\nSummary:")
    print(json.dumps(all_stats, indent=2))


if __name__ == "__main__":
    main()
