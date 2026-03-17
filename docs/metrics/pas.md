# PAS — Profile Accuracy Score

> Measures how accurately the memory system recalls specific persona details after ingesting conversation events.

---

## Why This Metric Exists

Profile Accuracy is the most fundamental dimension of the CRI Benchmark. A memory system must, at minimum, be able to accurately recall what it has been told. If a user explicitly states their name, occupation, hometown, or preferences during a conversation, the system should be able to retrieve those facts later.

Without accurate profile recall, no higher-order memory capability (belief updating, temporal tracking, conflict resolution) matters — the foundation is broken.

## What It Measures

PAS evaluates **factual recall accuracy** of stored persona details. It checks whether the memory system correctly captured and can retrieve specific profile attributes such as:

- Demographics (name, age, location)
- Professional information (occupation, employer)
- Preferences (favorite foods, hobbies)
- Explicitly stated facts about the user

## How It Works

### Algorithm

For every profile dimension in the ground truth's `final_profile`:

1. **Query** the adapter for facts related to the dimension's `query_topic`.
2. If the dimension value is a list (multi-value dimension), create one binary check per list element. Otherwise, create a single check.
3. For each check, generate an LLM judge prompt using the `pas_check` rubric and evaluate it with the `BinaryJudge`.
4. A **YES** verdict means the check passed — the fact was found.

### Scoring

```
PAS = passed_checks / total_checks
```

Score range: **0.0 – 1.0** (0.0 when no checks exist).

### Rubric Design

The judge prompt emphasizes **semantic equivalence** — the stored fact does not need to use the exact same words as the ground truth. If the meaning is the same, it counts as a match.

Example: If the expected value is `"software engineer"` and the system stores `"works as a developer in software"`, the judge should return YES.

## Default Weight

| Profile | Weight |
|---------|--------|
| Core | 0.25 |
| Extended | 0.20 |
| Full | 0.20 |

PAS carries the highest weight in the Core profile because it tests the most basic capability — a system that cannot recall explicit facts will fail at everything else.

## Interpretation

| Score Range | Interpretation |
|-------------|----------------|
| 0.90 – 1.00 | Excellent — near-perfect recall of stated facts |
| 0.70 – 0.89 | Good — captures most profile information |
| 0.50 – 0.69 | Moderate — significant gaps in recall |
| 0.00 – 0.49 | Poor — fundamental extraction or retrieval failures |

## Ground Truth Requirements

PAS requires `final_profile` in the ground truth — a dictionary of `ProfileDimension` objects, each with:

- `dimension_name` — what the fact is about (e.g., `"occupation"`)
- `query_topic` — the topic string used to query the adapter
- `value` — the expected value (string or list of strings)

## Related Dimensions

- **DBU** — tests whether updated facts overwrite old ones (PAS only tests the final state)
- **QRP** — tests retrieval precision (PAS tests whether the fact exists at all)
- **MEI** — tests storage efficiency (PAS doesn't care how many other facts are stored)
