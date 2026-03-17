# CRQ — Conflict Resolution Quality

> Measures how well the memory system handles contradictory information — whether it identifies conflicts, resolves them appropriately, and reflects the correct resolution.

---

## Why This Metric Exists

Real conversations contain contradictions. A user might say "I'm vegetarian" in one message and mention eating chicken in another. External information might conflict with self-reported facts. The same topic might be discussed differently across sessions.

A naive memory system will either store both contradictory facts (creating confusion) or arbitrarily discard one. A good memory system detects conflicts and resolves them using appropriate strategies — recency, authority, explicit correction, or contextual disambiguation.

CRQ evaluates this conflict resolution capability.

## What It Measures

CRQ evaluates three aspects of conflict handling:

- **Detection** — Does the system recognize contradictory information?
- **Resolution strategy** — Does it apply appropriate resolution (recency, authority, explicit correction)?
- **Stored state** — Does the stored knowledge reflect the correct resolution?

## How It Works

### Algorithm

For each `ConflictScenario` in the ground truth:

1. **Query** the adapter for facts related to the conflict's `topic`.
2. Build a judge prompt via `crq_resolution_check` that includes the expected correct resolution.
3. Submit to the `BinaryJudge` — a **YES** verdict means the conflict was resolved correctly.

### Scoring

```
CRQ = passed_checks / total_checks
```

Score range: **0.0 – 1.0**. Defaults to 1.0 when there are no conflict scenarios (vacuously correct).

### Rubric Design

The rubric asks the judge whether the stored facts reflect the **correct resolution**, not whether both sides of the conflict are stored. The system may retain historical context ("previously said X, now says Y"), but the current state must reflect the correct answer.

## Default Weight

| Profile | Weight |
|---------|--------|
| Core | 0.10 |
| Extended | 0.10 |
| Full | 0.10 |

CRQ carries a lower weight because conflict resolution is an advanced capability. Many real-world conversations contain few genuine contradictions, and basic systems can function adequately without sophisticated conflict resolution.

## Interpretation

| Score Range | Interpretation |
|-------------|----------------|
| 0.90 – 1.00 | Excellent — reliably resolves contradictions |
| 0.70 – 0.89 | Good — handles most conflicts correctly |
| 0.50 – 0.69 | Moderate — inconsistent conflict handling |
| 0.00 – 0.49 | Poor — fails to resolve contradictions or resolves them incorrectly |

## Ground Truth Requirements

CRQ requires `conflicts` in the ground truth — a list of `ConflictScenario` objects, each with:

- `conflict_id` — unique identifier
- `topic` — the topic area where the conflict occurred
- `resolution_type` — the expected resolution strategy (e.g., `"recency"`, `"explicit_correction"`)
- `correct_resolution` — the expected correct resolution text

## Related Dimensions

- **DBU** — tests straightforward supersession; CRQ tests genuine contradictions
- **TC** — some conflicts arise from temporal ambiguity
- **QRP** — conflict resolution affects retrieval quality
