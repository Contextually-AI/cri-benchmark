# TC — Temporal Coherence

> Measures how well the memory system handles the temporal dimension of knowledge — understanding what is current versus outdated, and recognizing the evolution of information over time.

---

## Why This Metric Exists

Knowledge has a time dimension. A user's current employer is different from their employer two years ago. A dietary restriction may be temporary. A goal may have a deadline. A memory system that treats all stored facts as equally current, with no sense of time, will inevitably produce confused or contradictory responses.

TC evaluates whether the system understands temporal validity — which facts are still current and which have expired or been superseded.

## What It Measures

TC evaluates whether the memory system correctly tracks the **temporal validity** of facts:

- Facts that **should be current** — are they present and treated as active?
- Facts that **should have expired** — does the system still assert them as currently valid?

This goes beyond simple belief updating (DBU). TC tests whether the system has a concept of time-bounded facts, not just fact replacement.

## How It Works

### Algorithm

For each `TemporalFact` in the ground truth's `temporal_facts`:

1. **Query** the adapter for facts related to the temporal fact's `query_topic`.
2. Build a judge prompt via `tc_temporal_validity_check`.
3. Collect a binary verdict from the `BinaryJudge`.
4. Map the verdict to pass/fail based on `should_be_current`:
   - If `should_be_current = True`: **YES** verdict = pass (fact is present and current)
   - If `should_be_current = False`: **NO** verdict = pass (system correctly does not assert the expired fact)

### Scoring

```
TC = passed_checks / total_checks
```

Score range: **0.0 – 1.0**. Defaults to 1.0 when there are no temporal facts (vacuously correct).

### Rubric Design

The rubric asks the judge to evaluate whether a fact is treated as **currently valid**, not merely whether it exists in storage. A system that stores "was temporarily vegetarian in January 2025" is fine — it only fails if it asserts the user IS currently vegetarian when that fact has expired.

## Default Weight

| Profile | Weight |
|---------|--------|
| Core | 0.15 |
| Extended | 0.10 |
| Full | 0.10 |

TC carries moderate weight. Temporal awareness is valuable but less fundamental than basic recall (PAS) or belief updating (DBU).

## Interpretation

| Score Range | Interpretation |
|-------------|----------------|
| 0.90 – 1.00 | Excellent — strong temporal awareness |
| 0.70 – 0.89 | Good — mostly correct temporal handling |
| 0.50 – 0.69 | Moderate — some confusion between current and expired facts |
| 0.00 – 0.49 | Poor — no meaningful temporal tracking |

## Ground Truth Requirements

TC requires `temporal_facts` in the ground truth — a list of `TemporalFact` objects, each with:

- `fact_id` — unique identifier
- `description` — human-readable description of the temporal fact
- `query_topic` — the topic string used to query the adapter
- `should_be_current` — whether this fact should be treated as currently valid at evaluation time

## Related Dimensions

- **DBU** — tests fact replacement; TC tests time-bounded validity more broadly
- **SFC** — tests forgetting of ephemeral facts; TC tests temporal awareness without requiring deletion
- **CRQ** — tests conflict resolution; some conflicts arise from temporal ambiguity
