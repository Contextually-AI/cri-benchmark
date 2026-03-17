# LNC — Long-Horizon Narrative Coherence

> Measures whether the memory system maintains a coherent narrative across causally connected events — not just isolated facts.

---

## Why This Metric Exists

Real user histories are not collections of independent data points — they are narratives. A user lived in San Francisco, got a new job, moved to Denver, and went vegetarian. These events are causally linked, forming a story.

A good memory system should understand that story — not just the individual facts. LNC evaluates this narrative coherence capability, inspired by ontology-as-memory research that describes agent lifecycle models where memory must maintain coherence throughout the agent's entire life.

## What It Measures

LNC evaluates three aspects of narrative coherence for each arc:

- **Sequence** — Does the system preserve the correct chronological order of events?
- **Causality** — Are causal relationships between events captured (e.g., "new job led to relocation")?
- **Consistency** — Is the narrative free of internal contradictions?

## How It Works

### Algorithm

For each `NarrativeArc` in the ground truth:

1. **Query** the adapter with the arc's `query_topic` to retrieve stored facts.
2. **Sequence check** — Build a prompt via `lnc_sequence_check` to verify events appear in correct chronological order. Expected verdict: **YES** (pass).
3. **Causality check** — Build a prompt via `lnc_causality_check` to verify causal relationships are preserved. Expected verdict: **YES** (pass).
4. **Contradiction check** — Build a prompt via `lnc_contradiction_check` to check for internal contradictions. Expected verdict: **NO** (pass — no contradictions found).
5. `arc_score = (sequence_pass + causality_pass + contradiction_pass) / 3`

### Scoring

```
LNC = mean(arc_scores)
arc_score = (sequence_correct + causality_preserved + no_contradictions) / 3
```

Score range: **0.0 – 1.0**. Defaults to 1.0 when there are no narrative arcs (vacuously correct).

### Rubric Design

Each check is a binary judge evaluation:

- The **sequence** rubric asks whether the stored facts reflect events in the correct chronological order. The system does not need exact wording — the progression must be preserved.
- The **causality** rubric asks whether causal connections are inferable from stored facts, even if not stated explicitly.
- The **contradiction** rubric asks whether the narrative contains genuinely incompatible assertions. Temporal progression (e.g., "lived in SF" then "moved to Denver") is not a contradiction.

## Default Weight

| Profile | Weight |
|---------|--------|
| Core | — (not included) |
| Extended | 0.05 |
| Full | 0.05 |

LNC is not included in the Core profile because it tests an advanced narrative capability beyond basic fact recall. It carries a modest weight in Extended and Full profiles, reflecting its importance as a differentiator for sophisticated memory systems.

## Interpretation

| Score Range | Interpretation |
|-------------|----------------|
| 0.90 – 1.00 | Excellent — coherent narratives with preserved causality |
| 0.70 – 0.89 | Good — mostly coherent with minor gaps in sequence or causality |
| 0.50 – 0.69 | Moderate — partial narrative coherence, some contradictions or missing causal links |
| 0.00 – 0.49 | Poor — fragmented narrative, significant contradictions or wrong ordering |

## Ground Truth Requirements

LNC requires `narrative_arcs` in the ground truth — a list of `NarrativeArc` objects, each with:

- `arc_id` — unique identifier
- `topic` — human-readable description of the narrative arc
- `events_in_order` — chronologically ordered list of events
- `causal_links` — causal relationships between events (e.g., `"new job → relocation"`)
- `query_topic` — topic string used to query the adapter
- `key_messages` — (optional) message IDs most relevant to this arc

## Related Dimensions

- **PAS** — tests recall of individual facts; LNC tests whether facts form a coherent story
- **DBU** — tests belief updates; LNC tests whether the sequence of updates makes narrative sense
- **TC** — tests temporal validity; LNC tests chronological ordering across causally linked events
- **CRQ** — tests conflict resolution; LNC tests narrative consistency (some contradictions may arise from narrative incoherence)
