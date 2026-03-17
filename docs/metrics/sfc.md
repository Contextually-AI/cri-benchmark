# SFC — Selective Forgetting Capability

> Evaluates whether a memory system can appropriately forget ephemeral, superseded, or session-contextual information while retaining facts that should persist.

---

## Why This Metric Exists

A good memory system does not only remember — it also knows what to let go.

Real-world memory systems must handle facts with different lifespans: permanent facts (name, date of birth), transient facts (current employer, city of residence), and ephemeral facts (current mood, what someone is doing right now). A sophisticated system should treat these differently.

SFC evaluates whether a system can **appropriately forget** ephemeral and outdated information. A system that stores thousands of irrelevant facts ("I'm having coffee right now", "working on a presentation today") alongside meaningful knowledge will degrade in retrieval quality and coherence over time.

## What It Measures

SFC evaluates two complementary aspects of memory hygiene:

1. **Should-forget** (weight: 0.6) — Facts that were mentioned but should have been discarded by the end of the conversation. Did the system correctly forget them?
2. **Should-remember** (weight: 0.4) — Facts from the final profile that must persist. Did the system retain them?

The higher weight on should-forget reflects that selective forgetting is the novel capability being tested — retention is already covered by PAS.

### Categories of Forgettable Facts

- **Ephemeral states** — "I'm having coffee right now", "feeling tired today"
- **Fully superseded** — Old facts with no historical value (unlike DBU which verifies the new value is present, SFC verifies the old one is not)
- **Session-contextual** — "Today I need help with X", relevant only to a specific session
- **Redundancies** — The same fact stated three different ways should be consolidated

## How It Works

### Algorithm

1. Retrieve all stored facts via `get_all_facts()`.
2. **Phase 1 — Should-forget checks**: For each `ForgettableFact` in the ground truth, use the `BinaryJudge` with `sfc_forgetting_check` to verify the fact is no longer stored.
   - **NO** verdict = correctly absent (pass)
   - **YES** verdict = still present (fail)
3. **Phase 2 — Should-remember checks**: For each profile dimension in `final_profile`, use `sfc_retention_check` to verify the fact is still present.
   - **YES** verdict = correctly present (pass)
   - **NO** verdict = missing (fail)

### Formula

```
should_forget_score  = correctly_absent / total_should_forget
should_remember_score = correctly_present / total_should_remember
SFC = 0.6 * should_forget_score + 0.4 * should_remember_score
```

Score range: **0.0 – 1.0**. Returns 1.0 when there are no forgettable facts and no profile items (vacuously correct).

## Default Weight

| Profile | Weight |
|---------|--------|
| Core | — (not in core) |
| Extended | 0.05 |
| Full | 0.05 |

SFC is not included in the Core profile because it requires additional ground truth annotations (`forgettable_facts`) that may not be present in all datasets.

## Interpretation

| Score Range | Interpretation |
|-------------|----------------|
| 0.90 – 1.00 | Excellent — appropriate forgetting with strong retention |
| 0.70 – 0.89 | Good — mostly correct hygiene with some lingering ephemeral facts |
| 0.50 – 0.69 | Moderate — retains too many ephemeral facts or loses some persistent ones |
| 0.00 – 0.49 | Poor — no meaningful selective forgetting |

## Ground Truth Requirements

SFC requires `forgettable_facts` in the ground truth — a list of `ForgettableFact` objects, each with:

- `fact_id` — unique identifier
- `text` — the fact that should have been forgotten
- `reason` — why it should be forgotten (e.g., `"ephemeral_state"`, `"session_context"`, `"fully_superseded"`)

SFC also uses `final_profile` from the ground truth for the should-remember checks (shared with PAS/MEI).

## Design Rationale

**Why weighted average instead of harmonic mean?** Unlike MEI where both aspects are equally critical, SFC deliberately emphasizes the novel capability (forgetting) over the already-tested capability (retention). The 60/40 split reflects this priority.

**Why is this separate from MEI?** MEI measures global storage efficiency (ratio of useful facts to total facts). SFC specifically tests whether the system handles fact lifespans correctly. A system could have good MEI by storing few facts overall while still retaining ephemeral noise. SFC catches that.

## Related Dimensions

- **MEI** — tests overall storage efficiency; SFC tests lifecycle-aware storage
- **TC** — tests temporal awareness; SFC tests whether temporal understanding leads to appropriate forgetting
- **DBU** — tests belief replacement; SFC tests whether the old value is actively removed
