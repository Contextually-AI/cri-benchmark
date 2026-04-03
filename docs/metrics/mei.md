# MEI — Memory Efficiency Index

> Measures global storage efficiency by comparing what the system stored against what it should have stored.

---

## Why This Metric Exists

A memory system that stores every user message verbatim can achieve perfect recall — but at the cost of massive redundancy, noise, and degraded retrieval quality. Conversely, a system that aggressively filters may miss important facts.

MEI measures this balance. It evaluates whether the system stores exactly what it needs — no more, no less. A perfect system stores exactly the N ground-truth facts and nothing else (MEI = 1.0). Storage efficiency directly impacts real-world performance — retrieval latency, token costs, and response quality all degrade with bloated storage.

## What It Measures

MEI evaluates two aspects of storage quality through their harmonic mean:

1. **Coverage** — How many ground-truth facts are represented in storage?
2. **Efficiency** — How lean is the storage relative to the useful facts it covers?

The harmonic mean ensures both aspects must be good for a high score. A system that stores everything gets high coverage but low efficiency. A system that stores very little gets high efficiency but low coverage.

## How It Works

### Algorithm

1. Build a list of expected ground-truth facts from `final_profile` (flattening multi-value dimensions). If there are no GT facts, return **0.0** (no data to evaluate).
2. Call `get_events()` to retrieve everything the system stored and record `total_stored`. If the adapter stored zero facts, return **0.0**.
3. Split stored facts into chunks of `MAX_FACTS_PER_PROMPT` (30). For each chunk, call the coverage judge once with the `mei_coverage_chunk_check` prompt — the judge returns a JSON array of the 0-based ground-truth fact indices covered by that chunk. Union these sets across all chunks. An early exit fires once all GT facts are confirmed.
4. Compute the formula.

This **chunk-outer** approach costs at most `⌈total_stored / 30⌉` LLM calls regardless of GT fact count, avoids silent truncation, and correctly handles adapters that store thousands of raw conversation turns.

### Formula

```
efficiency_ratio = covered_gt_facts / total_facts_stored
coverage_factor  = covered_gt_facts / total_gt_facts
MEI = harmonic_mean(efficiency_ratio, coverage_factor)
```

Score range: **0.0 – 1.0**. Returns 0.0 if no GT facts exist or the adapter stores zero facts.

### Example

A system that stores 50 facts covering 10 out of 12 ground-truth facts:
- Coverage = 10/12 = 0.833
- Efficiency = 10/50 = 0.200
- MEI = 2 * 0.833 * 0.200 / (0.833 + 0.200) = **0.322**

The low MEI correctly reflects that despite decent coverage, the system is storing 5x more facts than necessary.

## Default Weight

| Profile | Weight |
|---------|--------|
| Core | 0.20 |
| Extended | 0.15 |
| Full | 0.15 |

MEI carries significant weight because storage efficiency directly impacts real-world system performance — retrieval latency, token costs, and response quality all degrade with bloated storage.

## Interpretation

| Score Range | Interpretation |
|-------------|----------------|
| 0.90 – 1.00 | Excellent — lean storage with near-complete coverage |
| 0.70 – 0.89 | Good — reasonable balance of coverage and efficiency |
| 0.50 – 0.69 | Moderate — either coverage gaps or significant redundancy |
| 0.00 – 0.49 | Poor — severe over-storage, under-coverage, or both |

## Ground Truth Requirements

MEI uses the existing `final_profile` from the ground truth. No additional ground truth annotations are needed — it derives expected facts from the same data PAS uses.

## Design Rationale

**Why harmonic mean instead of arithmetic mean?** The harmonic mean is stricter — it penalizes imbalance. A system with 95% coverage but 5% efficiency would get an arithmetic mean of 50% but a harmonic mean of only 9.5%. This correctly reflects that such extreme imbalance makes the system impractical.

**Why not just count duplicates?** Simple duplicate counting misses semantic redundancy. "Works as a software engineer" and "Is employed as a developer" are not textual duplicates but are semantically redundant. The LLM judge handles this naturally.

## Related Dimensions

- **PAS** — uses the same ground-truth facts but only tests recall, not efficiency
- **QRP** — tests retrieval quality; MEI tests storage quality
