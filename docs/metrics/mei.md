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

1. Build a list of expected ground-truth facts from `final_profile` (flattening multi-value dimensions).
2. Call `get_all_facts()` to retrieve everything the system stored.
3. For each ground-truth fact, use the `BinaryJudge` with the `mei_coverage_check` rubric to determine if it is covered by any stored fact.
4. Compute the formula.

### Formula

```
efficiency_ratio = covered_gt_facts / total_facts_stored
coverage_factor  = covered_gt_facts / total_gt_facts
MEI = harmonic_mean(efficiency_ratio, coverage_factor)
```

Score range: **0.0 – 1.0**. Returns 0.0 if the adapter stores zero facts.

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
- **SFC** — tests forgetting of ephemeral facts; MEI tests overall storage hygiene
- **QRP** — tests retrieval quality; MEI tests storage quality
