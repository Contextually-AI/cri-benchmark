# QRP — Query Response Precision

> Measures the precision, relevance, and completeness of the memory system's responses to queries about the user.

---

## Why This Metric Exists

A memory system that stores all the right facts but returns them indiscriminately is not useful. When asked about the user's dietary preferences, it should return diet-related facts — not their job history, childhood memories, and pet's name.

QRP evaluates **retrieval quality**: does the system return relevant facts (recall) while filtering out irrelevant ones (precision)? This is the dimension that most directly measures the quality of the `query()` interface.

## What It Measures

QRP evaluates two complementary aspects of retrieval:

1. **Recall** — Are the expected relevant facts present in the response?
2. **Precision** — Are irrelevant facts correctly excluded from the response?

Both matter. A system that returns everything has perfect recall but zero precision. A system that returns nothing has perfect precision but zero recall.

## How It Works

### Algorithm

For each `QueryRelevancePair` in the ground truth:

1. **Query** the adapter with the pair's query to obtain returned facts.
2. **Relevance checks (recall)**: For each expected relevant fact, build a prompt via `qrp_relevance_check` and submit to the judge. **YES** = fact was found (pass).
3. **Irrelevance checks (precision)**: For each expected irrelevant fact, build a prompt via `qrp_irrelevance_check`. **YES** = irrelevant fact was incorrectly included (fail). **NO** = correctly excluded (pass).
4. Compute per-pair metrics:
   - `recall = relevant_found / total_relevant` (1.0 if no relevant facts defined)
   - `precision = irrelevant_excluded / total_irrelevant` (1.0 if no irrelevant facts defined)
5. `pair_score = 0.5 * recall + 0.5 * precision`

### Scoring

```
QRP = mean(pair_scores)
```

Score range: **0.0 – 1.0**. Returns 1.0 when there are no query-relevance pairs (vacuously correct).

### Rubric Design

Two separate rubrics handle recall and precision. The recall rubric checks whether a specific expected fact is semantically present in the returned results. The precision rubric checks whether a specific irrelevant fact was incorrectly included.

## Default Weight

| Profile | Weight |
|---------|--------|
| Core | 0.10 |
| Extended | 0.10 |
| Full | 0.10 |

QRP carries a lower weight because it evaluates retrieval quality rather than knowledge accuracy. A system can score perfectly on PAS/DBU/TC/CRQ while having imperfect retrieval filtering.

## Interpretation

| Score Range | Interpretation |
|-------------|----------------|
| 0.90 – 1.00 | Excellent — precise, relevant retrieval |
| 0.70 – 0.89 | Good — mostly relevant responses with some noise |
| 0.50 – 0.69 | Moderate — returns relevant facts but with significant noise, or misses some relevant facts |
| 0.00 – 0.49 | Poor — retrieval is essentially random or returns everything |

## Ground Truth Requirements

QRP requires `query_relevance_pairs` in the ground truth — a list of `QueryRelevancePair` objects, each with:

- `query_id` — unique identifier
- `query` — the query string posed to the adapter
- `expected_relevant_facts` — list of facts that should be in the response
- `expected_irrelevant_facts` — list of facts that should NOT be in the response

## Related Dimensions

- **PAS** — tests whether facts exist at all; QRP tests whether they are correctly retrieved
- **MEI** — tests storage efficiency; QRP tests retrieval efficiency
- **CRQ** — conflict resolution affects what gets returned for ambiguous queries
