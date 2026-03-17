# DBU — Dynamic Belief Updating

> Measures how well the memory system updates its beliefs when new information contradicts or supersedes previous knowledge.

---

## Why This Metric Exists

Users' lives change. They switch jobs, move cities, change preferences, and update their plans. A memory system that cannot update its beliefs is dangerous — it will confidently assert outdated information as current truth.

DBU tests the critical capability of **knowledge evolution**: when a user says "I just moved to Denver" after previously saying they live in Portland, the system must update its stored location to Denver and stop asserting Portland as the current residence.

## What It Measures

DBU evaluates two complementary aspects of belief updating:

1. **Recency** — Does the system reflect the new, updated value?
2. **Staleness** — Does the system still assert the old value as the current truth?

A belief change passes only when both conditions are met: the new value IS present AND the old value is NOT asserted as current.

> Note: Historical mentions of the old value are acceptable (e.g., "previously lived in Portland"). The staleness check only flags the old value if it is presented as the **current** truth.

## How It Works

### Algorithm

For each `BeliefChange` recorded in the ground truth:

1. **Query** the adapter for facts related to the belief change's `query_topic`.
2. **Recency check** — Build a prompt via `dbu_recency_check` to verify the new value is present. Expected verdict: **YES**.
3. **Staleness check** — Build a prompt via `dbu_staleness_check` to verify the old value is not asserted as current. Expected verdict: **NO**.
4. The belief change **passes** only when `recency == YES` AND `staleness == NO`.

### Scoring

```
DBU = passed_belief_changes / total_belief_changes
```

Score range: **0.0 – 1.0**. Defaults to 1.0 when there are no belief changes in the ground truth (vacuously correct).

### Rubric Design

The staleness rubric distinguishes between historical context and current assertion. Storing "used to live in Portland" is fine. Storing "lives in Portland" when the user moved to Denver is a failure.

## Default Weight

| Profile | Weight |
|---------|--------|
| Core | 0.20 |
| Extended | 0.20 |
| Full | 0.20 |

DBU carries heavy weight because failing to update beliefs leads to actively wrong information being served, which is worse than missing information entirely.

## Interpretation

| Score Range | Interpretation |
|-------------|----------------|
| 0.90 – 1.00 | Excellent — beliefs update reliably |
| 0.70 – 0.89 | Good — most updates captured, occasional staleness |
| 0.50 – 0.69 | Moderate — inconsistent updating behavior |
| 0.00 – 0.49 | Poor — system retains outdated beliefs as current |

## Ground Truth Requirements

DBU requires `changes` in the ground truth — a list of `BeliefChange` objects, each with:

- `fact` — the fact that changed (e.g., `"city of residence"`)
- `old_value` — the previous value (e.g., `"Portland"`)
- `new_value` — the updated value (e.g., `"Denver"`)
- `query_topic` — the topic string used to query the adapter

## Related Dimensions

- **PAS** — tests recall of the final state; DBU specifically tests the transition
- **TC** — tests temporal awareness more broadly; DBU focuses on belief replacement
- **CRQ** — tests conflict resolution; DBU tests straightforward supersession
