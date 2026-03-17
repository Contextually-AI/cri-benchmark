# SSI — Scale Sensitivity Index

> Measures how composite CRI degrades (or remains stable) as the volume of ingested messages increases.

---

## Why This Metric Exists

A memory system that scores CRI 0.85 with 100 messages but drops to 0.40 with 1,000 has a fundamental scalability problem. Existing dimensions evaluate quality at a single data volume — none of them answer the question: **does this system scale?**

SSI fills this gap by running the full benchmark at multiple data-volume cutoff points and measuring how the composite CRI changes. It is a **meta-metric**: it does not evaluate memory content directly, but rather evaluates how all other dimensions behave under increasing load.

## What It Measures

SSI measures the **degradation rate** of composite CRI as the number of ingested messages increases from 25% to 100% of the dataset. A system that maintains (or improves) its CRI as data grows receives a high SSI score; a system whose quality deteriorates receives a low one.

## How It Works

### Algorithm

1. Segment the dataset into 4 cutoff points: **25%, 50%, 75%, 100%** of messages.
2. For each cutoff point, create a **fresh adapter instance** and ingest only the messages up to that point.
3. Run the full scoring engine (all enabled dimensions) and record the composite CRI score.
4. Compute the degradation rate between the smallest and largest scale points.
5. Derive the SSI score.

A fresh adapter is created at each scale point to prevent state leakage between runs.

### Formula

```
scales = [0.25, 0.50, 0.75, 1.00]

CRI_s = composite CRI using the first s × len(messages) messages

degradation_rate = (CRI_25% − CRI_100%) / CRI_25%

SSI = 1 − max(0, degradation_rate)
```

Score range: **0.0 – 1.0**. Returns 1.0 when there are no messages (vacuously correct).

**Note:** If CRI improves with more data (negative degradation), SSI is capped at 1.0 — it only penalizes degradation, not improvement.

## Default Weight

SSI is **not included** in the CRI composite score. It is reported separately as a meta-metric alongside performance profiles (latency, memory growth).

| Profile | Included | Runs SSI |
|---------|----------|----------|
| Core | No | No |
| Extended | No | No |
| Full | Reported separately | Yes |

SSI can also be enabled on any profile via the `--scale-test` CLI flag.

## Interpretation

| Score Range | Interpretation |
|-------------|----------------|
| 0.95 – 1.00 | Excellent — near-zero degradation at scale |
| 0.80 – 0.94 | Good — minor quality loss with increased volume |
| 0.60 – 0.79 | Moderate — noticeable degradation, scalability concerns |
| 0.00 – 0.59 | Poor — significant quality loss at scale |

## Cost

SSI requires **4 full benchmark runs** (one per scale point), which means approximately 4× the evaluation cost (LLM judge calls, compute time). This is why it is only enabled in the `full` profile or via explicit `--scale-test` flag.

## Ground Truth Requirements

No additional ground truth is required. SSI reuses the same ground truth and scoring engine as the core dimensions, evaluating only facts that were mentioned within each message subset.

## Related Dimensions

- **All dimensions** — SSI measures how every dimension's score changes under scale pressure
- **MEI** — storage efficiency is often the first dimension to degrade as data volume grows
- **QRP** — retrieval precision tends to degrade with larger fact stores
