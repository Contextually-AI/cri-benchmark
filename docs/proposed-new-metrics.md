# Proposed New Metrics for CRI Benchmark

> Potential future metrics identified from analysis of memory system architectures and CRI gap analysis.
> Date: 2026-03-11

---

## Table of Contents

- [Executive Summary](#executive-summary)
- [Evaluation Criteria for Each Metric](#evaluation-criteria-for-each-metric)
- [General Ranking](#general-ranking)
- [Proposed Metrics (Ordered by Potential)](#proposed-metrics)
  1. [DAS — Durability-Aware Storage](#1-das--durability-aware-storage)
  2. [IFE — Implicit Fact Extraction](#2-ife--implicit-fact-extraction)
  3. [CSC — Cross-Session Continuity](#3-csc--cross-session-continuity)
  4. [CIQ — Causal Inference Quality](#4-ciq--causal-inference-quality)
  5. [MER — Multi-Entity Reasoning](#5-mer--multi-entity-reasoning)
  6. [SAQ — State Abstraction Quality](#6-saq--state-abstraction-quality)
  7. [SAB — Sensitivity-Aware Behavior](#7-sab--sensitivity-aware-behavior)
  8. [CCS — Confidence Calibration Score](#8-ccs--confidence-calibration-score)
  9. [RBE — Retrieval Budget Efficiency](#9-rbe--retrieval-budget-efficiency)
  10. [KGC — Knowledge Graph Coherence](#10-kgc--knowledge-graph-coherence)
  11. [OCA — Classification Accuracy](#11-oca--classification-accuracy)
  12. [EIF — Export-Import Fidelity](#12-eif--export-import-fidelity)
- [Complete Comparison Matrix](#complete-comparison-matrix)
- [Phased Implementation Recommendations](#phased-implementation-recommendations)

---

## Executive Summary

**12 candidate metrics** are proposed to expand the CRI Benchmark beyond its current 6 implemented dimensions (+ SSI, now implemented). Each metric was identified through analysis of existing memory system architectures and gaps in the current CRI evaluation.

---

## Evaluation Criteria for Each Metric

Each proposed metric is evaluated across 5 criteria on a 1–5 scale:

| Criterion | Scale | Description |
|-----------|-------|-------------|
| **Usefulness** | 1=low, 5=very high | How much value it adds to the benchmark |
| **Implementation difficulty** | 1=easy, 5=very hard | Development effort required |
| **Changes required** | 1=few, 5=many | Modifications to the current model (adapter, ground truth, datasets, rubrics) |
| **Novelty** | 1=low, 5=very high | How original it is relative to existing benchmarks |
| **Scientific rigor** | 1=low, 5=very high | Methodological soundness and literature support |

### Potential Score Calculation

To produce a useful ranking where **higher = more recommended**, the following formula is used:

```
Potential = Usefulness + (6 - Difficulty) + (6 - Changes) + Novelty + Rigor
```

This inverts the difficulty and changes scales (where less is better) and adds the direct scales. Theoretical range: 5–25. Higher score = more attractive metric to implement.

---

## General Ranking

| # | Code | Name | Useful. | Diffic. | Changes | Novel. | Rigor | **Potential** |
|---|------|------|---------|---------|---------|--------|-------|---------------|
| 1 | **DAS** | Durability-Aware Storage | 4 | 2 | 2 | 4 | 5 | **21** |
| 2 | **IFE** | Implicit Fact Extraction | 5 | 2 | 3 | 4 | 4 | **20** |
| 3 | **CSC** | Cross-Session Continuity | 4 | 3 | 3 | 4 | 4 | **18** |
| 4 | **CIQ** | Causal Inference Quality | 4 | 3 | 3 | 5 | 5 | **18** |
| 5 | **MER** | Multi-Entity Reasoning | 5 | 3 | 4 | 5 | 4 | **17** |
| 6 | **SAQ** | State Abstraction Quality | 4 | 3 | 3 | 4 | 4 | **16** |
| 7 | **SAB** | Sensitivity-Aware Behavior | 4 | 3 | 4 | 5 | 4 | **16** |
| 8 | **CCS** | Confidence Calibration Score | 3 | 3 | 4 | 4 | 5 | **15** |
| 9 | **RBE** | Retrieval Budget Efficiency | 3 | 4 | 4 | 3 | 4 | **12** |
| 10 | **KGC** | Knowledge Graph Coherence | 4 | 4 | 5 | 5 | 4 | **12** |
| 11 | **OCA** | Classification Accuracy | 3 | 4 | 5 | 3 | 4 | **11** |
| 12 | **EIF** | Export-Import Fidelity | 3 | 4 | 5 | 4 | 3 | **11** |

---

## Proposed Metrics

---

### 1. DAS — Durability-Aware Storage

**Potential: 21/25**

#### Motivation

Memory systems commonly define durability levels: `permanent` (name, date of birth), `transient` (current city, job), `ephemeral` (mood, current activity). TC measures whether the system understands *when* something is current. But it does not measure whether the system treats facts of different *temporal nature* differently. A person's name is permanent; their current city is transient; their current activity is ephemeral. A good system must reflect these differences.

#### Formal Definition

```
DAS evaluates whether the system correctly treats facts of different durability.

For each fact annotated with durability:
  - permanent facts → must always persist (check: present? YES = pass)
  - transient facts → must persist until superseded (check: current value? YES = pass)
  - ephemeral facts → must disappear after a certain time (check: absent? YES = pass)

DAS = weighted_average(
  permanent_retention,    # weight 0.3
  transient_correctness,  # weight 0.4
  ephemeral_expiry        # weight 0.3
)
```

#### Ground Truth Extension Required

```json
{
  "durability_annotations": [
    {
      "fact_id": "da-01",
      "text": "Name is Marcus Rivera",
      "durability": "permanent",
      "expected_present": true
    },
    {
      "fact_id": "da-02",
      "text": "Works at a fintech startup",
      "durability": "transient",
      "expected_present": true,
      "note": "Current value, not yet superseded"
    },
    {
      "fact_id": "da-03",
      "text": "Having coffee right now",
      "durability": "ephemeral",
      "expected_present": false,
      "mentioned_at_message": 150
    }
  ]
}
```

#### Scores

| Criterion | Score | Justification |
|-----------|-------|---------------|
| Usefulness | 4 | Differentiates sophisticated systems from simple ones; grounded in real memory system design |
| Difficulty | 2 | Simple scoring logic; mainly requires data annotation |
| Changes | 2 | New field in ground truth + annotations in datasets; adapter protocol unchanged |
| Novelty | 4 | Durability-aware storage not measured in any existing benchmark |
| Rigor | 5 | Backed by existing memory system implementations that use durability tiers |

#### Compatibility with Current Model

**Requires minor changes:**
- **Adapter protocol:** No changes — uses existing `get_all_facts()` and `query()`
- **Ground truth schema:** Add `durability_annotations: list[DurabilityAnnotation]` to `GroundTruth` model
- **Datasets:** Add durability annotations to the canonical dataset (can be partially derived from existing `temporal_facts`)
- **Rubrics:** New rubrics for each durability type
- **Overlap with TC:** There is partial overlap with TC (temporal facts). Could be designed as a *generalization* that subsumes aspects of TC.

---

### 2. IFE — Implicit Fact Extraction

**Potential: 20/25**

#### Motivation

PAS measures whether the system remembers *explicitly stated* facts. But in real conversations, many facts are implicit. If someone says "I gave Rocky a bone after our walk," the implicit facts are: (1) has a pet, (2) probably a dog, (3) named Rocky, (4) walks it regularly. A good memory system should capture these derived facts.

#### Formal Definition

```
IFE = correctly_extracted_implicit / total_implicit_facts

where:
  total_implicit_facts = facts that can be logically derived from messages
                         but are NEVER stated explicitly
  correctly_extracted = implicit facts present in get_all_facts() or query()
```

#### Ground Truth Extension Required

```json
{
  "implicit_facts": [
    {
      "fact_id": "if-01",
      "implicit_fact": "Marcus has a pet",
      "derivable_from_messages": [30, 45, 120],
      "source_quotes": [
        "Luna, my rescue cat, is curled up on my desk",
        "Need to buy more cat food"
      ],
      "reasoning": "Multiple references to a cat named Luna implies pet ownership",
      "query_topic": "pets"
    },
    {
      "fact_id": "if-02",
      "implicit_fact": "Marcus works during daytime hours",
      "derivable_from_messages": [15, 80, 200],
      "source_quotes": [
        "exercises in the mornings before work",
        "just got back from the climbing gym after work"
      ],
      "reasoning": "Morning exercise before work + climbing after work implies daytime work schedule",
      "query_topic": "work schedule"
    }
  ]
}
```

#### Evaluation Method

1. For each `implicit_fact`, call `query(query_topic)` on the adapter.
2. Generate a rubric asking whether the stored facts reflect the implicit knowledge.
3. BinaryJudge evaluates presence of the implicit fact in the response.

#### Rubric Template

```
TASK
Determine if the memory system has correctly inferred an implicit fact from conversation context.

IMPLICIT FACT (never stated directly)
"{implicit_fact.implicit_fact}"

This fact can be logically derived from:
{source_quotes}

STORED FACTS (from query about "{implicit_fact.query_topic}")
{numbered list of stored facts}

QUESTION
Do the stored facts above capture or reflect the implicit fact, either directly or through related stored information?

Answer YES or NO.
```

#### Scores

| Criterion | Score | Justification |
|-----------|-------|---------------|
| Usefulness | 5 | Differentiates sophisticated extraction systems from simple keyword matchers; direct application in real chatbots |
| Difficulty | 2 | Simple scoring with BinaryJudge; difficulty lies in designing good implicit facts |
| Changes | 3 | New field in ground truth + careful design of implicit facts for each dataset |
| Novelty | 4 | No existing benchmark measures implicit user fact extraction |
| Rigor | 4 | Well-grounded methodology; requires careful ground truth to avoid ambiguity |

#### Compatibility with Current Model

**Requires moderate changes:**
- **Adapter protocol:** No changes — uses existing `query()`
- **Ground truth schema:** Add `implicit_facts: list[ImplicitFact]` to `GroundTruth` model
- **Datasets:** Design and annotate implicit facts for the canonical dataset. This requires careful analysis of conversations to identify facts that are derivable but never explicitly stated
- **Chat design:** Current datasets may need additional conversations containing richer implicit facts
- **Rubrics:** New rubrics specific to inference

---

### 3. CSC — Cross-Session Continuity

**Potential: 18/25**

#### Motivation

The current model ingests all messages in a single call to `ingest()`. But in reality, users interact across multiple sessions separated by hours, days, or weeks. CSC evaluates whether the system maintains coherence when messages are ingested in separate batches (simulating sessions).

#### Formal Definition

```
CSC compares CRI obtained with monolithic ingestion vs. session-based ingestion.

CRI_monolithic = CRI with ingest(all_messages) (single call)
CRI_sessioned = CRI with ingest(session_1), ingest(session_2), ..., ingest(session_n)

CSC = CRI_sessioned / CRI_monolithic

CSC = 1.0 → No degradation from sessions
CSC < 1.0 → Sessions degrade memory
CSC > 1.0 → Sessions improve memory (possible if the system consolidates between sessions)
```

#### Evaluation Method

1. Partition dataset messages by `session_id` (or by `day` if no session_id).
2. Run the full benchmark with monolithic ingestion → get CRI_monolithic.
3. Run the benchmark with session-based ingestion (N calls to `ingest()`) → get CRI_sessioned.
4. Compute ratio.

#### Scores

| Criterion | Score | Justification |
|-----------|-------|---------------|
| Usefulness | 4 | Directly relevant for real-world use; many systems fail here |
| Difficulty | 3 | Requires running the benchmark 2 times with different ingestion modes |
| Changes | 3 | Runner needs session-based ingestion mode; datasets need clear session boundaries |
| Novelty | 4 | No benchmark measures this explicitly |
| Rigor | 4 | Simple and clear experimental design; reproducible result |

#### Compatibility with Current Model

**Requires moderate changes:**
- **Adapter protocol:** `ingest()` already accepts `list[Message]` and can be called multiple times. However, the current contract does not specify whether the adapter must support multiple calls
- **Ground truth:** No changes
- **Datasets:** Need consistent `session_id` across all messages
- **Runner:** Needs session-based ingestion mode (`cri run --session-mode`)
- **Adapter protocol docs:** Document that adapters should support multiple calls to `ingest()`

---

### 4. CIQ — Causal Inference Quality

**Potential: 18/25**

#### Motivation

Current dimensions evaluate *recall* (does it remember the fact?) and *update* (did it update the fact?). CIQ evaluates *reasoning*: given a set of stored facts, can the system infer causal relationships? Example: "Marcus moved to Denver" + "Marcus changed jobs" → Can the system infer a relationship between both events?

#### Formal Definition

```
CIQ evaluates the system's ability to reason causally about stored facts.

For each causal chain defined in ground truth:
  1. Query the adapter about the causal topic
  2. Judge: Do the stored facts enable inferring the causal relationship?
  3. Judge: Does the system avoid inferring incorrect causal relationships?

CIQ = average(
  correct_inferences / total_causal_chains,
  1 - incorrect_inferences / total_non_causal_pairs
)
```

#### Ground Truth Extension Required

```json
{
  "causal_chains": [
    {
      "chain_id": "cc-01",
      "cause": "Marcus got new job opportunities in Denver",
      "effect": "Marcus moved from San Francisco to Denver",
      "query_topic": "Why did Marcus move?",
      "evidence_messages": [180, 195, 200],
      "expected_in_memory": true
    },
    {
      "chain_id": "cc-02",
      "cause": "Marcus became health-conscious",
      "effect": "Marcus went vegetarian",
      "query_topic": "Why did Marcus change diet?",
      "evidence_messages": [450, 480, 500],
      "expected_in_memory": true
    }
  ],
  "non_causal_pairs": [
    {
      "pair_id": "nc-01",
      "fact_a": "Marcus moved to Denver",
      "fact_b": "Marcus has a cat named Luna",
      "note": "These facts are not causally related"
    }
  ]
}
```

#### Scores

| Criterion | Score | Justification |
|-----------|-------|---------------|
| Usefulness | 4 | Differentiates intelligent systems from simple key-value stores; relevant for advanced assistants |
| Difficulty | 3 | Causality rubrics are more complex; requires ground truth with causal chains |
| Changes | 3 | New field in ground truth; datasets need annotated causal chains |
| Novelty | 5 | No existing user memory benchmark measures causal inference |
| Rigor | 5 | Causal inference evaluation is well-established in ML research |

#### Compatibility with Current Model

**Requires moderate changes:**
- **Adapter protocol:** The current `query()` function returns facts; to evaluate causality, ideally the adapter should be able to answer "why?" questions. If not, it can be evaluated whether stored facts *contain* causal information, even if the adapter doesn't explicitly reason
- **Ground truth:** New field `causal_chains` + `non_causal_pairs`
- **Datasets:** Annotate causal chains in the canonical dataset
- **Fundamental limitation:** The current adapter protocol (`query() → list[StoredFact]`) is not designed for reasoning. CIQ can only evaluate whether stored facts *contain* causal information, not whether the system *reasons* causally. For a complete evaluation, a method like `reason(question: str) → str` would be needed in the adapter.

---

### 5. MER — Multi-Entity Reasoning

**Potential: 17/25**

#### Motivation

The CRI prototype explicitly lists "no cross-entity reasoning" as a limitation. The prototype evaluates a single entity (person) at a time.

Real people constantly mention other people: "My girlfriend Maria works at Google," "My boss Pedro asked me to...," "I'm having dinner with my brother." A good memory system should track not just the primary user's facts but also those of mentioned entities, and the relationships between them.

#### Formal Definition

```
MER evaluates the system's ability to track multiple entities and their relationships.

For each secondary entity defined in ground truth:
  1. Query the adapter about the entity
  2. Judge: Did the system store the entity's key facts?
  3. Judge: Is the relationship with the primary user correct?

For each inter-entity relationship:
  1. Query the adapter about the relationship
  2. Judge: Is the relationship correct?

MER = weighted_average(
  entity_recall,        # weight 0.4 — Does it remember secondary entity facts?
  relationship_accuracy # weight 0.6 — Are relationships correct?
)
```

#### Ground Truth Extension Required

```json
{
  "secondary_entities": [
    {
      "entity_id": "se-01",
      "name": "Luna",
      "type": "pet",
      "relationship_to_primary": "Marcus's rescue cat",
      "known_facts": ["rescue cat", "curls up on desk"],
      "query_topic": "Luna"
    },
    {
      "entity_id": "se-02",
      "name": "Sarah",
      "type": "person",
      "relationship_to_primary": "Marcus's climbing partner",
      "known_facts": ["climbing partner", "met at the gym"],
      "query_topic": "Sarah"
    }
  ],
  "entity_relationships": [
    {
      "rel_id": "rel-01",
      "entity_a": "Marcus",
      "entity_b": "Luna",
      "relationship": "owner-pet",
      "query_topic": "Marcus's relationship with Luna"
    }
  ]
}
```

#### Scores

| Criterion | Score | Justification |
|-----------|-------|---------------|
| Usefulness | 5 | Fundamental for real assistants; all users talk about other people/entities |
| Difficulty | 3 | Relatively simple scoring; complexity lies in dataset design |
| Changes | 4 | Requires significantly rethinking datasets; current chats lack sufficiently rich secondary entities; new ground truth schema |
| Novelty | 5 | No user memory benchmark evaluates multi-entity reasoning |
| Rigor | 4 | Clear experimental design; gap identified across multiple sources |

#### Compatibility with Current Model

**Requires significant dataset changes:**
- **Adapter protocol:** `query()` can search by entity name; no changes needed
- **Ground truth:** Add `secondary_entities` and `entity_relationships`
- **Datasets:** **Largest change:** Current chats mention Luna (the cat) and little else. For robust evaluation, conversations with multiple well-developed secondary entities (friends, family, colleagues, etc.) are needed
- **Chat design:** Recommended to create a new dataset specifically for MER, or enrich persona-1-base with more entities

---

### 6. SAQ — State Abstraction Quality

**Potential: 16/25**

#### Motivation

Current dimensions evaluate individual facts. SAQ evaluates whether the system can generate *summaries* or *abstractions* of all stored information. Example: given 1,000 messages about Marcus, can the system generate a coherent summary profile? This is the ability to consolidate accumulated knowledge into higher-level representations — one of the capabilities where systems fail the most.

#### Formal Definition

```
SAQ evaluates the quality of abstractions/summaries generated by the system.

1. Query the adapter with a broad topic (e.g., "everything about Marcus")
2. Compare returned facts against an "ideal profile" from ground truth
3. Judge: Do the facts form a coherent and complete summary?

SAQ = average(
  completeness,   # Does it cover all key aspects?
  conciseness,    # Does it avoid redundancy and noise?
  accuracy        # Are the facts correct?
)
```

#### Scores

| Criterion | Score | Justification |
|-----------|-------|---------------|
| Usefulness | 4 | Useful for evaluating advanced systems with consolidation; less relevant for simple RAG |
| Difficulty | 3 | Multi-dimensional rubrics are complex; but decomposable into binary checks |
| Changes | 3 | New ground truth (ideal profile), possibly new adapter protocol method (`summarize()`?) |
| Novelty | 4 | State abstraction not measured in user memory benchmarks |
| Rigor | 4 | Well-grounded methodology |

#### Compatibility with Current Model

**Requires moderate changes:**
- **Adapter protocol:** Ideally add `summarize(topic: str) -> str` to the protocol, but can be evaluated with existing `query()` using broad topics
- **Ground truth:** Add `ideal_summaries: dict[str, str]` — expected summaries per topic
- **Datasets:** Create ideal summaries for each persona
- **Rubrics:** Summary evaluation rubrics (completeness, conciseness, accuracy)
- **Alternative without adapter changes:** Use `get_all_facts()` and evaluate whether the complete fact set forms a coherent "summary" — similar to MEI but evaluating coherence instead of efficiency

---

### 7. SAB — Sensitivity-Aware Behavior

**Potential: 16/25**

#### Motivation

Many memory systems define sensitivity levels (e.g., public, basic, sensitive, restricted). No current dimension evaluates whether the system treats sensitive information differently. In a real context, "My social security number is 123-45-6789" should be treated with more care than "I like sushi." SAB evaluates whether the system recognizes and respects different sensitivity levels.

#### Formal Definition

```
SAB evaluates system behavior with information of different sensitivity.

For each fact with sensitivity annotation:
  - public facts → should be returned freely in queries
  - sensitive facts → should be stored but with care
  - restricted facts → ideally should not be stored in plain text

Tests:
1. Does the system store public facts correctly? (baseline)
2. Does the system identify/mark sensitive facts? (metadata check)
3. Does the system reject or redact highly restricted facts? (privacy check)

SAB = weighted_average(
  public_handling,        # weight 0.2
  sensitive_handling,     # weight 0.3
  restricted_handling     # weight 0.5
)
```

#### Scores

| Criterion | Score | Justification |
|-----------|-------|---------------|
| Usefulness | 4 | Critical for compliance (GDPR, HIPAA) in real deployment; less relevant for purely technical evaluation |
| Difficulty | 3 | Requires defining sensitivity taxonomy and annotating data |
| Changes | 4 | New ground truth, new dataset data, possibly new fields in StoredFact metadata; current chats lack rich sensitive information |
| Novelty | 5 | No memory benchmark evaluates privacy-awareness |
| Rigor | 4 | Backed by real-world compliance requirements and existing memory system implementations with sensitivity tiers |

#### Compatibility with Current Model

**Requires significant changes:**
- **Adapter protocol:** For full evaluation, the adapter would need a method like `query(topic, sensitivity_level?)` or sensitivity metadata in `StoredFact`. However, partial evaluation is possible by checking if restricted facts are *rejected* (don't appear in `get_all_facts()`)
- **Ground truth:** Add `sensitivity_annotations: list[SensitivityAnnotation]` with each fact categorized by level
- **Datasets:** **Significant change:** Current chats do not include sensitive information (phone numbers, addresses, medical information, etc.). Conversations with facts at different sensitivity levels are needed
- **Rubrics:** Specialized rubrics for each sensitivity tier

---

### 8. CCS — Confidence Calibration Score

**Potential: 15/25**

#### Motivation

Many memory systems assign confidence scores to stored facts. But are they well-calibrated? A system that assigns 0.95 confidence to everything (including incorrect facts) is worse than one that assigns 0.60 to ambiguous facts and 0.95 to clear ones. CCS measures calibration.

#### Formal Definition

```
CCS evaluates whether the adapter's confidence scores correlate with actual correctness.

For each stored fact with confidence > 0:
  1. Verify if the fact is correct (vs. ground truth) → binary correctness
  2. Group by confidence buckets (0.0-0.2, 0.2-0.4, ..., 0.8-1.0)
  3. For each bucket: actual_accuracy = correct_in_bucket / total_in_bucket

CCS = 1 - ECE (Expected Calibration Error)
ECE = Σ (bucket_size/total) × |bucket_accuracy - bucket_confidence|
```

#### Scores

| Criterion | Score | Justification |
|-----------|-------|---------------|
| Usefulness | 3 | Useful for systems that generate confidence scores; irrelevant for systems without confidence |
| Difficulty | 3 | ECE formula is well-known; requires fact matching against ground truth |
| Changes | 4 | Requires adapters to expose confidence in metadata; many adapters don't; needs ground truth of per-fact correctness |
| Novelty | 4 | Calibration is standard in ML but not applied to memory systems |
| Rigor | 5 | ECE is a well-established metric in ML with decades of research |

#### Compatibility with Current Model

**Requires that the adapter provides confidence scores:**
- **Adapter protocol:** `StoredFact.metadata` can already contain `confidence`, but it is not mandatory. Would need to be documented as an expected field (not mandatory — if no confidence is provided, the dimension is skipped with weight re-normalization)
- **Ground truth:** Needs a list of expected "correct facts" and "incorrect facts"
- **Datasets:** No changes needed
- **Limitation:** Only evaluable for adapters that provide confidence scores. Recommended as an optional dimension.

---

### 9. RBE — Retrieval Budget Efficiency

**Potential: 12/25**

#### Motivation

In real use, LLMs have limited context windows. You cannot send 500 facts to the prompt. RBE evaluates how good the adapter's *selection* is when asked to return facts under a token budget.

#### Formal Definition

```
RBE evaluates retrieval quality under budget constraints.

For each QRP ground truth query, but with budget constraint:
  1. query(topic, max_tokens=N) → facts limited by budget
  2. Evaluate relevance of returned facts vs. expected_relevant_facts

RBE is evaluated at multiple budget levels:
  budgets = [100, 500, 1000, 5000] tokens

RBE = AUC of the quality@budget curve
```

#### Scores

| Criterion | Score | Justification |
|-----------|-------|---------------|
| Usefulness | 3 | Relevant for real-world use with LLMs; less relevant for abstract memory evaluation |
| Difficulty | 4 | Requires adapter protocol extension to support budgets |
| Changes | 4 | New parameter in `query()`, changes to adapter protocol, rubrics, and datasets |
| Novelty | 3 | Budget-constrained retrieval exists in IR; new for memory benchmarks |
| Rigor | 4 | AUC is a standard metric |

#### Compatibility with Current Model

**Requires adapter protocol extension:**
- **Adapter protocol:** `query(topic: str)` does not accept a budget parameter. Would need to change to `query(topic: str, max_tokens: int | None = None)` or add a new method `query_with_budget(topic: str, budget: int) -> list[StoredFact]`
- **Ground truth:** Can reuse existing `query_relevance_pairs`
- **Datasets:** No changes
- **Existing adapters:** All adapters would need to implement budget logic (or ignore the parameter)
- **Alternative without changes:** Results from `query()` could be truncated to the first N tokens and the quality of that subset evaluated (assuming the adapter returns facts in relevance order). This is an imperfect proxy but requires no protocol changes.

---

### 10. KGC — Knowledge Graph Coherence

**Potential: 12/25**

#### Motivation

Stored facts are not independent — they form a knowledge graph. "Marcus works at a fintech" + "Marcus lives in Denver" + "Marcus has a cat Luna" form a subgraph. KGC evaluates whether the implicit relationships in this graph are coherent (no structural contradictions, relationships are valid, etc.).

#### Formal Definition

```
KGC evaluates the coherence of the implicit knowledge graph.

1. Extract all facts via get_all_facts()
2. For each pair of facts that share an entity:
   a. Judge: Are they mutually coherent?
   b. Judge: Is the implicit relationship valid?
3. For each transitive triplet (A→B, B→C → A→C):
   a. Judge: Is transitivity maintained?

KGC = coherent_pairs / total_evaluated_pairs
```

#### Scores

| Criterion | Score | Justification |
|-----------|-------|---------------|
| Usefulness | 4 | Important for graph-based systems; less relevant for flat fact stores |
| Difficulty | 4 | Requires pair/triplet analysis of facts; O(n²) complexity |
| Changes | 5 | Needs graph representation in ground truth; adapter protocol ideally needs graph export; datasets need rich entities |
| Novelty | 5 | No benchmark evaluates knowledge graph coherence in memory systems |
| Rigor | 4 | Graph consistency is an established field with well-defined methods |

#### Compatibility with Current Model

**Requires significant changes:**
- **Adapter protocol:** Can be partially evaluated with `get_all_facts()`, but ideally needs `export_graph() -> dict` or similar
- **Ground truth:** Needs expected graph representation (`expected_graph: list[Triple]` with subject-predicate-object)
- **Datasets:** Current chats lack sufficient inter-entity relationships for meaningful evaluation
- **Cost:** O(n²) evaluation can be expensive in judge API calls
- **Simplified alternative:** Evaluate only coherence of facts about the same entity (instead of the full graph), which is more feasible with current infrastructure

---

### 11. OCA — Classification Accuracy

**Potential: 11/25**

#### Motivation

If a system classifies "Marcus lives in Denver" as a "hobby" fact instead of "location," its internal organization is deficient, which will degrade future retrieval and updates. OCA evaluates whether the system's internal classification of facts into categories (WHO, WHAT, WHERE, WHEN, etc.) is correct.

#### Formal Definition

```
OCA evaluates the precision of classification of stored facts.

For each stored fact with label metadata:
  1. Compare assigned label vs. expected label from ground truth
  2. OCA = correct_labels / total_labeled_facts

If the adapter does not expose labels: dimension not evaluable (skip + re-normalize weights)
```

#### Scores

| Criterion | Score | Justification |
|-----------|-------|---------------|
| Usefulness | 3 | Only relevant for systems with explicit categorization; not all adapters classify |
| Difficulty | 4 | Requires fact matching and label verification; dependent on adapter metadata |
| Changes | 5 | Requires adapters to expose labels in metadata; ground truth needs expected labels per fact |
| Novelty | 3 | Classification accuracy is standard; application to memory categorization is new |
| Rigor | 4 | Classification accuracy is a well-established metric |

#### Compatibility with Current Model

**Requires significant changes and is classification-specific:**
- **Adapter protocol:** `StoredFact.metadata` can contain labels, but it's not standard. Only works for adapters with internal classification
- **Ground truth:** Needs `expected_labels: dict[fact_id, str]` for each fact
- **Recommendation:** Implement as an optional extension for adapters with classification, not as a core CRI dimension

---

### 12. EIF — Export-Import Fidelity

**Potential: 11/25**

#### Motivation

What happens to memory when a user switches platforms? EIF evaluates whether the system can export its state, import it into a new instance, and maintain the same response quality.

#### Formal Definition

```
EIF evaluates the fidelity of the export→import cycle.

1. Run complete benchmark → get CRI_original
2. Export adapter state → export_data
3. Create new adapter instance
4. Import export_data
5. Run evaluation (without re-ingestion) → get CRI_imported

EIF = CRI_imported / CRI_original

EIF = 1.0 → Perfect fidelity
EIF < 1.0 → Information loss in the cycle
```

#### Scores

| Criterion | Score | Justification |
|-----------|-------|---------------|
| Usefulness | 3 | Relevant for portability; not all users migrate systems |
| Difficulty | 4 | Requires 2 benchmark executions + export/import logic |
| Changes | 5 | Requires new methods in adapter protocol (export, import); not all adapters support this |
| Novelty | 4 | Formally evaluated portability is new in benchmarks |
| Rigor | 3 | Simple concept but hard to standardize (what export format?) |

#### Compatibility with Current Model

**Requires significant adapter protocol extension:**
- **Adapter protocol:** Needs `export_state() -> dict` and `import_state(data: dict) -> None` as new methods. This fundamentally changes the protocol from 3 to 5 methods
- **Ground truth:** No changes — reuses the same evaluation pipeline
- **Datasets:** No changes
- **Existing adapters:** All adapters would need to implement export/import
- **Recommendation:** Implement as an optional adapter protocol extension for systems that support portability. Could be a separate "extended benchmark" from the core CRI

---

## Complete Comparison Matrix

| # | Code | Usefulness | Difficulty | Changes | Novelty | Rigor | **Potential** | Adapter Changes | Ground Truth Changes | Dataset Changes |
|---|------|-----------|-----------|---------|---------|-------|---------------|-----------------|---------------------|-----------------|
| 1 | DAS | 4 | 2 | 2 | 4 | 5 | **21** | None | Minor | Minor |
| 2 | IFE | 5 | 2 | 3 | 4 | 4 | **20** | None | Moderate | Moderate |
| 3 | CSC | 4 | 3 | 3 | 4 | 4 | **18** | Minor | None | Minor |
| 4 | CIQ | 4 | 3 | 3 | 5 | 5 | **18** | Minor* | Moderate | Moderate |
| 5 | MER | 5 | 3 | 4 | 5 | 4 | **17** | None | Significant | Significant |
| 6 | SAQ | 4 | 3 | 3 | 4 | 4 | **16** | Optional | Moderate | Moderate |
| 7 | SAB | 4 | 3 | 4 | 5 | 4 | **16** | Optional | Moderate | Significant |
| 8 | CCS | 3 | 3 | 4 | 4 | 5 | **15** | Optional | Minor | None |
| 9 | RBE | 3 | 4 | 4 | 3 | 4 | **12** | Required | None | None |
| 10 | KGC | 4 | 4 | 5 | 5 | 4 | **12** | Optional | Significant | Significant |
| 11 | OCA | 3 | 4 | 5 | 3 | 4 | **11** | Required | Significant | Moderate |
| 12 | EIF | 3 | 4 | 5 | 4 | 3 | **11** | Required | None | None |

*CIQ: the current adapter protocol is sufficient for partial evaluation; complete evaluation would require a `reason()` method.

---
