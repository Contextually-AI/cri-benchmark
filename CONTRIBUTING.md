# Contributing to CRI Benchmark

Thank you for your interest in contributing to the **CRI Benchmark — Contextual Resonance Index**! This project aims to become an open-source standard for evaluating AI long-term memory systems, and community contributions are essential to that goal.

Whether you're fixing a bug, adding a new evaluation dimension, building an adapter for your memory system, or improving documentation — this guide will help you get started.

---

## Table of Contents

- [Development Setup](#development-setup)
- [Code Style](#code-style)
- [How to Add a New Metric Dimension](#how-to-add-a-new-metric-dimension)
- [How to Add a New Adapter](#how-to-add-a-new-adapter)
- [How to Create New Datasets](#how-to-create-new-datasets)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Code of Conduct](#code-of-conduct)

---

## Development Setup

### Prerequisites

| Requirement | Version |
|-------------|---------|
| Python      | ≥ 3.10  |
| Git         | any recent |
| (Optional) Docker | for containerised runs |

### Clone, Install & Verify

```bash
# 1. Clone the repository
git clone https://github.com/Contextually-AI/cri-benchmark.git
cd cri

# 2. Create & activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows

# 3. Install in editable mode with dev dependencies
pip install -e ".[dev]"

# 4. (Optional) Install the RAG adapter extras
pip install -e ".[rag]"

# 5. Verify everything works
pytest                           # run the full test suite
ruff check src/ tests/           # lint
mypy src/                        # type-check
```

> **Tip:** If you only want to run a quick sanity check, `pytest tests/test_models.py` exercises the core data models without any external dependencies.

### Project Layout

```
cri/
├── src/cri/                          # Main package
│   ├── models.py                     # Pydantic v2 data models
│   ├── adapter.py                    # MemoryAdapter protocol
│   ├── judge.py                      # BinaryJudge (LLM-as-judge)
│   ├── runner.py                     # CLI entry point (Click)
│   ├── reporter.py                   # Result reporting / formatting
│   ├── performance.py                # Latency & memory profiling
│   ├── scoring/
│   │   ├── engine.py                 # ScoringEngine orchestrator
│   │   ├── rubrics.py                # Judge prompt templates
│   │   └── dimensions/              # One module per metric
│   │       ├── base.py              # MetricDimension ABC
│   │       ├── pas.py               # Persona Accuracy Score
│   │       ├── dbu.py               # Dynamic Belief Updating
│   │       ├── mei.py               # Memory Efficiency Index
│   │       ├── tc.py                # Temporal Coherence
│   │       ├── crq.py               # Conflict Resolution Quality
│   │       └── qrp.py              # Query Response Precision
│   └── datasets/
│       ├── loader.py                 # Dataset loading
│       ├── generator.py              # Synthetic dataset generation
│       └── personas/
│           └── specs.py              # PersonaSpec definitions
├── tests/                            # Test suite (mirrors src/ layout)
│   ├── conftest.py                   # Shared fixtures & mock helpers
│   ├── test_models.py
│   ├── test_dimensions/             # Per-dimension scorer tests
│   └── ...
├── docs/                             # Documentation (Markdown + Mermaid)
├── examples/                         # Example adapters & usage
└── pyproject.toml                    # Build config, tool settings
```

---

## Code Style

We enforce a consistent code style across the project. CI will reject PRs that fail lint or type checks.

### Toolchain

| Tool   | Purpose              | Config location   |
|--------|----------------------|-------------------|
| **ruff**  | Formatting + linting | `pyproject.toml` `[tool.ruff]` |
| **mypy**  | Static type checking | `pyproject.toml` `[tool.mypy]` |

### Running the checks

```bash
# Format (auto-fix)
ruff format src/ tests/

# Lint (report issues)
ruff check src/ tests/

# Lint (auto-fix where possible)
ruff check --fix src/ tests/

# Type check (strict mode)
mypy src/
```

### Style Rules

1. **Type hints everywhere.** All function signatures must have complete type annotations — parameters *and* return types. Use `from __future__ import annotations` at the top of every module for PEP 604 union syntax (`str | None`).

2. **Docstrings on all public symbols.** Use Google-style or NumPy-style docstrings. At minimum, include a one-line summary. For complex functions, document parameters, return values, and notable exceptions.

   ```python
   def retrieve(self, query: str) -> list[StoredFact]:
       """Retrieve stored facts relevant to the given query.

       Parameters
       ----------
       query : str
           Natural-language description of the information being requested.

       Returns
       -------
       list[StoredFact]
           Facts the memory system considers relevant. May be empty.
       """
   ```

3. **Line length: 100 characters.** Configured in `pyproject.toml`.

4. **Import order.** Ruff enforces isort-compatible ordering: stdlib → third-party → local. Let `ruff format` handle it.

5. **No bare `except`.** Always catch specific exception types.

6. **Prefer `Literal` and `Enum` over raw strings** for domain values (see `cri.models.Dimension`).

---

## How to Add a New Metric Dimension

The CRI Benchmark evaluates memory systems across multiple **dimensions**, each measuring a distinct property. Adding a new dimension involves four steps:

### Step 1 — Create the Scorer Module

Create a new file in `src/cri/scoring/dimensions/`. For example, to add a "Semantic Stability" dimension (`SST`):

```python
# src/cri/scoring/dimensions/sst.py
"""Semantic Stability (SST) dimension scorer.

Measures whether the memory system's stored knowledge remains semantically
stable across repeated ingestion of the same information.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from cri.models import DimensionResult, Verdict
from cri.scoring.dimensions.base import MetricDimension
from cri.scoring.rubrics import sst_check  # you'll create this next

if TYPE_CHECKING:
    from cri.adapter import MemoryAdapter
    from cri.judge import BinaryJudge
    from cri.models import GroundTruth


class SSTDimension(MetricDimension):
    name = "SST"
    description = "Measures semantic stability of stored knowledge over repeated ingestion."

    async def score(
        self,
        adapter: MemoryAdapter,
        ground_truth: GroundTruth,
        judge: BinaryJudge,
    ) -> DimensionResult:
        passed = 0
        total = 0
        details: list[dict] = []

        # 1. Derive checks from ground_truth
        # 2. Retrieve facts from the adapter for each check
        # 3. Use the judge to evaluate (YES/NO verdict)
        # 4. Aggregate results

        for check in self._derive_checks(ground_truth):
            total += 1
            facts = adapter.retrieve(check.topic)
            prompt = sst_check(check, facts)
            result = await judge.evaluate(f"sst-{check.id}", prompt)

            if result.verdict == Verdict.YES:
                passed += 1

            details.append({
                "check_id": f"sst-{check.id}",
                "verdict": result.verdict.value,
            })

        score = passed / total if total > 0 else 0.0
        return DimensionResult(
            dimension_name=self.name,
            score=score,
            passed_checks=passed,
            total_checks=total,
            details=details,
        )
```

**Key rules:**

- Subclass `MetricDimension` (from `cri.scoring.dimensions.base`).
- Define class-level `name` (short code) and `description` (human-readable).
- Implement the `async def score(...)` method returning a `DimensionResult`.
- The `__init_subclass__` hook will raise `TypeError` if `name` or `description` is missing.

### Step 2 — Add a Rubric Function

Add a binary-verdict rubric function in `src/cri/scoring/rubrics.py`:

```python
def sst_check(check: SSTCheck, facts: list[StoredFact]) -> str:
    """Generate a judge prompt for a semantic stability check.

    Returns a prompt expecting YES if the system preserved semantic meaning,
    NO if meaning drifted or was lost.
    """
    fact_text = format_facts(facts)
    return (
        f"The memory system was given the same information multiple times.\n"
        f"Expected stable fact: {check.expected}\n\n"
        f"Stored facts:\n{fact_text}\n\n"
        f"Does the stored knowledge preserve the original semantic meaning?\n"
        f"Answer YES or NO."
    )
```

Rubric design guidelines:

- Keep prompts **concise and unambiguous**.
- Respect `MAX_FACTS_PER_PROMPT` (currently 30) — truncate with `format_facts()`.
- Emphasise **semantic equivalence**, not exact text matching.
- For negative checks (where YES = failure), make the interpretation explicit in the prompt.

### Step 3 — Register the Dimension

1. **Export from the dimensions package** — add your class to `src/cri/scoring/dimensions/__init__.py`:

   ```python
   from cri.scoring.dimensions.sst import SSTDimension

   __all__ = [
       # ... existing exports ...
       "SSTDimension",
   ]
   ```

2. **Register in the ScoringEngine** — add your dimension to the default dimension list in `src/cri/scoring/engine.py`:

   ```python
   # In ScoringEngine.__init__ or the default dimensions list:
   SSTDimension(),
   ```

3. **Add to the Dimension enum** — add `SST = "sst"` to `cri.models.Dimension`.

4. **Add a default weight** — update `ScoringConfig` default weights so they still sum to 1.0.

### Step 4 — Document the Dimension

Create `docs/methodology/metrics/sst.md` following the pattern of existing metric docs (e.g., `pas.md`, `dbu.md`). Include:

- What the dimension measures and why it matters
- How checks are derived from ground truth
- The rubric logic and verdict interpretation
- Scoring formula
- Example scenarios
- Known limitations

---

## How to Add a New Adapter

An adapter wraps your memory system so the CRI benchmark can evaluate it. Thanks to Python's structural subtyping, **you don't need to import or subclass anything from CRI**.

### Step 1 — Implement the `MemoryAdapter` Protocol

Your class must expose three methods with these exact signatures:

```python
from cri.models import Message, StoredFact

class MyMemoryAdapter:
    """CRI adapter for the Acme Memory Engine."""

    def __init__(self, config: dict) -> None:
        self._engine = AcmeEngine(config)

    def ingest(self, messages: list[Message]) -> None:
        """Process conversation messages and store extracted facts."""
        for msg in messages:
            self._engine.process(
                text=msg.content,
                role=msg.role,
                timestamp=msg.timestamp,
            )

    def retrieve(self, query: str) -> list[StoredFact]:
        """Retrieve facts relevant to a query string."""
        results = self._engine.search(query)
        return [
            StoredFact(text=r.text, metadata={"score": r.relevance})
            for r in results
        ]

    def get_events(self) -> list[StoredFact]:
        """Dump the entire fact store for auditing."""
        return [
            StoredFact(text=f.text, metadata=f.meta)
            for f in self._engine.dump()
        ]
```

**Method responsibilities:**

| Method | Called when | Expected behavior |
|--------|-----------|-------------------|
| `ingest(messages)` | Before scoring begins | Parse messages, extract facts, store them |
| `retrieve(query)` | During each dimension's scoring | Return *only* facts relevant to the query |
| `get_events()` | During memory hygiene audits | Return *every* stored fact |

### Step 2 — Verify Protocol Compliance

```python
from cri.adapter import MemoryAdapter

adapter = MyMemoryAdapter(config={})
assert isinstance(adapter, MemoryAdapter), "Protocol not satisfied!"
```

The `@runtime_checkable` decorator on `MemoryAdapter` enables this check. The CRI runner performs it automatically before starting a benchmark run.

### Step 3 — Register Your Adapter (Optional)

If contributing the adapter to the CRI repository:

1. Place it in `src/cri/adapters/` (or `examples/adapters/` for reference implementations).
2. Add any extra dependencies to `pyproject.toml` under an optional extras group:
   ```toml
   [project.optional-dependencies]
   acme = ["acme-memory>=1.0"]
   ```
3. Write tests in `tests/test_adapters/test_acme_adapter.py`.

### Step 4 — Document Your Adapter

- Add usage instructions to `docs/guides/integration.md` or a new file under `docs/guides/`.
- Include a minimal working example showing how to run the benchmark with your adapter.

---

## How to Create New Datasets

Datasets drive the CRI benchmark. Each dataset consists of a simulated conversation, ground truth annotations, and metadata.

### Dataset Structure

Each dataset lives in its own directory:

```
src/cri/datasets/persona-4-expert/
├── conversations.jsonl      # One Message JSON object per line
├── ground_truth.json        # Expected outcomes (GroundTruth model)
└── metadata.json            # DatasetMetadata (persona ID, complexity, etc.)
```

### Step 1 — Define a Persona Specification

Create a `PersonaSpec` in `src/cri/datasets/personas/specs.py` (or a new file):

```python
PERSONA_4_EXPERT = PersonaSpec(
    id="persona-4-expert",
    name="Expert User",
    complexity=3,   # 1=basic, 2=intermediate, 3=advanced
    profile_dimensions=[
        ProfileDimension(
            dimension_name="occupation",
            value="Machine Learning Researcher",
            query_topic="What does the user do for work?",
            category="demographics",
        ),
        # ... more dimensions
    ],
    changes=[
        BeliefChange(
            fact="employer",
            old_value="University Lab",
            new_value="AI Startup",
            query_topic="Where does the user work?",
            changed_around_msg=25,
            key_messages=[25, 26],
        ),
        # ... more changes
    ],
    conflicts=[...],
    temporal_facts=[...],
    noise_examples=[...],
    signal_examples=[...],
    query_relevance_pairs=[...],
)
```

**Guidelines for persona design:**

- Cover all six dimensions: **PAS**, **DBU**, **MEI**, **TC**, **CRQ**, **QRP**.
- Include at least 2–3 belief changes that happen at different points in the conversation.
- Add conflict scenarios with clear correct resolutions.
- Include both signal messages (fact-bearing) and noise messages (greetings, filler).
- Use the **6-W framework** for profile categories: WHO, WHAT, WHERE, WHEN, WHY, HOW.

### Step 2 — Generate the Dataset

```python
from cri.models import GeneratorConfig
from cri.datasets.generator import DatasetGenerator
from pathlib import Path

gen = DatasetGenerator(GeneratorConfig(seed=42))
dataset = gen.generate(PERSONA_4_EXPERT)
gen.save_dataset(dataset, Path("src/cri/datasets/persona-4-expert"))
```

The generator is **deterministic** — the same seed always produces the same output. No LLM API calls are made during generation.

### Step 3 — Validate the Dataset

```python
from cri.datasets.loader import load_dataset

# Will raise ValidationError if structure is invalid
ds = load_dataset(Path("src/cri/datasets/persona-4-expert"))
assert len(ds.conversations) > 0
assert ds.ground_truth.final_profile
assert len(ds.ground_truth.changes) > 0
```

Also verify manually:

- [ ] `conversations.jsonl` messages are in chronological order
- [ ] `ground_truth.json` reflects the *final* state after all changes
- [ ] Every `BeliefChange` references valid `key_messages` IDs
- [ ] `query_relevance_pairs` have non-overlapping relevant/irrelevant facts
- [ ] Temporal facts have valid date ranges

### Step 4 — Add to the Canonical Suite (Optional)

1. Add the persona's JSON data files to `src/cri/datasets/<persona-id>/`.
2. Verify the dataset loads and validates: `cri validate-dataset src/cri/datasets/<persona-id>`.

---

## Testing

We use **pytest** with **pytest-asyncio** for async test support.

### Running Tests

```bash
# Full suite
pytest

# With coverage report
pytest --cov=cri --cov-report=html
# Open htmlcov/index.html to view

# Specific file
pytest tests/test_models.py

# Specific test by name
pytest -k "test_persona_accuracy"

# Only dimension tests
pytest tests/test_dimensions/

# Verbose output
pytest -v
```

### Where Tests Go

Tests mirror the source layout:

| Source module | Test file |
|--------------|-----------|
| `src/cri/models.py` | `tests/test_models.py` |
| `src/cri/adapter.py` | `tests/test_adapter.py` |
| `src/cri/judge.py` | `tests/test_judge.py`, `tests/test_binary_judge.py` |
| `src/cri/scoring/rubrics.py` | `tests/test_rubrics.py` |
| `src/cri/scoring/engine.py` | `tests/test_scoring_engine.py` |
| `src/cri/scoring/dimensions/pas.py` | `tests/test_dimensions/test_pas.py` |
| `src/cri/scoring/dimensions/dbu.py` | `tests/test_dimensions/test_dbu.py` |
| `src/cri/datasets/loader.py` | `tests/test_dataset_loader.py` |
| `src/cri/reporter.py` | `tests/test_reporter.py` |

### What to Test

1. **Models** — Validate construction, serialization/deserialization, default values, and validation errors.

2. **Scorers / Dimensions** — Use the `MockJudge` from `conftest.py` to test scoring logic without LLM calls:
   ```python
   async def test_pas_perfect_score(mock_judge, sample_ground_truth):
       scorer = ProfileAccuracyScore()
       # MockJudge returns YES by default → perfect score
       result = await scorer.score(adapter, sample_ground_truth, mock_judge)
       assert result.score == 1.0
   ```

3. **Adapters** — Verify protocol compliance and correct fact extraction / retrieval behavior.

4. **Rubrics** — Check that generated prompts contain the expected elements and stay within `MAX_FACTS_PER_PROMPT`.

5. **Dataset loading** — Test both happy path and error cases (missing files, malformed JSON).

### Shared Fixtures

`tests/conftest.py` provides reusable fixtures:

| Fixture | Description |
|---------|-------------|
| `sample_messages` | 12 messages spanning a multi-day conversation |
| `sample_ground_truth` | Fully populated `GroundTruth` with all annotation types |
| `sample_stored_facts` | List of `StoredFact` objects |
| `sample_scoring_config` | `ScoringConfig` with default weights |
| `mock_judge` | `MockJudge` that returns `YES` by default (configurable) |
| `mock_adapter` | `MockAdapter` with basic store/query support |

**Configuring the mock judge per test:**

```python
from cri.models import Verdict

def test_partial_failures(mock_judge):
    mock_judge.overrides["check-occupation"] = Verdict.NO
    # Now only that specific check will fail
```

### Test Requirements

- **No network calls.** Tests must never hit external APIs. The CI pipeline sets `CRI_TESTING=1` and blanks API keys.
- **Deterministic.** Tests must produce the same result on every run. Use fixed seeds, mock randomness where needed.
- **Fast.** Individual tests should complete in under 1 second. If a test is slow, mark it with `@pytest.mark.slow`.

---

## Pull Request Process

### 1. Branch

Create a feature branch from `main`:

```bash
git checkout main
git pull origin main
git checkout -b feature/my-contribution
```

Branch naming conventions:

| Prefix | Use for |
|--------|---------|
| `feature/` | New features, dimensions, adapters |
| `fix/` | Bug fixes |
| `docs/` | Documentation-only changes |
| `refactor/` | Code restructuring without behavior change |
| `test/` | Test additions or improvements |

### 2. Develop & Test

```bash
# Make your changes, then:
ruff format src/ tests/          # format
ruff check src/ tests/           # lint
mypy src/                        # type-check
pytest                           # full test suite
```

All four checks must pass before submitting.

### 3. Commit

Write clear, descriptive commit messages:

```
feat(dimensions): add Semantic Stability (SST) dimension

- Implement SSTDimension scorer with binary verdict model
- Add sst_check rubric function
- Register in ScoringEngine and Dimension enum
- Add documentation in docs/methodology/metrics/sst.md
- Add tests in tests/test_dimensions/test_sst.py
```

### 4. Submit

```bash
git push origin feature/my-contribution
```

Then open a Pull Request on GitHub. In the PR description:

- **What**: Describe the change clearly
- **Why**: Explain the motivation
- **How to test**: Steps reviewers can follow to verify
- **Related issues**: Link any relevant GitHub issues

### 5. Review & Merge

- CI must be green (lint + type-check + tests).
- At least one maintainer review is required.
- Squash-merge is preferred for clean history.

---

## Code of Conduct

Be respectful, constructive, and inclusive. We follow standard open-source community guidelines. Harassment, discrimination, or hostility of any kind will not be tolerated.

---

## License

By contributing, you agree that your contributions will be licensed under the **MIT License** (see [LICENSE](LICENSE)).

---

## Questions?

- Open a [GitHub Discussion](https://github.com/Contextually-AI/cri-benchmark/discussions) for general questions
- Open a [GitHub Issue](https://github.com/Contextually-AI/cri-benchmark/issues) for bugs or feature requests
- Read the [Documentation](docs/README.md) for detailed guides on every aspect of the benchmark
