# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.2] - 2026-04-06

### Added

- Richer dataset metadata: `persona_name`, `description`, and `complexity_level` fields on `DatasetMetadata`
- Per-dimension progress reporting during benchmark evaluation with elapsed time and pass/fail indicators
- New dataset loader functions: `get_persona()`, `list_persona_specs()`, `load_persona_spec()`
- Package data support: datasets are now bundled inside the `cri` package for reliable `pip install` usage

### Changed

- Datasets moved from `datasets/canonical/` to `src/cri/datasets/` and included as package data
- `list_canonical_datasets()` renamed to `list_datasets()`
- `RichPersonaSpec` renamed to `PersonaSpec`; persona data loaded from JSON files instead of hardcoded Python constants
- LLM judge response caching now enabled by default (`--cache` flag)
- Dataset loader rewritten to support the new package-data layout and persona loading

### Removed

- ~1,700 lines of hardcoded persona specifications from `specs.py`
- `datasets/canonical/` directory and `datasets/README.md` (datasets now live inside the package)

## [0.1.1] - 2026-04-02

### Added

- Disk-based LLM response cache (`llm_cache.py`) to avoid redundant API calls across runs
- Support for single-judge mode (`num_runs=1`) and single-metric evaluation via CLI
- Async parallel LLM judging with semaphore-based concurrency control for significantly faster benchmark runs
- PowerShell runner script (`run.ps1`) for Windows support

### Changed

- Replaced three generated canonical datasets (`persona-1-basic`, `persona-2-intermediate`, `persona-3-advanced`) with a single hand-crafted dataset (`persona-1-base`: Marcus Rivera — 2862 messages, 289 days)
- MEI (Memory Efficiency Index) formula changed to pure coverage metric, removing the efficiency penalty
- Reduced benchmark from nine to six scoring dimensions by removing SFC, LNC, and ARS
- Rewrote scoring rubrics to match the six-dimension structure
- Improved type annotations and static analysis compliance across all scoring dimensions
- Refactored tests to use factory-based response generation instead of mocks
- BinaryJudge now accepts even `num_runs` values (ties resolve to NO) with a warning

### Fixed

- MEI truncation bug that could produce incorrect scores for edge-case inputs
- Empty-facts inflation bias where scores were artificially high when no facts were present
- Vacuous defaults bias in scoring that could mask poor memory system performance

### Removed

- SFC (Signal Fidelity & Calibration) scoring dimension
- LNC (Longitudinal Narrative Coherence) scoring dimension
- ARS (Adaptive Retrieval Strategy) scoring dimension
- `ForgettableFact` model and related dataset generation logic

## [0.1.0] - 2026-03-16

### Added

- Core benchmark framework with six evaluation dimensions (PAS, DBU, MEI, TC, CRQ, QRP)
- Composite CRI score with configurable dimension weights
- Hybrid scoring engine: deterministic checks + LLM-as-judge with majority voting
- `MemoryAdapter` protocol (structural subtyping) aligned with UPP operations
- Canonical dataset: persona-1-base (Marcus Rivera — 2862 messages, 289 days)
- Dataset generator for custom benchmark scenarios
- Performance profiling: latency and memory growth reporting
- Four reference adapter implementations: full-context, RAG, no-memory, UPP
- CLI entry point (`cri run`, `cri list-datasets`)
- Rich console reporter with bar charts and dimension breakdowns
- JSON and Markdown report output formats
- CI pipeline with lint, type-check, and test matrix (Python 3.11, 3.12)
- Automated PyPI publishing via GitHub Actions (trusted publisher)

[Unreleased]: https://github.com/Contextually-AI/cri-benchmark/compare/v0.1.2...HEAD
[0.1.2]: https://github.com/Contextually-AI/cri-benchmark/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/Contextually-AI/cri-benchmark/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/Contextually-AI/cri-benchmark/releases/tag/v0.1.0
