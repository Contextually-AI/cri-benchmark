# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-03-16

### Added

- Core benchmark framework with nine evaluation dimensions (PAS, DBU, MEI, TC, CRQ, QRP, SFC, LNC, ARS)
- Composite CRI score with configurable dimension weights
- Hybrid scoring engine: deterministic checks + LLM-as-judge with majority voting
- `MemoryAdapter` protocol (structural subtyping) aligned with UPP operations
- Canonical datasets: persona-1-basic, persona-2-intermediate, persona-3-advanced
- Dataset generator for custom benchmark scenarios
- Performance profiling: latency and memory growth reporting
- Four reference adapter implementations: full-context, RAG, no-memory, UPP
- CLI entry point (`cri run`, `cri list-datasets`)
- Rich console reporter with bar charts and dimension breakdowns
- JSON and Markdown report output formats
- CI pipeline with lint, type-check, and test matrix (Python 3.11, 3.12)
- Automated PyPI publishing via GitHub Actions (trusted publisher)

[Unreleased]: https://github.com/Contextually-AI/cri-benchmark/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Contextually-AI/cri-benchmark/releases/tag/v0.1.0
