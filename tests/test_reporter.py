"""Tests for the CRI Benchmark report generator.

Tests cover the API: ``to_json``, ``to_markdown``, ``to_console``,
``generate_comparison_table``.

Coverage includes:
- to_json: valid JSON, file writing, round-trip parsing
- to_markdown: structure, section headings, dimension display, performance metrics
- to_console: smoke test
- generate_comparison_table: column headers, mixed results
"""

from __future__ import annotations

import json as json_mod
from pathlib import Path

import pytest

from cri.models import (
    BenchmarkResult,
    CRIResult,
    DimensionResult,
    JudgmentResult,
    PerformanceProfile,
    Verdict,
)
from cri.reporter import BenchmarkReporter

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_cri_result() -> CRIResult:
    """Create a CRIResult."""
    return CRIResult(
        system_name="ontology-memory-v2",
        cri=7.80,
        pas=8.50,
        dbu=7.20,
        tc=5.50,
        crq=9.10,
        qrp=7.00,
        mei=6.80,
        dimension_weights={
            "PAS": 0.25,
            "DBU": 0.20,
            "MEI": 0.20,
            "TC": 0.15,
            "CRQ": 0.10,
            "QRP": 0.10,
        },
        details={
            "PAS": DimensionResult(
                dimension_name="PAS",
                score=0.85,
                passed_checks=17,
                total_checks=20,
            ),
            "DBU": DimensionResult(
                dimension_name="DBU",
                score=0.72,
                passed_checks=18,
                total_checks=25,
            ),
            "MEI": DimensionResult(
                dimension_name="MEI",
                score=0.68,
                passed_checks=34,
                total_checks=50,
            ),
            "TC": DimensionResult(
                dimension_name="TC",
                score=0.55,
                passed_checks=11,
                total_checks=20,
            ),
            "CRQ": DimensionResult(
                dimension_name="CRQ",
                score=0.91,
                passed_checks=10,
                total_checks=11,
            ),
            "QRP": DimensionResult(
                dimension_name="QRP",
                score=0.70,
                passed_checks=14,
                total_checks=20,
            ),
        },
    )


@pytest.fixture
def sample_benchmark_result(sample_cri_result: CRIResult) -> BenchmarkResult:
    """Create a full BenchmarkResult wrapping a CRIResult."""
    return BenchmarkResult(
        run_id="run-001",
        adapter_name="ontology-memory-v2",
        dataset_id="persona-complex-90d",
        started_at="2026-03-11T10:00:00",
        completed_at="2026-03-11T10:05:30",
        cri_result=sample_cri_result,
        performance_profile=PerformanceProfile(
            ingest_latency_ms=12.5,
            query_latency_avg_ms=45.3,
            query_latency_p95_ms=120.0,
            query_latency_p99_ms=250.0,
            total_facts_stored=342,
            memory_growth_curve=[(10, 15), (50, 80), (100, 160)],
            judge_api_calls=146,
            judge_total_cost_estimate=1.23,
        ),
        judge_log=[
            JudgmentResult(
                check_id="chk-001",
                verdict=Verdict.YES,
                votes=[Verdict.YES, Verdict.YES],
                unanimous=True,
                prompt="Is X correct?",
                raw_responses=["YES", "YES"],
            ),
        ],
    )


@pytest.fixture
def reporter() -> BenchmarkReporter:
    return BenchmarkReporter()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestToConsole:
    def test_smoke(
        self,
        reporter: BenchmarkReporter,
        sample_benchmark_result: BenchmarkResult,
    ) -> None:
        reporter.to_console(sample_benchmark_result)

    def test_cri_result_direct(
        self,
        reporter: BenchmarkReporter,
        sample_cri_result: CRIResult,
    ) -> None:
        reporter.to_console(sample_cri_result)


class TestToJson:
    def test_valid_json(
        self,
        reporter: BenchmarkReporter,
        sample_benchmark_result: BenchmarkResult,
    ) -> None:
        result = reporter.to_json(sample_benchmark_result)
        data = json_mod.loads(result)
        assert "cri_result" in data

    def test_write_file(
        self,
        reporter: BenchmarkReporter,
        sample_benchmark_result: BenchmarkResult,
        tmp_path: Path,
    ) -> None:
        out = tmp_path / "report.json"
        reporter.to_json(sample_benchmark_result, output_path=out)
        assert out.exists()
        data = json_mod.loads(out.read_text())
        assert "cri_result" in data


class TestToMarkdown:
    def test_structure(
        self,
        reporter: BenchmarkReporter,
        sample_benchmark_result: BenchmarkResult,
    ) -> None:
        md = reporter.to_markdown(sample_benchmark_result)
        assert "# CRI Benchmark Report" in md
        assert "ontology-memory-v2" in md
        assert "PAS" in md

    def test_performance_section(
        self,
        reporter: BenchmarkReporter,
        sample_benchmark_result: BenchmarkResult,
    ) -> None:
        md = reporter.to_markdown(sample_benchmark_result)
        assert "Performance" in md
        assert "12.5" in md  # ingest latency


class TestComparisonTable:
    def test_columns(self, reporter: BenchmarkReporter, sample_cri_result: CRIResult) -> None:
        table = reporter.generate_comparison_table([sample_cri_result])
        assert "System" in table
        assert "CRI" in table
        assert "PAS" in table

    def test_multiple_systems(
        self,
        reporter: BenchmarkReporter,
        sample_cri_result: CRIResult,
    ) -> None:
        r2 = CRIResult(
            system_name="baseline-rag",
            cri=4.50,
            pas=5.00,
            dbu=3.00,
            mei=4.50,
            tc=0.0,
            crq=0.0,
            qrp=0.0,
            dimension_weights={
                "PAS": 0.25,
                "DBU": 0.20,
                "MEI": 0.20,
                "TC": 0.15,
                "CRQ": 0.10,
                "QRP": 0.10,
            },
            details={},
        )
        table = reporter.generate_comparison_table([sample_cri_result, r2])
        assert "ontology-memory-v2" in table
        assert "baseline-rag" in table


class TestScoreStyle:
    def test_boundaries(self) -> None:
        r = BenchmarkReporter
        assert "green" in r._score_style(0.95)
        assert "green" in r._score_style(0.75)
        assert "yellow" in r._score_style(0.55)
        assert "red" in r._score_style(0.35)
        assert "red" in r._score_style(0.15)
