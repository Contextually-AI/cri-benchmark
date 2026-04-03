"""Tests for the MEI (Memory Efficiency Index) dimension scorer."""

from __future__ import annotations

from typing import Any

import pytest

from cri.models import (
    DimensionResult,
    GroundTruth,
    ProfileDimension,
    StoredFact,
)
from cri.scoring.dimensions.mei import MEIDimension

# ---------------------------------------------------------------------------
# Test helpers — lightweight mocks
# ---------------------------------------------------------------------------


class MockCoverageJudge:
    """Mock judge with a ``judge_coverage()`` method.

    Returns the same *covered_indices* set for every chunk call.  Because
    MEI unions results across all chunks, returning the same set every time
    is idempotent and produces the expected final coverage.
    """

    def __init__(self, covered_indices: set[int] | None = None) -> None:
        self.covered_indices: set[int] = covered_indices if covered_indices is not None else set()
        self.call_log: list[dict[str, Any]] = []

    async def judge_coverage(self, check_id: str, prompt: str) -> set[int]:
        self.call_log.append({"check_id": check_id, "prompt": prompt})
        return self.covered_indices


class MockAdapter:
    """Mock adapter with a fixed list of stored facts."""

    def __init__(self, stored_facts: list[StoredFact] | None = None) -> None:
        self._facts: list[StoredFact] = stored_facts or []
        self.queries: list[str] = []

    def retrieve(self, topic: str) -> list[StoredFact]:
        self.queries.append(topic)
        return self._facts

    def get_events(self) -> list[StoredFact]:
        return list(self._facts)

    def ingest(self, messages: list[Any]) -> None:
        pass


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def empty_ground_truth() -> GroundTruth:
    """GroundTruth with an empty profile (no gt facts)."""
    return GroundTruth(
        final_profile={},
        changes=[],
        noise_examples=[],
        signal_examples=[],
        conflicts=[],
        temporal_facts=[],
        query_relevance_pairs=[],
    )


@pytest.fixture()
def simple_ground_truth() -> GroundTruth:
    """GroundTruth with three single-value profile dimensions."""
    return GroundTruth(
        final_profile={
            "occupation": ProfileDimension(
                dimension_name="occupation",
                value="Software Engineer",
                query_topic="What does the user do for work?",
                category="demographics",
            ),
            "location": ProfileDimension(
                dimension_name="location",
                value="New York",
                query_topic="Where does the user live?",
                category="demographics",
            ),
            "age": ProfileDimension(
                dimension_name="age",
                value="30",
                query_topic="How old is the user?",
                category="demographics",
            ),
        },
        changes=[],
        noise_examples=[],
        signal_examples=[],
        conflicts=[],
        temporal_facts=[],
        query_relevance_pairs=[],
    )


@pytest.fixture()
def multi_value_ground_truth() -> GroundTruth:
    """GroundTruth with one multi-value profile dimension."""
    return GroundTruth(
        final_profile={
            "hobbies": ProfileDimension(
                dimension_name="hobbies",
                value=["hiking", "photography"],
                query_topic="What are the user's hobbies?",
                category="interests",
            ),
        },
        changes=[],
        noise_examples=[],
        signal_examples=[],
        conflicts=[],
        temporal_facts=[],
        query_relevance_pairs=[],
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMEIDimensionAttributes:
    """Verify class-level attributes."""

    def test_name(self) -> None:
        scorer = MEIDimension()
        assert scorer.name == "MEI"

    def test_description_is_non_empty(self) -> None:
        scorer = MEIDimension()
        assert len(scorer.description) > 20

    def test_repr(self) -> None:
        scorer = MEIDimension()
        assert repr(scorer) == "MEIDimension(name='MEI')"


class TestMEIEmptyProfile:
    """Empty profile → score 0.0 (no data to evaluate)."""

    @pytest.mark.asyncio
    async def test_empty_profile_returns_zero(self, empty_ground_truth: GroundTruth) -> None:
        scorer = MEIDimension()
        adapter = MockAdapter(stored_facts=[StoredFact(text="some fact")])
        judge = MockCoverageJudge()

        result = await scorer.score(adapter, empty_ground_truth, judge)

        assert isinstance(result, DimensionResult)
        assert result.dimension_name == "MEI"
        assert result.score == 0.0
        assert result.passed_checks == 0
        assert result.total_checks == 0
        assert result.details == []

    @pytest.mark.asyncio
    async def test_empty_profile_does_not_call_judge(self, empty_ground_truth: GroundTruth) -> None:
        scorer = MEIDimension()
        adapter = MockAdapter()
        judge = MockCoverageJudge()

        await scorer.score(adapter, empty_ground_truth, judge)

        assert judge.call_log == []


class TestMEIAllFactsCovered:
    """All gt facts covered with exact stored count → high score."""

    @pytest.mark.asyncio
    async def test_all_covered_exact_storage(self, simple_ground_truth: GroundTruth) -> None:
        """3 gt facts, 3 stored facts, all covered → coverage=1.0 efficiency=1.0, MEI=1.0."""
        scorer = MEIDimension()
        adapter = MockAdapter(
            stored_facts=[
                StoredFact(text="User is a Software Engineer."),
                StoredFact(text="User lives in New York."),
                StoredFact(text="User is 30 years old."),
            ]
        )
        judge = MockCoverageJudge(covered_indices={0, 1, 2})

        result = await scorer.score(adapter, simple_ground_truth, judge)

        assert result.dimension_name == "MEI"
        assert result.score == 1.0
        assert result.passed_checks == 3
        assert result.total_checks == 3

    @pytest.mark.asyncio
    async def test_all_covered_more_stored(self, simple_ground_truth: GroundTruth) -> None:
        """3 gt facts covered, 6 stored facts → coverage=1.0 (pure coverage, no efficiency penalty)."""
        scorer = MEIDimension()
        adapter = MockAdapter(stored_facts=[StoredFact(text=f"fact-{i}") for i in range(6)])
        judge = MockCoverageJudge(covered_indices={0, 1, 2})

        result = await scorer.score(adapter, simple_ground_truth, judge)

        assert result.score == 1.0
        assert result.passed_checks == 3


class TestMEIMixedCoverage:
    """Mixed coverage → proportional score using harmonic mean."""

    @pytest.mark.asyncio
    async def test_partial_coverage(self, simple_ground_truth: GroundTruth) -> None:
        """2 of 3 gt facts covered with 3 stored → coverage=2/3, efficiency=2/3, MEI=2/3."""
        scorer = MEIDimension()
        adapter = MockAdapter(
            stored_facts=[
                StoredFact(text="fact-a"),
                StoredFact(text="fact-b"),
                StoredFact(text="fact-c"),
            ]
        )
        judge = MockCoverageJudge(covered_indices={0, 1})  # index 2 not covered

        result = await scorer.score(adapter, simple_ground_truth, judge)

        # 2 covered out of 3, 3 stored → coverage=2/3, efficiency=2/3
        # harmonic mean of (2/3, 2/3) = 2/3
        assert result.passed_checks == 2
        assert result.total_checks == 3
        assert abs(result.score - round(2.0 / 3.0, 4)) < 1e-9

    @pytest.mark.asyncio
    async def test_zero_stored_facts_returns_zero(self, simple_ground_truth: GroundTruth) -> None:
        """No stored facts → score 0.0 immediately."""
        scorer = MEIDimension()
        adapter = MockAdapter(stored_facts=[])
        judge = MockCoverageJudge()

        result = await scorer.score(adapter, simple_ground_truth, judge)

        assert result.score == 0.0
        assert result.passed_checks == 0
        assert result.total_checks == 3
        # Judge should not have been called
        assert judge.call_log == []


class TestMEIHarmonicMean:
    """Score uses harmonic mean of coverage and efficiency."""

    @pytest.mark.asyncio
    async def test_harmonic_mean_applied(self, simple_ground_truth: GroundTruth) -> None:
        """Verify MEI is harmonic mean of coverage and efficiency, not arithmetic."""
        scorer = MEIDimension()
        adapter = MockAdapter(stored_facts=[StoredFact(text=f"f{i}") for i in range(3)])
        judge = MockCoverageJudge(covered_indices={0, 1, 2})

        result = await scorer.score(adapter, simple_ground_truth, judge)

        # coverage=1.0, efficiency=1.0 → harmonic=1.0
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_harmonic_mean_vs_arithmetic(self, simple_ground_truth: GroundTruth) -> None:
        """Pure coverage: score equals coverage regardless of stored fact count."""
        scorer = MEIDimension()
        # 3 gt facts, 9 stored facts. All 3 covered → coverage=1.0
        adapter = MockAdapter(stored_facts=[StoredFact(text=f"f{i}") for i in range(9)])
        judge = MockCoverageJudge(covered_indices={0, 1, 2})

        result = await scorer.score(adapter, simple_ground_truth, judge)

        assert result.score == 1.0


class TestMEIChunkScanning:
    """Coverage is checked correctly across multiple chunks."""

    @pytest.mark.asyncio
    async def test_large_storage_uses_multiple_chunks(self, simple_ground_truth: GroundTruth) -> None:
        """35 stored facts → 2 chunks (30 + 5), all chunks scanned concurrently."""
        scorer = MEIDimension()
        adapter = MockAdapter(stored_facts=[StoredFact(text=f"f{i}") for i in range(35)])
        judge = MockCoverageJudge(covered_indices={0, 1, 2})

        result = await scorer.score(adapter, simple_ground_truth, judge)

        # All chunks fire concurrently (no early exit with parallel scanning)
        assert len(judge.call_log) == 2
        assert result.passed_checks == 3

    @pytest.mark.asyncio
    async def test_all_chunks_scanned_concurrently(self, simple_ground_truth: GroundTruth) -> None:
        """All chunks are scanned concurrently even when coverage found early."""
        scorer = MEIDimension()
        # 60 facts → 2 chunks, both scanned concurrently
        adapter = MockAdapter(stored_facts=[StoredFact(text=f"f{i}") for i in range(60)])
        judge = MockCoverageJudge(covered_indices={0, 1, 2})

        await scorer.score(adapter, simple_ground_truth, judge)

        assert len(judge.call_log) == 2  # both chunks scanned concurrently

    @pytest.mark.asyncio
    async def test_partial_coverage_all_chunks_scanned(self, simple_ground_truth: GroundTruth) -> None:
        """Partial coverage → all chunks are still scanned."""
        scorer = MEIDimension()
        # 60 facts → 2 chunks; judge only covers index 0
        adapter = MockAdapter(stored_facts=[StoredFact(text=f"f{i}") for i in range(60)])
        judge = MockCoverageJudge(covered_indices={0})

        await scorer.score(adapter, simple_ground_truth, judge)

        assert len(judge.call_log) == 2  # both chunks scanned


class TestMEIDetailRecords:
    """Detail records contain expected fields."""

    @pytest.mark.asyncio
    async def test_detail_records_have_required_fields(self, simple_ground_truth: GroundTruth) -> None:
        scorer = MEIDimension()
        adapter = MockAdapter(
            stored_facts=[
                StoredFact(text="User is a Software Engineer."),
                StoredFact(text="User lives in New York."),
                StoredFact(text="User is 30 years old."),
            ]
        )
        judge = MockCoverageJudge(covered_indices={0, 1, 2})

        result = await scorer.score(adapter, simple_ground_truth, judge)

        per_check_details = [d for d in result.details if not d.get("summary")]
        assert len(per_check_details) == 3
        for detail in per_check_details:
            assert "check_id" in detail
            assert "gt_key" in detail
            assert "gt_value" in detail
            assert "verdict" in detail
            assert "passed" in detail
            assert isinstance(detail["check_id"], str)
            assert detail["check_id"].startswith("mei-coverage-")
            assert detail["verdict"] in ("YES", "NO")
            assert isinstance(detail["passed"], bool)

    @pytest.mark.asyncio
    async def test_summary_detail_present(self, simple_ground_truth: GroundTruth) -> None:
        scorer = MEIDimension()
        adapter = MockAdapter(stored_facts=[StoredFact(text="fact")])
        judge = MockCoverageJudge(covered_indices={0, 1, 2})

        result = await scorer.score(adapter, simple_ground_truth, judge)

        summaries = [d for d in result.details if d.get("summary")]
        assert len(summaries) == 1
        summary = summaries[0]
        assert "total_stored_facts" in summary
        assert "total_gt_facts" in summary
        assert "covered_gt_facts" in summary
        assert "coverage" in summary
        assert "chunks_scanned" in summary

    @pytest.mark.asyncio
    async def test_multi_value_dimension_creates_one_check_per_value(self, multi_value_ground_truth: GroundTruth) -> None:
        """Multi-value dimensions expand into one coverage check per value."""
        scorer = MEIDimension()
        adapter = MockAdapter(stored_facts=[StoredFact(text="User enjoys hiking and photography.")])
        judge = MockCoverageJudge(covered_indices={0, 1})

        result = await scorer.score(adapter, multi_value_ground_truth, judge)

        per_check_details = [d for d in result.details if not d.get("summary")]
        assert result.total_checks == 2  # hiking + photography
        assert len(per_check_details) == 2
