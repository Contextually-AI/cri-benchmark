"""Tests for the MEI (Memory Efficiency Index) dimension scorer."""

from __future__ import annotations

from typing import Any

import pytest

from cri.models import (
    DimensionResult,
    GroundTruth,
    JudgmentResult,
    ProfileDimension,
    StoredFact,
    Verdict,
)
from cri.scoring.dimensions.mei import MEIDimension

# ---------------------------------------------------------------------------
# Test helpers — lightweight mocks
# ---------------------------------------------------------------------------


class MockBinaryJudge:
    """Mock binary judge with a synchronous ``judge()`` method.

    By default every check gets ``Verdict.YES``.  Override specific
    check_ids via the *overrides* dict.
    """

    def __init__(
        self,
        default_verdict: Verdict = Verdict.YES,
        overrides: dict[str, Verdict] | None = None,
    ) -> None:
        self.default_verdict = default_verdict
        self.overrides = overrides or {}
        self.call_log: list[dict[str, Any]] = []

    def judge(self, check_id: str, prompt: str) -> JudgmentResult:
        verdict = self.overrides.get(check_id, self.default_verdict)
        self.call_log.append({"check_id": check_id, "prompt": prompt})
        return JudgmentResult(
            check_id=check_id,
            verdict=verdict,
            votes=[verdict, verdict, verdict],
            unanimous=True,
            prompt=prompt,
            raw_responses=[verdict.value] * 3,
        )


class MockAdapter:
    """Mock adapter with a fixed list of stored facts."""

    def __init__(self, stored_facts: list[StoredFact] | None = None) -> None:
        self._facts: list[StoredFact] = stored_facts or []
        self.queries: list[str] = []

    def query(self, topic: str) -> list[StoredFact]:
        self.queries.append(topic)
        return self._facts

    def get_all_facts(self) -> list[StoredFact]:
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
    """Empty profile → vacuously correct score of 1.0."""

    @pytest.mark.asyncio
    async def test_empty_profile_returns_one(self, empty_ground_truth: GroundTruth) -> None:
        scorer = MEIDimension()
        adapter = MockAdapter(stored_facts=[StoredFact(text="some fact")])
        judge = MockBinaryJudge()

        result = await scorer.score(adapter, empty_ground_truth, judge)

        assert isinstance(result, DimensionResult)
        assert result.dimension_name == "MEI"
        assert result.score == 1.0
        assert result.passed_checks == 0
        assert result.total_checks == 0
        assert result.details == []

    @pytest.mark.asyncio
    async def test_empty_profile_does_not_call_judge(self, empty_ground_truth: GroundTruth) -> None:
        scorer = MEIDimension()
        adapter = MockAdapter()
        judge = MockBinaryJudge()

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
        judge = MockBinaryJudge(default_verdict=Verdict.YES)

        result = await scorer.score(adapter, simple_ground_truth, judge)

        assert result.dimension_name == "MEI"
        assert result.score == 1.0
        assert result.passed_checks == 3
        assert result.total_checks == 3

    @pytest.mark.asyncio
    async def test_all_covered_more_stored(self, simple_ground_truth: GroundTruth) -> None:
        """3 gt facts covered, 6 stored facts → efficiency=0.5, coverage=1.0, MEI=harmonic."""
        scorer = MEIDimension()
        # 6 stored facts, all 3 gt facts covered
        adapter = MockAdapter(stored_facts=[StoredFact(text=f"fact-{i}") for i in range(6)])
        judge = MockBinaryJudge(default_verdict=Verdict.YES)

        result = await scorer.score(adapter, simple_ground_truth, judge)

        # coverage=1.0, efficiency=3/6=0.5 → harmonic = 2*1.0*0.5/(1.0+0.5) = 2/3
        expected = round(2 * 1.0 * 0.5 / (1.0 + 0.5), 4)
        assert result.score == expected
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
        # Fail the "age" check (check index 2 = mei-coverage-2, but order depends on dict)
        # Use NO verdict for exactly one check
        judge = MockBinaryJudge(
            default_verdict=Verdict.YES,
            overrides={"mei-coverage-2": Verdict.NO},
        )

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
        judge = MockBinaryJudge()

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
        # 3 gt facts, 3 stored. Judge passes all.
        adapter = MockAdapter(stored_facts=[StoredFact(text=f"f{i}") for i in range(3)])
        judge = MockBinaryJudge(default_verdict=Verdict.YES)

        result = await scorer.score(adapter, simple_ground_truth, judge)

        # coverage=1.0, efficiency=1.0 → harmonic=1.0
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_harmonic_mean_vs_arithmetic(self, simple_ground_truth: GroundTruth) -> None:
        """Harmonic mean is lower than arithmetic mean for unequal values."""
        scorer = MEIDimension()
        # 3 gt facts, 9 stored facts. All 3 covered.
        # coverage=1.0, efficiency=3/9=0.333 → harmonic ≈ 0.5, arithmetic = 0.667
        adapter = MockAdapter(stored_facts=[StoredFact(text=f"f{i}") for i in range(9)])
        judge = MockBinaryJudge(default_verdict=Verdict.YES)

        result = await scorer.score(adapter, simple_ground_truth, judge)

        efficiency = 3.0 / 9.0
        coverage = 1.0
        expected_harmonic = round(2.0 * efficiency * coverage / (efficiency + coverage), 4)
        expected_arithmetic = (efficiency + coverage) / 2.0
        assert result.score == expected_harmonic
        assert result.score < expected_arithmetic


class TestMEIDetailRecords:
    """Detail records contain expected fields."""

    @pytest.mark.asyncio
    async def test_detail_records_have_required_fields(
        self, simple_ground_truth: GroundTruth
    ) -> None:
        scorer = MEIDimension()
        adapter = MockAdapter(
            stored_facts=[
                StoredFact(text="User is a Software Engineer."),
                StoredFact(text="User lives in New York."),
                StoredFact(text="User is 30 years old."),
            ]
        )
        judge = MockBinaryJudge(default_verdict=Verdict.YES)

        result = await scorer.score(adapter, simple_ground_truth, judge)

        # Last detail is summary; preceding ones are per-check
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
        judge = MockBinaryJudge(default_verdict=Verdict.YES)

        result = await scorer.score(adapter, simple_ground_truth, judge)

        summaries = [d for d in result.details if d.get("summary")]
        assert len(summaries) == 1
        summary = summaries[0]
        assert "total_stored_facts" in summary
        assert "total_gt_facts" in summary
        assert "covered_gt_facts" in summary
        assert "coverage" in summary
        assert "efficiency" in summary
        assert "mei" in summary

    @pytest.mark.asyncio
    async def test_multi_value_dimension_creates_one_check_per_value(
        self, multi_value_ground_truth: GroundTruth
    ) -> None:
        """Multi-value dimensions expand into one coverage check per value."""
        scorer = MEIDimension()
        adapter = MockAdapter(stored_facts=[StoredFact(text="User enjoys hiking and photography.")])
        judge = MockBinaryJudge(default_verdict=Verdict.YES)

        result = await scorer.score(adapter, multi_value_ground_truth, judge)

        per_check_details = [d for d in result.details if not d.get("summary")]
        assert result.total_checks == 2  # hiking + photography
        assert len(per_check_details) == 2
