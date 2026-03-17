"""Tests for the LNC (Long-Horizon Narrative Coherence) dimension scorer."""

from __future__ import annotations

from typing import Any

import pytest

from cri.models import (
    DimensionResult,
    GroundTruth,
    JudgmentResult,
    NarrativeArc,
    StoredFact,
    Verdict,
)
from cri.scoring.dimensions.lnc import LNCDimension

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

    def retrieve(self, topic: str) -> list[StoredFact]:
        return self._facts

    def get_events(self) -> list[StoredFact]:
        return list(self._facts)

    def ingest(self, messages: list[Any]) -> None:
        pass


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_RELOCATION_ARC = NarrativeArc(
    arc_id="arc-relocation",
    topic="Alex's relocation",
    events_in_order=[
        "Lived in San Francisco",
        "Got new job opportunities in Denver",
        "Moved to Denver",
        "Settled into Denver life",
    ],
    causal_links=[
        "new job opportunities → decision to move",
        "move to Denver → change in lifestyle",
    ],
    query_topic="Alex's living situation history",
)

_DIET_ARC = NarrativeArc(
    arc_id="arc-diet",
    topic="Alex's dietary evolution",
    events_in_order=[
        "Was an omnivore",
        "Started considering health changes",
        "Went vegetarian",
    ],
    causal_links=[
        "health awareness → dietary change",
    ],
    query_topic="Alex's diet and food preferences history",
)


@pytest.fixture()
def empty_ground_truth() -> GroundTruth:
    """GroundTruth with no narrative arcs."""
    return GroundTruth(
        final_profile={},
        narrative_arcs=[],
        changes=[],
        noise_examples=[],
        signal_examples=[],
        conflicts=[],
        temporal_facts=[],
        query_relevance_pairs=[],
    )


@pytest.fixture()
def single_arc_ground_truth() -> GroundTruth:
    """GroundTruth with a single narrative arc."""
    return GroundTruth(
        final_profile={},
        narrative_arcs=[_RELOCATION_ARC],
        changes=[],
        noise_examples=[],
        signal_examples=[],
        conflicts=[],
        temporal_facts=[],
        query_relevance_pairs=[],
    )


@pytest.fixture()
def multi_arc_ground_truth() -> GroundTruth:
    """GroundTruth with two narrative arcs."""
    return GroundTruth(
        final_profile={},
        narrative_arcs=[_RELOCATION_ARC, _DIET_ARC],
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


class TestLNCDimensionAttributes:
    """Verify class-level attributes."""

    def test_name(self) -> None:
        scorer = LNCDimension()
        assert scorer.name == "LNC"

    def test_description_is_non_empty(self) -> None:
        scorer = LNCDimension()
        assert len(scorer.description) > 20

    def test_repr(self) -> None:
        scorer = LNCDimension()
        assert repr(scorer) == "LNCDimension(name='LNC')"


class TestLNCVacuousCase:
    """No narrative arcs → score 1.0."""

    @pytest.mark.asyncio
    async def test_empty_returns_one(self, empty_ground_truth: GroundTruth) -> None:
        scorer = LNCDimension()
        adapter = MockAdapter(stored_facts=[StoredFact(text="some fact")])
        judge = MockBinaryJudge()

        result = await scorer.score(adapter, empty_ground_truth, judge)

        assert isinstance(result, DimensionResult)
        assert result.dimension_name == "LNC"
        assert result.score == 1.0
        assert result.passed_checks == 0
        assert result.total_checks == 0
        assert result.details == []

    @pytest.mark.asyncio
    async def test_empty_does_not_call_judge(self, empty_ground_truth: GroundTruth) -> None:
        scorer = LNCDimension()
        adapter = MockAdapter()
        judge = MockBinaryJudge()

        await scorer.score(adapter, empty_ground_truth, judge)

        assert judge.call_log == []


class TestLNCAllPass:
    """All three checks pass for every arc → score 1.0."""

    @pytest.mark.asyncio
    async def test_perfect_score_single_arc(self, single_arc_ground_truth: GroundTruth) -> None:
        scorer = LNCDimension()
        adapter = MockAdapter(
            stored_facts=[
                StoredFact(text="Alex lived in San Francisco"),
                StoredFact(text="Got new job in Denver"),
                StoredFact(text="Moved to Denver"),
            ]
        )
        # Sequence=YES (pass), Causality=YES (pass), Contradiction=NO (pass)
        judge = MockBinaryJudge(
            default_verdict=Verdict.YES,
            overrides={"lnc-contra-arc-relocation": Verdict.NO},
        )

        result = await scorer.score(adapter, single_arc_ground_truth, judge)

        assert result.score == 1.0
        assert result.passed_checks == 3
        assert result.total_checks == 3

    @pytest.mark.asyncio
    async def test_perfect_score_multi_arc(self, multi_arc_ground_truth: GroundTruth) -> None:
        scorer = LNCDimension()
        adapter = MockAdapter(stored_facts=[StoredFact(text="narrative facts")])
        judge = MockBinaryJudge(
            default_verdict=Verdict.YES,
            overrides={
                "lnc-contra-arc-relocation": Verdict.NO,
                "lnc-contra-arc-diet": Verdict.NO,
            },
        )

        result = await scorer.score(adapter, multi_arc_ground_truth, judge)

        assert result.score == 1.0
        assert result.passed_checks == 6
        assert result.total_checks == 6

    @pytest.mark.asyncio
    async def test_three_judge_calls_per_arc(self, single_arc_ground_truth: GroundTruth) -> None:
        scorer = LNCDimension()
        adapter = MockAdapter()
        judge = MockBinaryJudge(
            default_verdict=Verdict.YES,
            overrides={"lnc-contra-arc-relocation": Verdict.NO},
        )

        await scorer.score(adapter, single_arc_ground_truth, judge)

        assert len(judge.call_log) == 3
        check_ids = [c["check_id"] for c in judge.call_log]
        assert "lnc-seq-arc-relocation" in check_ids
        assert "lnc-caus-arc-relocation" in check_ids
        assert "lnc-contra-arc-relocation" in check_ids


class TestLNCPartialFail:
    """Some checks fail → partial score."""

    @pytest.mark.asyncio
    async def test_sequence_fails(self, single_arc_ground_truth: GroundTruth) -> None:
        """Sequence wrong, causality ok, no contradictions → arc_score = 2/3."""
        scorer = LNCDimension()
        adapter = MockAdapter()
        judge = MockBinaryJudge(
            default_verdict=Verdict.YES,
            overrides={
                "lnc-seq-arc-relocation": Verdict.NO,  # sequence wrong
                "lnc-contra-arc-relocation": Verdict.NO,  # no contradictions (pass)
            },
        )

        result = await scorer.score(adapter, single_arc_ground_truth, judge)

        expected = 2.0 / 3.0
        assert abs(result.score - expected) < 1e-3
        assert result.passed_checks == 2
        assert result.total_checks == 3

    @pytest.mark.asyncio
    async def test_contradiction_found(self, single_arc_ground_truth: GroundTruth) -> None:
        """Contradictions found (YES) → contradiction check fails, arc_score = 2/3."""
        scorer = LNCDimension()
        adapter = MockAdapter()
        # Sequence=YES (pass), Causality=YES (pass), Contradiction=YES (fail!)
        judge = MockBinaryJudge(default_verdict=Verdict.YES)

        result = await scorer.score(adapter, single_arc_ground_truth, judge)

        expected = 2.0 / 3.0
        assert abs(result.score - expected) < 1e-3
        assert result.passed_checks == 2

    @pytest.mark.asyncio
    async def test_mixed_arcs(self, multi_arc_ground_truth: GroundTruth) -> None:
        """One arc perfect, one arc all-fail → average = 0.5."""
        scorer = LNCDimension()
        adapter = MockAdapter()
        judge = MockBinaryJudge(
            default_verdict=Verdict.NO,  # all fail by default
            overrides={
                # Arc 1 (relocation): all pass
                "lnc-seq-arc-relocation": Verdict.YES,
                "lnc-caus-arc-relocation": Verdict.YES,
                # lnc-contra-arc-relocation: NO = no contradictions = pass (default)
                # Arc 2 (diet): all fail
                # seq: NO = fail, caus: NO = fail, contra: NO = pass (1/3)
            },
        )

        result = await scorer.score(adapter, multi_arc_ground_truth, judge)

        # Arc 1: 3/3 = 1.0, Arc 2: 1/3 (only contradiction passes)
        # Average = (1.0 + 1/3) / 2 = 2/3
        expected = (1.0 + 1.0 / 3.0) / 2.0
        assert abs(result.score - expected) < 1e-3


class TestLNCAllFail:
    """All checks fail → minimum score."""

    @pytest.mark.asyncio
    async def test_all_fail_single_arc(self, single_arc_ground_truth: GroundTruth) -> None:
        """Sequence=NO, Causality=NO, Contradiction=YES → arc_score = 0.0."""
        scorer = LNCDimension()
        adapter = MockAdapter()
        # seq=NO (fail), caus=NO (fail), contra=YES (fail — contradictions found)
        judge = MockBinaryJudge(
            default_verdict=Verdict.NO,
            overrides={"lnc-contra-arc-relocation": Verdict.YES},
        )

        result = await scorer.score(adapter, single_arc_ground_truth, judge)

        assert result.score == 0.0
        assert result.passed_checks == 0
        assert result.total_checks == 3


class TestLNCDetails:
    """Verify the structure of per-arc detail records."""

    @pytest.mark.asyncio
    async def test_detail_fields(self, single_arc_ground_truth: GroundTruth) -> None:
        scorer = LNCDimension()
        adapter = MockAdapter(stored_facts=[StoredFact(text="some fact")])
        judge = MockBinaryJudge(
            default_verdict=Verdict.YES,
            overrides={"lnc-contra-arc-relocation": Verdict.NO},
        )

        result = await scorer.score(adapter, single_arc_ground_truth, judge)

        assert len(result.details) == 1
        detail = result.details[0]
        assert detail["arc_id"] == "arc-relocation"
        assert detail["topic"] == "Alex's relocation"
        assert detail["stored_facts_count"] == 1
        assert detail["sequence_verdict"] == "YES"
        assert detail["sequence_passed"] is True
        assert detail["causality_verdict"] == "YES"
        assert detail["causality_passed"] is True
        assert detail["contradiction_verdict"] == "NO"
        assert detail["contradiction_passed"] is True
        assert detail["arc_score"] == 1.0

    @pytest.mark.asyncio
    async def test_multi_arc_details(self, multi_arc_ground_truth: GroundTruth) -> None:
        scorer = LNCDimension()
        adapter = MockAdapter()
        judge = MockBinaryJudge(
            default_verdict=Verdict.YES,
            overrides={
                "lnc-contra-arc-relocation": Verdict.NO,
                "lnc-contra-arc-diet": Verdict.NO,
            },
        )

        result = await scorer.score(adapter, multi_arc_ground_truth, judge)

        assert len(result.details) == 2
        arc_ids = [d["arc_id"] for d in result.details]
        assert "arc-relocation" in arc_ids
        assert "arc-diet" in arc_ids
