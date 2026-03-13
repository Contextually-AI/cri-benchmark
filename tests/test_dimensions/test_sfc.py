"""Tests for the SFC (Selective Forgetting Capability) dimension scorer."""

from __future__ import annotations

from typing import Any

import pytest

from cri.models import (
    DimensionResult,
    ForgettableFact,
    GroundTruth,
    JudgmentResult,
    ProfileDimension,
    StoredFact,
    Verdict,
)
from cri.scoring.dimensions.sfc import SFCDimension

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

    def query(self, topic: str) -> list[StoredFact]:
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
    """GroundTruth with no forgettable facts and no profile dimensions."""
    return GroundTruth(
        final_profile={},
        forgettable_facts=[],
        changes=[],
        noise_examples=[],
        signal_examples=[],
        conflicts=[],
        temporal_facts=[],
        query_relevance_pairs=[],
    )


@pytest.fixture()
def forgettable_only_ground_truth() -> GroundTruth:
    """GroundTruth with forgettable facts and no profile dimensions."""
    return GroundTruth(
        final_profile={},
        forgettable_facts=[
            ForgettableFact(
                fact_id="ff-hiking",
                text="User goes hiking every weekend.",
                reason="Superseded by biking preference.",
                mentioned_at_message=10,
                should_be_absent_after=50,
            ),
            ForgettableFact(
                fact_id="ff-ios",
                text="User uses iOS.",
                reason="User switched to Android.",
                mentioned_at_message=20,
                should_be_absent_after=60,
            ),
        ],
        changes=[],
        noise_examples=[],
        signal_examples=[],
        conflicts=[],
        temporal_facts=[],
        query_relevance_pairs=[],
    )


@pytest.fixture()
def profile_only_ground_truth() -> GroundTruth:
    """GroundTruth with profile dimensions but no forgettable facts."""
    return GroundTruth(
        final_profile={
            "occupation": ProfileDimension(
                dimension_name="occupation",
                value="Software Engineer",
                query_topic="What does the user do for work?",
                category="demographics",
            ),
        },
        forgettable_facts=[],
        changes=[],
        noise_examples=[],
        signal_examples=[],
        conflicts=[],
        temporal_facts=[],
        query_relevance_pairs=[],
    )


@pytest.fixture()
def combined_ground_truth() -> GroundTruth:
    """GroundTruth with both forgettable facts and profile dimensions."""
    return GroundTruth(
        final_profile={
            "activity": ProfileDimension(
                dimension_name="activity",
                value="biking",
                query_topic="What outdoor activity does the user prefer?",
                category="interests",
            ),
        },
        forgettable_facts=[
            ForgettableFact(
                fact_id="ff-hiking",
                text="User goes hiking every weekend.",
                reason="Superseded by biking preference.",
                mentioned_at_message=10,
                should_be_absent_after=50,
            ),
        ],
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


class TestSFCDimensionAttributes:
    """Verify class-level attributes."""

    def test_name(self) -> None:
        scorer = SFCDimension()
        assert scorer.name == "SFC"

    def test_description_is_non_empty(self) -> None:
        scorer = SFCDimension()
        assert len(scorer.description) > 20

    def test_repr(self) -> None:
        scorer = SFCDimension()
        assert repr(scorer) == "SFCDimension(name='SFC')"


class TestSFCVacuousCase:
    """No forgettable facts and no profile → score 1.0."""

    @pytest.mark.asyncio
    async def test_empty_returns_one(self, empty_ground_truth: GroundTruth) -> None:
        scorer = SFCDimension()
        adapter = MockAdapter(stored_facts=[StoredFact(text="some fact")])
        judge = MockBinaryJudge()

        result = await scorer.score(adapter, empty_ground_truth, judge)

        assert isinstance(result, DimensionResult)
        assert result.dimension_name == "SFC"
        assert result.score == 1.0
        assert result.passed_checks == 0
        assert result.total_checks == 0
        assert result.details == []

    @pytest.mark.asyncio
    async def test_empty_does_not_call_judge(self, empty_ground_truth: GroundTruth) -> None:
        scorer = SFCDimension()
        adapter = MockAdapter()
        judge = MockBinaryJudge()

        await scorer.score(adapter, empty_ground_truth, judge)

        assert judge.call_log == []


class TestSFCForgetting:
    """Tests for the should-forget sub-dimension."""

    @pytest.mark.asyncio
    async def test_correctly_forgotten_high_score(
        self, forgettable_only_ground_truth: GroundTruth
    ) -> None:
        """Forgettable facts absent (NO verdict) → forget sub-score = 1.0."""
        scorer = SFCDimension()
        adapter = MockAdapter()
        # NO verdict = fact is correctly absent = pass
        judge = MockBinaryJudge(default_verdict=Verdict.NO)

        result = await scorer.score(adapter, forgettable_only_ground_truth, judge)

        # forget_score=1.0, remember_score=1.0 (no profile, defaults to 1.0)
        # composite = 0.6 * 1.0 + 0.4 * 1.0 = 1.0
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_still_retained_low_score(
        self, forgettable_only_ground_truth: GroundTruth
    ) -> None:
        """Forgettable facts still present (YES verdict) → forget sub-score = 0.0."""
        scorer = SFCDimension()
        adapter = MockAdapter(
            stored_facts=[
                StoredFact(text="User goes hiking every weekend."),
                StoredFact(text="User uses iOS."),
            ]
        )
        # YES verdict = fact is still present = failure
        judge = MockBinaryJudge(default_verdict=Verdict.YES)

        result = await scorer.score(adapter, forgettable_only_ground_truth, judge)

        # forget_score=0.0, remember_score=1.0 (no profile items → defaults 1.0)
        # composite = 0.6 * 0.0 + 0.4 * 1.0 = 0.4
        assert abs(result.score - 0.4) < 1e-9

    @pytest.mark.asyncio
    async def test_partial_forgetting(self, forgettable_only_ground_truth: GroundTruth) -> None:
        """One of two forgettable facts correctly forgotten → forget_score=0.5."""
        scorer = SFCDimension()
        adapter = MockAdapter()
        judge = MockBinaryJudge(
            default_verdict=Verdict.NO,  # default = forgotten = pass
            overrides={"sfc-forget-ff-hiking": Verdict.YES},  # still present = fail
        )

        result = await scorer.score(adapter, forgettable_only_ground_truth, judge)

        # forget_score=0.5, remember_score=1.0 (no profile)
        # composite = 0.6 * 0.5 + 0.4 * 1.0 = 0.7
        assert abs(result.score - 0.7) < 1e-9

    @pytest.mark.asyncio
    async def test_forget_check_ids_use_fact_id(
        self, forgettable_only_ground_truth: GroundTruth
    ) -> None:
        """Check IDs for forget checks follow pattern sfc-forget-{fact_id}."""
        scorer = SFCDimension()
        adapter = MockAdapter()
        judge = MockBinaryJudge(default_verdict=Verdict.NO)

        result = await scorer.score(adapter, forgettable_only_ground_truth, judge)

        check_ids = {d["check_id"] for d in result.details if not d.get("summary")}
        assert "sfc-forget-ff-hiking" in check_ids
        assert "sfc-forget-ff-ios" in check_ids


class TestSFCRetention:
    """Tests for the should-remember sub-dimension."""

    @pytest.mark.asyncio
    async def test_all_profile_facts_retained(self, profile_only_ground_truth: GroundTruth) -> None:
        """All profile facts present (YES verdict) → remember_score=1.0."""
        scorer = SFCDimension()
        adapter = MockAdapter(stored_facts=[StoredFact(text="User is a Software Engineer.")])
        judge = MockBinaryJudge(default_verdict=Verdict.YES)

        result = await scorer.score(adapter, profile_only_ground_truth, judge)

        # forget_score=1.0 (no forgettable facts → defaults 1.0), remember_score=1.0
        # composite = 0.6 * 1.0 + 0.4 * 1.0 = 1.0
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_profile_fact_missing(self, profile_only_ground_truth: GroundTruth) -> None:
        """Profile fact absent (NO verdict) → remember_score=0.0."""
        scorer = SFCDimension()
        adapter = MockAdapter()
        judge = MockBinaryJudge(default_verdict=Verdict.NO)

        result = await scorer.score(adapter, profile_only_ground_truth, judge)

        # forget_score=1.0 (no forgettable), remember_score=0.0
        # composite = 0.6 * 1.0 + 0.4 * 0.0 = 0.6
        assert abs(result.score - 0.6) < 1e-9

    @pytest.mark.asyncio
    async def test_retain_check_ids_use_dimension_name(
        self, profile_only_ground_truth: GroundTruth
    ) -> None:
        """Retention check IDs follow pattern sfc-retain-{dim_name}."""
        scorer = SFCDimension()
        adapter = MockAdapter()
        judge = MockBinaryJudge(default_verdict=Verdict.YES)

        result = await scorer.score(adapter, profile_only_ground_truth, judge)

        check_ids = {d["check_id"] for d in result.details if not d.get("summary")}
        assert "sfc-retain-occupation" in check_ids


class TestSFCWeightedComposite:
    """Weighted combination of forget and remember sub-scores."""

    @pytest.mark.asyncio
    async def test_weights_applied_correctly(self, combined_ground_truth: GroundTruth) -> None:
        """Verify composite = 0.6 * forget_score + 0.4 * remember_score."""
        scorer = SFCDimension()
        adapter = MockAdapter()
        # Forget check: YES = fact still present = fail (forget_score=0)
        # Retain check: YES = fact retained = pass (remember_score=1)
        judge = MockBinaryJudge(default_verdict=Verdict.YES)

        result = await scorer.score(adapter, combined_ground_truth, judge)

        # forget_score=0.0 (YES on forget = fail), remember_score=1.0
        # composite = 0.6 * 0.0 + 0.4 * 1.0 = 0.4
        assert abs(result.score - 0.4) < 1e-9

    @pytest.mark.asyncio
    async def test_best_of_both_worlds(self, combined_ground_truth: GroundTruth) -> None:
        """Forgotten correctly + retained correctly → score 1.0."""
        scorer = SFCDimension()
        adapter = MockAdapter()
        # Forget check expects NO for pass, retain check expects YES for pass
        judge = MockBinaryJudge(
            default_verdict=Verdict.YES,
            overrides={"sfc-forget-ff-hiking": Verdict.NO},  # correctly forgotten
        )

        result = await scorer.score(adapter, combined_ground_truth, judge)

        # forget_score=1.0, remember_score=1.0 → composite=1.0
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_summary_detail_contains_sub_scores(
        self, combined_ground_truth: GroundTruth
    ) -> None:
        scorer = SFCDimension()
        adapter = MockAdapter()
        judge = MockBinaryJudge(
            default_verdict=Verdict.YES,
            overrides={"sfc-forget-ff-hiking": Verdict.NO},
        )

        result = await scorer.score(adapter, combined_ground_truth, judge)

        summaries = [d for d in result.details if d.get("summary")]
        assert len(summaries) == 1
        summary = summaries[0]
        assert "forget_score" in summary
        assert "remember_score" in summary
        assert "composite" in summary
        assert "weight_forget" in summary
        assert "weight_remember" in summary
        assert summary["weight_forget"] == 0.6
        assert summary["weight_remember"] == 0.4

    @pytest.mark.asyncio
    async def test_per_check_details_have_sub_dimension_field(
        self, combined_ground_truth: GroundTruth
    ) -> None:
        scorer = SFCDimension()
        adapter = MockAdapter()
        judge = MockBinaryJudge(default_verdict=Verdict.NO)

        result = await scorer.score(adapter, combined_ground_truth, judge)

        non_summary = [d for d in result.details if not d.get("summary")]
        sub_dimensions = {d["sub_dimension"] for d in non_summary}
        assert "should_forget" in sub_dimensions
        assert "should_remember" in sub_dimensions
