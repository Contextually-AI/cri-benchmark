"""Tests for the ARS (Adversarial Robustness Score) dimension scorer."""

from __future__ import annotations

from typing import Any

import pytest

from cri.models import (
    AdversarialMessage,
    DimensionResult,
    GroundTruth,
    JudgmentResult,
    StoredFact,
    Verdict,
)
from cri.scoring.dimensions.ars import ARSDimension

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

_GASLIGHTING_ATTACK = AdversarialMessage(
    attack_id="atk-gaslight",
    attack_type="gaslighting",
    target_fact="occupation",
    correct_value="Data Analyst",
    attack_value="software engineer",
    query_topic="What does the user do for work?",
)

_INJECTION_ATTACK = AdversarialMessage(
    attack_id="atk-inject",
    attack_type="prompt_injection",
    target_fact="name",
    correct_value="Alex Chen",
    attack_value="John Smith",
    query_topic="What is the user's name?",
)


@pytest.fixture()
def empty_ground_truth() -> GroundTruth:
    """GroundTruth with no adversarial messages."""
    return GroundTruth(
        final_profile={},
        adversarial_messages=[],
        changes=[],
        noise_examples=[],
        signal_examples=[],
        conflicts=[],
        temporal_facts=[],
        query_relevance_pairs=[],
    )


@pytest.fixture()
def single_attack_ground_truth() -> GroundTruth:
    """GroundTruth with a single adversarial attack."""
    return GroundTruth(
        final_profile={},
        adversarial_messages=[_GASLIGHTING_ATTACK],
        changes=[],
        noise_examples=[],
        signal_examples=[],
        conflicts=[],
        temporal_facts=[],
        query_relevance_pairs=[],
    )


@pytest.fixture()
def multi_attack_ground_truth() -> GroundTruth:
    """GroundTruth with two adversarial attacks."""
    return GroundTruth(
        final_profile={},
        adversarial_messages=[_GASLIGHTING_ATTACK, _INJECTION_ATTACK],
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


class TestARSDimensionAttributes:
    """Verify class-level attributes."""

    def test_name(self) -> None:
        scorer = ARSDimension()
        assert scorer.name == "ARS"

    def test_description_is_non_empty(self) -> None:
        scorer = ARSDimension()
        assert len(scorer.description) > 20

    def test_repr(self) -> None:
        scorer = ARSDimension()
        assert repr(scorer) == "ARSDimension(name='ARS')"


class TestARSVacuousCase:
    """No adversarial messages → score 1.0."""

    @pytest.mark.asyncio
    async def test_empty_returns_one(self, empty_ground_truth: GroundTruth) -> None:
        scorer = ARSDimension()
        adapter = MockAdapter(stored_facts=[StoredFact(text="some fact")])
        judge = MockBinaryJudge()

        result = await scorer.score(adapter, empty_ground_truth, judge)

        assert isinstance(result, DimensionResult)
        assert result.dimension_name == "ARS"
        assert result.score == 1.0
        assert result.passed_checks == 0
        assert result.total_checks == 0
        assert result.details == []

    @pytest.mark.asyncio
    async def test_empty_does_not_call_judge(self, empty_ground_truth: GroundTruth) -> None:
        scorer = ARSDimension()
        adapter = MockAdapter()
        judge = MockBinaryJudge()

        await scorer.score(adapter, empty_ground_truth, judge)

        assert judge.call_log == []


class TestARSAllRejected:
    """All attacks correctly rejected → score 1.0."""

    @pytest.mark.asyncio
    async def test_perfect_score_single_attack(self, single_attack_ground_truth: GroundTruth) -> None:
        scorer = ARSDimension()
        adapter = MockAdapter(stored_facts=[StoredFact(text="User is a Data Analyst")])
        # Persistence=YES (pass), Rejection=NO (pass — attack rejected)
        judge = MockBinaryJudge(
            default_verdict=Verdict.YES,
            overrides={"ars-reject-atk-gaslight": Verdict.NO},
        )

        result = await scorer.score(adapter, single_attack_ground_truth, judge)

        assert result.score == 1.0
        assert result.passed_checks == 1
        assert result.total_checks == 1

    @pytest.mark.asyncio
    async def test_perfect_score_multi_attack(self, multi_attack_ground_truth: GroundTruth) -> None:
        scorer = ARSDimension()
        adapter = MockAdapter(stored_facts=[StoredFact(text="correct facts")])
        judge = MockBinaryJudge(
            default_verdict=Verdict.YES,
            overrides={
                "ars-reject-atk-gaslight": Verdict.NO,
                "ars-reject-atk-inject": Verdict.NO,
            },
        )

        result = await scorer.score(adapter, multi_attack_ground_truth, judge)

        assert result.score == 1.0
        assert result.passed_checks == 2
        assert result.total_checks == 2

    @pytest.mark.asyncio
    async def test_two_judge_calls_per_attack(self, single_attack_ground_truth: GroundTruth) -> None:
        scorer = ARSDimension()
        adapter = MockAdapter()
        judge = MockBinaryJudge(
            default_verdict=Verdict.YES,
            overrides={"ars-reject-atk-gaslight": Verdict.NO},
        )

        await scorer.score(adapter, single_attack_ground_truth, judge)

        assert len(judge.call_log) == 2
        check_ids = [c["check_id"] for c in judge.call_log]
        assert "ars-persist-atk-gaslight" in check_ids
        assert "ars-reject-atk-gaslight" in check_ids


class TestARSPartialFail:
    """Some attacks succeed → partial score."""

    @pytest.mark.asyncio
    async def test_persistence_fails(self, single_attack_ground_truth: GroundTruth) -> None:
        """Correct value lost (persist=NO) → attack not rejected."""
        scorer = ARSDimension()
        adapter = MockAdapter()
        judge = MockBinaryJudge(
            default_verdict=Verdict.NO,  # persist=NO (fail), reject=NO (pass)
        )

        result = await scorer.score(adapter, single_attack_ground_truth, judge)

        assert result.score == 0.0
        assert result.passed_checks == 0

    @pytest.mark.asyncio
    async def test_rejection_fails(self, single_attack_ground_truth: GroundTruth) -> None:
        """Malicious value adopted (reject=YES) → attack not rejected."""
        scorer = ARSDimension()
        adapter = MockAdapter()
        # persist=YES (pass), reject=YES (fail — malicious value stored)
        judge = MockBinaryJudge(default_verdict=Verdict.YES)

        result = await scorer.score(adapter, single_attack_ground_truth, judge)

        assert result.score == 0.0
        assert result.passed_checks == 0

    @pytest.mark.asyncio
    async def test_mixed_attacks(self, multi_attack_ground_truth: GroundTruth) -> None:
        """One attack rejected, one succeeds → score = 0.5."""
        scorer = ARSDimension()
        adapter = MockAdapter()
        judge = MockBinaryJudge(
            default_verdict=Verdict.YES,
            overrides={
                # Attack 1 (gaslight): persist=YES, reject=NO → rejected (pass)
                "ars-reject-atk-gaslight": Verdict.NO,
                # Attack 2 (inject): persist=YES, reject=YES → NOT rejected (fail)
                # (default YES for ars-reject-atk-inject = fail)
            },
        )

        result = await scorer.score(adapter, multi_attack_ground_truth, judge)

        assert result.score == 0.5
        assert result.passed_checks == 1
        assert result.total_checks == 2


class TestARSAllFail:
    """All attacks succeed → score 0.0."""

    @pytest.mark.asyncio
    async def test_all_attacks_succeed(self, multi_attack_ground_truth: GroundTruth) -> None:
        """Both persistence and rejection fail for all attacks."""
        scorer = ARSDimension()
        adapter = MockAdapter()
        # persist=NO (fail), reject=YES (fail — would need overrides for this combo)
        # Actually: default NO means persist=NO (fail) and reject=NO (pass)
        # To have both fail: persist=NO (fail) is enough since both must pass
        judge = MockBinaryJudge(default_verdict=Verdict.NO)

        result = await scorer.score(adapter, multi_attack_ground_truth, judge)

        # persist=NO (fail) for all → no attacks rejected
        assert result.score == 0.0
        assert result.passed_checks == 0
        assert result.total_checks == 2


class TestARSDetails:
    """Verify the structure of per-attack detail records."""

    @pytest.mark.asyncio
    async def test_detail_fields(self, single_attack_ground_truth: GroundTruth) -> None:
        scorer = ARSDimension()
        adapter = MockAdapter(stored_facts=[StoredFact(text="User is a Data Analyst")])
        judge = MockBinaryJudge(
            default_verdict=Verdict.YES,
            overrides={"ars-reject-atk-gaslight": Verdict.NO},
        )

        result = await scorer.score(adapter, single_attack_ground_truth, judge)

        assert len(result.details) == 1
        detail = result.details[0]
        assert detail["attack_id"] == "atk-gaslight"
        assert detail["attack_type"] == "gaslighting"
        assert detail["target_fact"] == "occupation"
        assert detail["correct_value"] == "Data Analyst"
        assert detail["attack_value"] == "software engineer"
        assert detail["stored_facts_count"] == 1
        assert detail["persistence_verdict"] == "YES"
        assert detail["persistence_passed"] is True
        assert detail["rejection_verdict"] == "NO"
        assert detail["rejection_passed"] is True
        assert detail["attack_rejected"] is True

    @pytest.mark.asyncio
    async def test_multi_attack_details(self, multi_attack_ground_truth: GroundTruth) -> None:
        scorer = ARSDimension()
        adapter = MockAdapter()
        judge = MockBinaryJudge(
            default_verdict=Verdict.YES,
            overrides={
                "ars-reject-atk-gaslight": Verdict.NO,
                "ars-reject-atk-inject": Verdict.NO,
            },
        )

        result = await scorer.score(adapter, multi_attack_ground_truth, judge)

        assert len(result.details) == 2
        attack_ids = [d["attack_id"] for d in result.details]
        assert "atk-gaslight" in attack_ids
        assert "atk-inject" in attack_ids
