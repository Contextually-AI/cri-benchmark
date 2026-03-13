"""Tests for the PAS dimension scorer.

Tests both the new :class:`ProfileAccuracyScore` (MetricDimension-based)
and the
"""

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
from cri.scoring.dimensions.pas import ProfileAccuracyScore

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


class MockQueryAdapter:
    """Mock adapter that returns predetermined facts per query topic.

    Facts are configured via the *facts_by_topic* mapping.  Topics not
    present in the mapping return an empty list.
    """

    def __init__(self, facts_by_topic: dict[str, list[StoredFact]] | None = None) -> None:
        self.facts_by_topic = facts_by_topic or {}
        self.queries: list[str] = []

    def query(self, topic: str) -> list[StoredFact]:
        self.queries.append(topic)
        return self.facts_by_topic.get(topic, [])

    def get_all_facts(self) -> list[StoredFact]:
        all_facts: list[StoredFact] = []
        for facts in self.facts_by_topic.values():
            all_facts.extend(facts)
        return all_facts

    def ingest(self, messages: list[Any]) -> None:
        pass


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


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
    """GroundTruth with a multi-value profile dimension (hobbies)."""
    return GroundTruth(
        final_profile={
            "hobbies": ProfileDimension(
                dimension_name="hobbies",
                value=["hiking", "photography", "cooking"],
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


@pytest.fixture()
def mixed_ground_truth() -> GroundTruth:
    """GroundTruth with both single-value and multi-value dimensions."""
    return GroundTruth(
        final_profile={
            "occupation": ProfileDimension(
                dimension_name="occupation",
                value="Software Engineer",
                query_topic="What does the user do for work?",
                category="demographics",
            ),
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


@pytest.fixture()
def empty_ground_truth() -> GroundTruth:
    """GroundTruth with an empty profile."""
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
def default_adapter() -> MockQueryAdapter:
    """An adapter that returns some facts for every query."""
    return MockQueryAdapter(
        facts_by_topic={
            "What does the user do for work?": [
                StoredFact(text="User is a Software Engineer."),
                StoredFact(text="User works at a tech company."),
            ],
            "Where does the user live?": [
                StoredFact(text="User lives in New York."),
            ],
            "How old is the user?": [
                StoredFact(text="User is 30 years old."),
            ],
            "What are the user's hobbies?": [
                StoredFact(text="User enjoys hiking and photography."),
                StoredFact(text="User likes cooking Italian food."),
            ],
        }
    )


# ---------------------------------------------------------------------------
# Tests — ProfileAccuracyScore (new MetricDimension-based scorer)
# ---------------------------------------------------------------------------


class TestProfileAccuracyScore:
    """Tests for the new binary-verdict PAS scorer."""

    def test_class_attributes(self) -> None:
        """ProfileAccuracyScore has correct name and description."""
        scorer = ProfileAccuracyScore()
        assert scorer.name == "PAS"
        assert "persona" in scorer.description.lower() or "factual" in scorer.description.lower()
        assert len(scorer.description) > 20

    def test_repr(self) -> None:
        scorer = ProfileAccuracyScore()
        assert repr(scorer) == "ProfileAccuracyScore(name='PAS')"

    @pytest.mark.asyncio
    async def test_all_checks_pass(
        self, simple_ground_truth: GroundTruth, default_adapter: MockQueryAdapter
    ) -> None:
        """When the judge says YES for every check, score should be 1.0."""
        scorer = ProfileAccuracyScore()
        judge = MockBinaryJudge(default_verdict=Verdict.YES)

        result = await scorer.score(default_adapter, simple_ground_truth, judge)

        assert isinstance(result, DimensionResult)
        assert result.dimension_name == "PAS"
        assert result.score == 1.0
        assert result.passed_checks == 3
        assert result.total_checks == 3
        assert len(result.details) == 3

    @pytest.mark.asyncio
    async def test_all_checks_fail(
        self, simple_ground_truth: GroundTruth, default_adapter: MockQueryAdapter
    ) -> None:
        """When the judge says NO for every check, score should be 0.0."""
        scorer = ProfileAccuracyScore()
        judge = MockBinaryJudge(default_verdict=Verdict.NO)

        result = await scorer.score(default_adapter, simple_ground_truth, judge)

        assert result.score == 0.0
        assert result.passed_checks == 0
        assert result.total_checks == 3

    @pytest.mark.asyncio
    async def test_mixed_results(
        self, simple_ground_truth: GroundTruth, default_adapter: MockQueryAdapter
    ) -> None:
        """Mixed verdicts produce a proportional score."""
        scorer = ProfileAccuracyScore()
        judge = MockBinaryJudge(
            default_verdict=Verdict.YES,
            overrides={"pas-age": Verdict.NO},
        )

        result = await scorer.score(default_adapter, simple_ground_truth, judge)

        assert result.passed_checks == 2
        assert result.total_checks == 3
        assert abs(result.score - 2.0 / 3.0) < 1e-9

    @pytest.mark.asyncio
    async def test_multi_value_dimension(
        self, multi_value_ground_truth: GroundTruth, default_adapter: MockQueryAdapter
    ) -> None:
        """Multi-value dimensions create one check per value."""
        scorer = ProfileAccuracyScore()
        judge = MockBinaryJudge(default_verdict=Verdict.YES)

        result = await scorer.score(default_adapter, multi_value_ground_truth, judge)

        assert result.total_checks == 3  # hiking, photography, cooking
        assert result.passed_checks == 3
        assert result.score == 1.0

        # Verify check IDs follow multi-value format
        check_ids = [d["check_id"] for d in result.details]
        assert "pas-hobbies-0" in check_ids
        assert "pas-hobbies-1" in check_ids
        assert "pas-hobbies-2" in check_ids

    @pytest.mark.asyncio
    async def test_multi_value_partial_pass(
        self, multi_value_ground_truth: GroundTruth, default_adapter: MockQueryAdapter
    ) -> None:
        """Multi-value dimension with some values failing."""
        scorer = ProfileAccuracyScore()
        judge = MockBinaryJudge(
            default_verdict=Verdict.YES,
            overrides={"pas-hobbies-2": Verdict.NO},
        )

        result = await scorer.score(default_adapter, multi_value_ground_truth, judge)

        assert result.passed_checks == 2
        assert result.total_checks == 3
        assert abs(result.score - 2.0 / 3.0) < 1e-9

    @pytest.mark.asyncio
    async def test_empty_profile(
        self, empty_ground_truth: GroundTruth, default_adapter: MockQueryAdapter
    ) -> None:
        """Empty profile produces score 0.0 with 0 checks."""
        scorer = ProfileAccuracyScore()
        judge = MockBinaryJudge()

        result = await scorer.score(default_adapter, empty_ground_truth, judge)

        assert result.score == 0.0
        assert result.passed_checks == 0
        assert result.total_checks == 0
        assert result.details == []

    @pytest.mark.asyncio
    async def test_check_id_format_single_value(
        self, simple_ground_truth: GroundTruth, default_adapter: MockQueryAdapter
    ) -> None:
        """Single-value dimensions produce check IDs like 'pas-{dim_name}'."""
        scorer = ProfileAccuracyScore()
        judge = MockBinaryJudge()

        result = await scorer.score(default_adapter, simple_ground_truth, judge)

        check_ids = {d["check_id"] for d in result.details}
        assert "pas-occupation" in check_ids
        assert "pas-location" in check_ids
        assert "pas-age" in check_ids

    @pytest.mark.asyncio
    async def test_mixed_single_and_multi_value(
        self, mixed_ground_truth: GroundTruth, default_adapter: MockQueryAdapter
    ) -> None:
        """Mixed single and multi-value dimensions produce correct check IDs."""
        scorer = ProfileAccuracyScore()
        judge = MockBinaryJudge()

        result = await scorer.score(default_adapter, mixed_ground_truth, judge)

        # 1 single-value + 2 multi-value = 3 total
        assert result.total_checks == 3
        check_ids = {d["check_id"] for d in result.details}
        assert "pas-occupation" in check_ids
        assert "pas-hobbies-0" in check_ids
        assert "pas-hobbies-1" in check_ids

    @pytest.mark.asyncio
    async def test_details_contain_expected_fields(
        self, simple_ground_truth: GroundTruth, default_adapter: MockQueryAdapter
    ) -> None:
        """Each detail dict contains all required fields."""
        scorer = ProfileAccuracyScore()
        judge = MockBinaryJudge(default_verdict=Verdict.YES)

        result = await scorer.score(default_adapter, simple_ground_truth, judge)

        for detail in result.details:
            assert "check_id" in detail
            assert "dimension_name" in detail
            assert "expected_value" in detail
            assert "verdict" in detail
            assert "passed" in detail
            assert isinstance(detail["check_id"], str)
            assert isinstance(detail["dimension_name"], str)
            assert isinstance(detail["expected_value"], str)
            assert detail["verdict"] in ("YES", "NO")
            assert isinstance(detail["passed"], bool)

    @pytest.mark.asyncio
    async def test_details_values_correct_on_pass(
        self, simple_ground_truth: GroundTruth, default_adapter: MockQueryAdapter
    ) -> None:
        """Passing checks have verdict=YES and passed=True."""
        scorer = ProfileAccuracyScore()
        judge = MockBinaryJudge(default_verdict=Verdict.YES)

        result = await scorer.score(default_adapter, simple_ground_truth, judge)

        for detail in result.details:
            assert detail["verdict"] == "YES"
            assert detail["passed"] is True

    @pytest.mark.asyncio
    async def test_details_values_correct_on_fail(
        self, simple_ground_truth: GroundTruth, default_adapter: MockQueryAdapter
    ) -> None:
        """Failing checks have verdict=NO and passed=False."""
        scorer = ProfileAccuracyScore()
        judge = MockBinaryJudge(default_verdict=Verdict.NO)

        result = await scorer.score(default_adapter, simple_ground_truth, judge)

        for detail in result.details:
            assert detail["verdict"] == "NO"
            assert detail["passed"] is False

    @pytest.mark.asyncio
    async def test_adapter_receives_correct_query_topics(
        self, simple_ground_truth: GroundTruth
    ) -> None:
        """The adapter's query() is called with the correct topic strings."""
        adapter = MockQueryAdapter()
        scorer = ProfileAccuracyScore()
        judge = MockBinaryJudge()

        await scorer.score(adapter, simple_ground_truth, judge)

        expected_topics = {
            "What does the user do for work?",
            "Where does the user live?",
            "How old is the user?",
        }
        assert set(adapter.queries) == expected_topics

    @pytest.mark.asyncio
    async def test_judge_receives_correct_prompts(self, default_adapter: MockQueryAdapter) -> None:
        """The judge receives prompts generated by pas_check with correct args."""
        gt = GroundTruth(
            final_profile={
                "occupation": ProfileDimension(
                    dimension_name="occupation",
                    value="Engineer",
                    query_topic="What does the user do for work?",
                ),
            },
            changes=[],
            noise_examples=[],
            signal_examples=[],
            conflicts=[],
            temporal_facts=[],
            query_relevance_pairs=[],
        )
        scorer = ProfileAccuracyScore()
        judge = MockBinaryJudge()

        await scorer.score(default_adapter, gt, judge)

        assert len(judge.call_log) == 1
        call = judge.call_log[0]
        assert call["check_id"] == "pas-occupation"
        # The prompt should contain the dimension name and expected value
        assert "occupation" in call["prompt"]
        assert "Engineer" in call["prompt"]

    @pytest.mark.asyncio
    async def test_empty_facts_from_adapter(self, simple_ground_truth: GroundTruth) -> None:
        """When adapter returns no facts, checks still run (judge decides)."""
        adapter = MockQueryAdapter()  # returns empty lists
        scorer = ProfileAccuracyScore()
        judge = MockBinaryJudge(default_verdict=Verdict.NO)

        result = await scorer.score(adapter, simple_ground_truth, judge)

        assert result.total_checks == 3
        assert result.passed_checks == 0
        assert result.score == 0.0

    @pytest.mark.asyncio
    async def test_multi_value_expected_values_in_details(
        self, multi_value_ground_truth: GroundTruth, default_adapter: MockQueryAdapter
    ) -> None:
        """Details for multi-value dimensions include each individual value."""
        scorer = ProfileAccuracyScore()
        judge = MockBinaryJudge()

        result = await scorer.score(default_adapter, multi_value_ground_truth, judge)

        expected_values = {d["expected_value"] for d in result.details}
        assert "hiking" in expected_values
        assert "photography" in expected_values
        assert "cooking" in expected_values
