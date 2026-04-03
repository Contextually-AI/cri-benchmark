"""Tests for the TC (Temporal Coherence) dimension scorer.

Tests both the new :class:`TCDimension` (MetricDimension-based) and the


Test strategy follows the standard CRI pattern:
1. Create mock adapter returning known facts
2. Create sample ground truth with temporal facts
3. Use mock judge with predetermined verdicts
4. Verify score matches expected value
5. Verify DimensionResult structure
"""

from __future__ import annotations

import asyncio
from typing import Any

from cri.models import (
    DimensionResult,
    GroundTruth,
    JudgmentResult,
    StoredFact,
    TemporalFact,
    Verdict,
)
from cri.scoring.dimensions.tc import TCDimension

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


_DEFAULT_TC_FACTS = [StoredFact(text="placeholder fact for testing")]


class MockTCAdapter:
    """Mock adapter that returns predetermined facts per query topic.

    Facts are configured via the *facts_by_topic* mapping. Topics not
    present in the mapping return a default non-empty fact list so that
    the judge evaluation path is exercised.
    """

    def __init__(self, facts_by_topic: dict[str, list[StoredFact]] | None = None) -> None:
        self.facts_by_topic = facts_by_topic or {}
        self.queries: list[str] = []

    def retrieve(self, topic: str) -> list[StoredFact]:
        self.queries.append(topic)
        return self.facts_by_topic.get(topic, list(_DEFAULT_TC_FACTS))

    def get_events(self) -> list[StoredFact]:
        all_facts: list[StoredFact] = []
        for facts in self.facts_by_topic.values():
            all_facts.extend(facts)
        return all_facts

    def ingest(self, messages: list[Any]) -> None:
        pass


class MockTCJudge:
    """Mock binary judge with deterministic verdicts per check_id."""

    def __init__(
        self,
        default_verdict: Verdict = Verdict.YES,
        overrides: dict[str, Verdict] | None = None,
    ) -> None:
        self.default_verdict = default_verdict
        self.overrides = overrides or {}
        self.call_log: list[dict[str, Any]] = []

    async def judge(self, check_id: str, prompt: str) -> JudgmentResult:
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

    async def judge_across_chunks(self, check_id: str, stored_facts: list[str], prompt_builder) -> JudgmentResult:
        prompt = prompt_builder(stored_facts)
        return await self.judge(check_id, prompt)


def _make_ground_truth(temporal_facts: list[TemporalFact] | None = None) -> GroundTruth:
    """Build a minimal GroundTruth with the given temporal facts."""
    return GroundTruth(
        final_profile={},
        changes=[],
        noise_examples=[],
        signal_examples=[],
        conflicts=[],
        temporal_facts=temporal_facts or [],
        query_relevance_pairs=[],
    )


def _make_temporal_fact(
    fact_id: str = "tf-1",
    description: str = "Test fact",
    value: str = "some value",
    valid_from: str | None = "2026-01-01",
    valid_until: str | None = None,
    query_topic: str = "test topic",
    should_be_current: bool = True,
) -> TemporalFact:
    """Create a TemporalFact for testing."""
    return TemporalFact(
        fact_id=fact_id,
        description=description,
        value=value,
        valid_from=valid_from,
        valid_until=valid_until,
        query_topic=query_topic,
        should_be_current=should_be_current,
    )


def _run(coro):
    """Helper to run an async coroutine synchronously."""
    return asyncio.get_event_loop().run_until_complete(coro)


# ---------------------------------------------------------------------------
# Tests — TCDimension (new MetricDimension-based scorer)
# ---------------------------------------------------------------------------


class TestTCDimension:
    """Tests for the new binary-verdict TCDimension scorer."""

    # -- Class attributes --------------------------------------------------

    def test_class_attributes(self) -> None:
        """TCDimension has correct name and description."""
        dim = TCDimension()
        assert dim.name == "TC"
        assert "temporal" in dim.description.lower()
        assert isinstance(dim.description, str)
        assert len(dim.description) > 20

    def test_repr(self) -> None:
        dim = TCDimension()
        assert "TC" in repr(dim)
        assert repr(dim) == "TCDimension(name='TC')"

    # -- Empty ground truth ------------------------------------------------

    def test_no_temporal_facts(self) -> None:
        """When there are no temporal facts, score should be 1.0 (vacuously correct)."""
        dim = TCDimension()
        adapter = MockTCAdapter()
        judge = MockTCJudge()
        gt = _make_ground_truth(temporal_facts=[])

        result = _run(dim.score(adapter, gt, judge))

        assert isinstance(result, DimensionResult)
        assert result.dimension_name == "TC"
        assert result.score == 1.0
        assert result.passed_checks == 0
        assert result.total_checks == 0
        assert result.details == []
        # Judge should not have been called
        assert len(judge.call_log) == 0

    # -- All checks pass ---------------------------------------------------

    def test_all_current_facts_pass(self) -> None:
        """All should_be_current=True facts with YES verdict → pass."""
        facts = [
            _make_temporal_fact(
                fact_id="tf-job",
                description="Senior Engineer role",
                query_topic="current job",
                should_be_current=True,
            ),
            _make_temporal_fact(
                fact_id="tf-city",
                description="Lives in NYC",
                query_topic="current city",
                should_be_current=True,
            ),
        ]
        adapter = MockTCAdapter(
            facts_by_topic={
                "current job": [StoredFact(text="User is a Senior Engineer")],
                "current city": [StoredFact(text="User lives in NYC")],
            }
        )
        judge = MockTCJudge(default_verdict=Verdict.YES)
        gt = _make_ground_truth(temporal_facts=facts)

        result = _run(TCDimension().score(adapter, gt, judge))

        assert result.score == 1.0
        assert result.passed_checks == 2
        assert result.total_checks == 2
        assert len(result.details) == 2

    def test_all_expired_facts_pass(self) -> None:
        """All should_be_current=False facts with NO verdict → pass.

        For expired facts, NO from judge means system correctly does NOT
        assert the expired fact as current.
        """
        facts = [
            _make_temporal_fact(
                fact_id="tf-old-job",
                description="Junior Engineer role",
                query_topic="old job title",
                should_be_current=False,
                valid_until="2026-01-10",
            ),
            _make_temporal_fact(
                fact_id="tf-old-city",
                description="Used to live in Boston",
                query_topic="old city",
                should_be_current=False,
                valid_until="2026-01-05",
            ),
        ]
        adapter = MockTCAdapter(
            facts_by_topic={
                "old job title": [StoredFact(text="User was promoted from Junior")],
                "old city": [StoredFact(text="User previously lived in Boston")],
            }
        )
        # NO verdict for expired facts = system correctly doesn't assert them
        judge = MockTCJudge(default_verdict=Verdict.NO)
        gt = _make_ground_truth(temporal_facts=facts)

        result = _run(TCDimension().score(adapter, gt, judge))

        assert result.score == 1.0
        assert result.passed_checks == 2
        assert result.total_checks == 2

    def test_mixed_current_and_expired_all_pass(self) -> None:
        """Mix of current and expired facts, all correctly handled."""
        facts = [
            _make_temporal_fact(
                fact_id="tf-current",
                description="Current job",
                query_topic="current job",
                should_be_current=True,
            ),
            _make_temporal_fact(
                fact_id="tf-expired",
                description="Old hobby",
                query_topic="old hobby",
                should_be_current=False,
            ),
        ]
        adapter = MockTCAdapter(
            facts_by_topic={
                "current job": [StoredFact(text="Senior Engineer")],
                "old hobby": [StoredFact(text="Used to play chess")],
            }
        )
        judge = MockTCJudge(
            default_verdict=Verdict.YES,
            overrides={
                "tc_tf-current": Verdict.YES,  # current fact is present → pass
                "tc_tf-expired": Verdict.NO,  # expired fact not asserted → pass
            },
        )
        gt = _make_ground_truth(temporal_facts=facts)

        result = _run(TCDimension().score(adapter, gt, judge))

        assert result.score == 1.0
        assert result.passed_checks == 2
        assert result.total_checks == 2

    # -- All checks fail ---------------------------------------------------

    def test_all_current_facts_fail(self) -> None:
        """Current facts with NO verdict → not found → fail."""
        facts = [
            _make_temporal_fact(
                fact_id="tf-1",
                description="Current job",
                query_topic="job",
                should_be_current=True,
            ),
        ]
        adapter = MockTCAdapter()
        judge = MockTCJudge(default_verdict=Verdict.NO)
        gt = _make_ground_truth(temporal_facts=facts)

        result = _run(TCDimension().score(adapter, gt, judge))

        assert result.score == 0.0
        assert result.passed_checks == 0
        assert result.total_checks == 1
        assert result.details[0]["passed"] is False

    def test_all_expired_facts_fail(self) -> None:
        """Expired facts with YES verdict → still asserted as current → fail."""
        facts = [
            _make_temporal_fact(
                fact_id="tf-1",
                description="Old job",
                query_topic="old job",
                should_be_current=False,
            ),
        ]
        adapter = MockTCAdapter(
            facts_by_topic={
                "old job": [StoredFact(text="User is a Junior Engineer")],
            }
        )
        judge = MockTCJudge(default_verdict=Verdict.YES)
        gt = _make_ground_truth(temporal_facts=facts)

        result = _run(TCDimension().score(adapter, gt, judge))

        assert result.score == 0.0
        assert result.passed_checks == 0
        assert result.total_checks == 1
        assert result.details[0]["passed"] is False

    # -- Mixed results (partial pass) --------------------------------------

    def test_mixed_results_fractional_score(self) -> None:
        """Some pass, some fail → fractional score."""
        facts = [
            _make_temporal_fact(
                fact_id="tf-a",
                description="Current job",
                query_topic="current job",
                should_be_current=True,
            ),
            _make_temporal_fact(
                fact_id="tf-b",
                description="Expired hobby",
                query_topic="old hobby",
                should_be_current=False,
            ),
            _make_temporal_fact(
                fact_id="tf-c",
                description="Current city",
                query_topic="current city",
                should_be_current=True,
            ),
        ]
        adapter = MockTCAdapter()
        judge = MockTCJudge(
            overrides={
                "tc_tf-a": Verdict.YES,  # current, YES → pass
                "tc_tf-b": Verdict.YES,  # expired, YES → fail (still asserted)
                "tc_tf-c": Verdict.NO,  # current, NO → fail (not found)
            },
        )
        gt = _make_ground_truth(temporal_facts=facts)

        result = _run(TCDimension().score(adapter, gt, judge))

        assert result.passed_checks == 1
        assert result.total_checks == 3
        assert abs(result.score - 1.0 / 3.0) < 1e-3

    def test_two_of_three_pass(self) -> None:
        """2/3 facts handled correctly → score ≈ 0.6667."""
        facts = [
            _make_temporal_fact(fact_id="tf-1", query_topic="t1", should_be_current=True),
            _make_temporal_fact(fact_id="tf-2", query_topic="t2", should_be_current=True),
            _make_temporal_fact(fact_id="tf-3", query_topic="t3", should_be_current=False),
        ]
        adapter = MockTCAdapter()
        judge = MockTCJudge(
            overrides={
                "tc_tf-1": Verdict.YES,  # current, found → pass
                "tc_tf-2": Verdict.YES,  # current, found → pass
                "tc_tf-3": Verdict.YES,  # expired, still asserted → FAIL
            },
        )
        gt = _make_ground_truth(temporal_facts=facts)

        result = _run(TCDimension().score(adapter, gt, judge))

        assert result.passed_checks == 2
        assert result.total_checks == 3
        assert abs(result.score - 2.0 / 3.0) < 1e-3

    # -- Single temporal fact ----------------------------------------------

    def test_single_current_fact_pass(self) -> None:
        """Single current fact correctly present → score 1.0."""
        facts = [
            _make_temporal_fact(
                fact_id="tf-solo",
                description="User's current location",
                query_topic="location",
                should_be_current=True,
            ),
        ]
        adapter = MockTCAdapter(
            facts_by_topic={
                "location": [StoredFact(text="User lives in NYC")],
            }
        )
        judge = MockTCJudge(default_verdict=Verdict.YES)
        gt = _make_ground_truth(temporal_facts=facts)

        result = _run(TCDimension().score(adapter, gt, judge))

        assert result.score == 1.0
        assert result.passed_checks == 1
        assert result.total_checks == 1

    def test_single_expired_fact_pass(self) -> None:
        """Single expired fact not asserted → score 1.0."""
        facts = [
            _make_temporal_fact(
                fact_id="tf-solo",
                description="Old address",
                query_topic="old address",
                should_be_current=False,
            ),
        ]
        adapter = MockTCAdapter()
        judge = MockTCJudge(default_verdict=Verdict.NO)
        gt = _make_ground_truth(temporal_facts=facts)

        result = _run(TCDimension().score(adapter, gt, judge))

        assert result.score == 1.0
        assert result.passed_checks == 1
        assert result.total_checks == 1

    def test_single_current_fact_fail(self) -> None:
        """Single current fact not found → score 0.0."""
        facts = [
            _make_temporal_fact(
                fact_id="tf-solo",
                query_topic="location",
                should_be_current=True,
            ),
        ]
        adapter = MockTCAdapter()
        judge = MockTCJudge(default_verdict=Verdict.NO)
        gt = _make_ground_truth(temporal_facts=facts)

        result = _run(TCDimension().score(adapter, gt, judge))

        assert result.score == 0.0
        assert result.passed_checks == 0
        assert result.total_checks == 1

    # -- Detail structure verification -------------------------------------

    def test_details_contain_expected_fields(self) -> None:
        """Each detail dict should contain all expected fields."""
        facts = [
            _make_temporal_fact(
                fact_id="tf-detail",
                description="Senior Engineer role",
                query_topic="job title",
                should_be_current=True,
            ),
        ]
        adapter = MockTCAdapter(
            facts_by_topic={
                "job title": [StoredFact(text="User is Senior Engineer")],
            }
        )
        judge = MockTCJudge(default_verdict=Verdict.YES)
        gt = _make_ground_truth(temporal_facts=facts)

        result = _run(TCDimension().score(adapter, gt, judge))

        assert len(result.details) == 1
        detail = result.details[0]

        expected_keys = {
            "check_id",
            "fact_id",
            "description",
            "should_be_current",
            "verdict",
            "passed",
            "num_stored_facts",
        }
        assert set(detail.keys()) == expected_keys

    def test_detail_values_on_pass_current(self) -> None:
        """Detail values for a passing current fact check."""
        facts = [
            _make_temporal_fact(
                fact_id="tf-x",
                description="Current pet",
                query_topic="pets",
                should_be_current=True,
            ),
        ]
        adapter = MockTCAdapter(
            facts_by_topic={
                "pets": [StoredFact(text="User has a dog"), StoredFact(text="User has a cat")],
            }
        )
        judge = MockTCJudge(default_verdict=Verdict.YES)
        gt = _make_ground_truth(temporal_facts=facts)

        result = _run(TCDimension().score(adapter, gt, judge))

        detail = result.details[0]
        assert detail["check_id"] == "tc_tf-x"
        assert detail["fact_id"] == "tf-x"
        assert detail["description"] == "Current pet"
        assert detail["should_be_current"] is True
        assert detail["verdict"] == "YES"
        assert detail["passed"] is True
        assert detail["num_stored_facts"] == 2

    def test_detail_values_on_fail_expired(self) -> None:
        """Detail values for a failing expired fact check (still asserted)."""
        facts = [
            _make_temporal_fact(
                fact_id="tf-y",
                description="Old job",
                query_topic="old job",
                should_be_current=False,
            ),
        ]
        adapter = MockTCAdapter(
            facts_by_topic={
                "old job": [StoredFact(text="User is a Junior Engineer")],
            }
        )
        judge = MockTCJudge(default_verdict=Verdict.YES)
        gt = _make_ground_truth(temporal_facts=facts)

        result = _run(TCDimension().score(adapter, gt, judge))

        detail = result.details[0]
        assert detail["check_id"] == "tc_tf-y"
        assert detail["should_be_current"] is False
        assert detail["verdict"] == "YES"
        assert detail["passed"] is False  # YES for expired = fail

    def test_detail_values_on_pass_expired(self) -> None:
        """Detail values for a passing expired fact check (not asserted)."""
        facts = [
            _make_temporal_fact(
                fact_id="tf-z",
                description="Old diet",
                query_topic="old diet",
                should_be_current=False,
            ),
        ]
        adapter = MockTCAdapter()
        judge = MockTCJudge(default_verdict=Verdict.NO)
        gt = _make_ground_truth(temporal_facts=facts)

        result = _run(TCDimension().score(adapter, gt, judge))

        detail = result.details[0]
        assert detail["should_be_current"] is False
        assert detail["verdict"] == "NO"
        assert detail["passed"] is True  # NO for expired = pass

    # -- Adapter interaction verification ----------------------------------

    def test_adapter_queried_with_correct_topics(self) -> None:
        """Adapter.retrieve is called with each temporal fact's query_topic."""
        facts = [
            _make_temporal_fact(fact_id="tf-1", query_topic="topic alpha"),
            _make_temporal_fact(fact_id="tf-2", query_topic="topic beta"),
            _make_temporal_fact(fact_id="tf-3", query_topic="topic gamma"),
        ]
        adapter = MockTCAdapter()
        judge = MockTCJudge(default_verdict=Verdict.YES)
        gt = _make_ground_truth(temporal_facts=facts)

        _run(TCDimension().score(adapter, gt, judge))

        assert adapter.queries == ["topic alpha", "topic beta", "topic gamma"]

    def test_empty_adapter_response(self) -> None:
        """When adapter returns no facts, all checks fail (no knowledge)."""
        facts = [
            _make_temporal_fact(
                fact_id="tf-empty",
                query_topic="nonexistent topic",
                should_be_current=True,
            ),
        ]
        # Explicitly return empty for this specific topic
        adapter = MockTCAdapter(facts_by_topic={"nonexistent topic": []})
        judge = MockTCJudge(default_verdict=Verdict.NO)
        gt = _make_ground_truth(temporal_facts=facts)

        result = _run(TCDimension().score(adapter, gt, judge))

        assert result.total_checks == 1
        assert result.passed_checks == 0  # No stored facts → fail
        assert result.details[0]["num_stored_facts"] == 0

    def test_adapter_with_many_facts(self) -> None:
        """Adapter returning many facts still works correctly."""
        many_facts = [StoredFact(text=f"Fact {i}") for i in range(50)]
        facts = [
            _make_temporal_fact(
                fact_id="tf-many",
                query_topic="many facts topic",
                should_be_current=True,
            ),
        ]
        adapter = MockTCAdapter(
            facts_by_topic={
                "many facts topic": many_facts,
            }
        )
        judge = MockTCJudge(default_verdict=Verdict.YES)
        gt = _make_ground_truth(temporal_facts=facts)

        result = _run(TCDimension().score(adapter, gt, judge))

        assert result.total_checks == 1
        assert result.details[0]["num_stored_facts"] == 50

    # -- Judge interaction verification ------------------------------------

    def test_judge_called_with_correct_check_ids(self) -> None:
        """Judge is called with check_id = 'tc_{fact_id}'."""
        facts = [
            _make_temporal_fact(fact_id="tf-alpha", query_topic="t1"),
            _make_temporal_fact(fact_id="tf-beta", query_topic="t2"),
        ]
        adapter = MockTCAdapter()
        judge = MockTCJudge(default_verdict=Verdict.YES)
        gt = _make_ground_truth(temporal_facts=facts)

        _run(TCDimension().score(adapter, gt, judge))

        check_ids = [call["check_id"] for call in judge.call_log]
        assert check_ids == ["tc_tf-alpha", "tc_tf-beta"]

    def test_judge_receives_correct_prompts_for_current_fact(self) -> None:
        """Judge prompt for a current fact should mention the description."""
        facts = [
            _make_temporal_fact(
                fact_id="tf-job",
                description="Senior Software Engineer role",
                query_topic="current job",
                should_be_current=True,
            ),
        ]
        adapter = MockTCAdapter(
            facts_by_topic={
                "current job": [StoredFact(text="User is a Senior Engineer")],
            }
        )
        judge = MockTCJudge(default_verdict=Verdict.YES)
        gt = _make_ground_truth(temporal_facts=facts)

        _run(TCDimension().score(adapter, gt, judge))

        assert len(judge.call_log) == 1
        prompt = judge.call_log[0]["prompt"]
        # Prompt should contain the fact description
        assert "Senior Software Engineer role" in prompt
        # Prompt should contain the stored fact text
        assert "User is a Senior Engineer" in prompt

    def test_judge_receives_correct_prompts_for_expired_fact(self) -> None:
        """Judge prompt for an expired fact should differ from current."""
        facts = [
            _make_temporal_fact(
                fact_id="tf-old",
                description="Old hobby: chess",
                query_topic="old hobbies",
                should_be_current=False,
            ),
        ]
        adapter = MockTCAdapter(
            facts_by_topic={
                "old hobbies": [StoredFact(text="User used to play chess")],
            }
        )
        judge = MockTCJudge(default_verdict=Verdict.NO)
        gt = _make_ground_truth(temporal_facts=facts)

        _run(TCDimension().score(adapter, gt, judge))

        assert len(judge.call_log) == 1
        prompt = judge.call_log[0]["prompt"]
        assert "Old hobby: chess" in prompt
        assert "User used to play chess" in prompt

    # -- Score rounding ----------------------------------------------------

    def test_score_is_rounded(self) -> None:
        """Score should be rounded to 4 decimal places."""
        # 1 out of 3 passes → 0.333333... → rounded to 0.3333
        facts = [
            _make_temporal_fact(fact_id="tf-1", query_topic="t1", should_be_current=True),
            _make_temporal_fact(fact_id="tf-2", query_topic="t2", should_be_current=True),
            _make_temporal_fact(fact_id="tf-3", query_topic="t3", should_be_current=True),
        ]
        adapter = MockTCAdapter()
        judge = MockTCJudge(
            default_verdict=Verdict.NO,
            overrides={"tc_tf-1": Verdict.YES},
        )
        gt = _make_ground_truth(temporal_facts=facts)

        result = _run(TCDimension().score(adapter, gt, judge))

        assert result.score == round(1.0 / 3.0, 4)

    # -- Edge cases --------------------------------------------------------

    def test_many_temporal_facts(self) -> None:
        """Scorer handles many temporal facts correctly."""
        n = 20
        facts = [
            _make_temporal_fact(
                fact_id=f"tf-{i}",
                query_topic=f"topic-{i}",
                should_be_current=(i % 2 == 0),
            )
            for i in range(n)
        ]
        # Even-indexed facts are current → need YES to pass
        # Odd-indexed facts are expired → need NO to pass
        overrides = {}
        for i in range(n):
            if i % 2 == 0:  # current
                overrides[f"tc_tf-{i}"] = Verdict.YES  # pass
            else:  # expired
                overrides[f"tc_tf-{i}"] = Verdict.NO  # pass

        adapter = MockTCAdapter()
        judge = MockTCJudge(overrides=overrides)
        gt = _make_ground_truth(temporal_facts=facts)

        result = _run(TCDimension().score(adapter, gt, judge))

        assert result.passed_checks == n
        assert result.total_checks == n
        assert result.score == 1.0

    def test_temporal_facts_with_none_validity(self) -> None:
        """Temporal facts with None valid_from/valid_until still work."""
        facts = [
            _make_temporal_fact(
                fact_id="tf-no-dates",
                description="Fact with no date constraints",
                valid_from=None,
                valid_until=None,
                query_topic="some topic",
                should_be_current=True,
            ),
        ]
        adapter = MockTCAdapter()
        judge = MockTCJudge(default_verdict=Verdict.YES)
        gt = _make_ground_truth(temporal_facts=facts)

        result = _run(TCDimension().score(adapter, gt, judge))

        assert result.total_checks == 1
        assert result.passed_checks == 1
        assert result.score == 1.0

    def test_dimension_result_type(self) -> None:
        """Result should always be a DimensionResult instance."""
        facts = [
            _make_temporal_fact(fact_id="tf-1", query_topic="t1"),
        ]
        adapter = MockTCAdapter()
        judge = MockTCJudge(default_verdict=Verdict.YES)
        gt = _make_ground_truth(temporal_facts=facts)

        result = _run(TCDimension().score(adapter, gt, judge))

        assert isinstance(result, DimensionResult)
        assert result.dimension_name == "TC"

    def test_all_expired_all_fail(self) -> None:
        """All expired facts with YES (still asserted) → score 0.0."""
        facts = [
            _make_temporal_fact(fact_id="tf-1", query_topic="t1", should_be_current=False),
            _make_temporal_fact(fact_id="tf-2", query_topic="t2", should_be_current=False),
        ]
        adapter = MockTCAdapter()
        judge = MockTCJudge(default_verdict=Verdict.YES)  # YES = still asserted = fail
        gt = _make_ground_truth(temporal_facts=facts)

        result = _run(TCDimension().score(adapter, gt, judge))

        assert result.score == 0.0
        assert result.passed_checks == 0
        assert result.total_checks == 2
