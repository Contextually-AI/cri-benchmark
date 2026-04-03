"""Tests for the DBU dimension scorer.

Tests both the new DBUDimension (MetricDimension) and the legacy DBUScorer.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from cri.models import (
    BeliefChange,
    DimensionResult,
    GroundTruth,
    JudgmentResult,
    StoredFact,
    Verdict,
)
from cri.scoring.dimensions.dbu import DBUDimension

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ground_truth(changes: list[BeliefChange] | None = None) -> GroundTruth:
    """Build a minimal GroundTruth with the given belief changes."""
    return GroundTruth(
        final_profile={},
        changes=changes or [],
        noise_examples=[],
        signal_examples=[],
        conflicts=[],
        temporal_facts=[],
        query_relevance_pairs=[],
    )


def _make_judgment(check_id: str, verdict: Verdict) -> JudgmentResult:
    """Build a JudgmentResult with the given verdict."""
    return JudgmentResult(
        check_id=check_id,
        verdict=verdict,
        votes=[verdict],
        unanimous=True,
        prompt="test prompt",
        raw_responses=["YES" if verdict is Verdict.YES else "NO"],
    )


def _make_adapter(facts: list[str] | None = None) -> MagicMock:
    """Build a mock adapter that returns the given fact strings for any query."""
    adapter = MagicMock()
    stored = [StoredFact(text=f) for f in (facts or [])]
    adapter.retrieve.return_value = stored
    adapter.get_events.return_value = stored
    return adapter


def _make_judge(verdicts: dict[str, Verdict]) -> MagicMock:
    """Build a mock judge that returns specified verdicts by check_id.

    Args:
        verdicts: Mapping from check_id to the desired Verdict.
    """
    judge = MagicMock()

    async def _judge_side_effect(check_id: str, prompt: str) -> JudgmentResult:
        v = verdicts.get(check_id, Verdict.NO)
        return _make_judgment(check_id, v)

    async def _chunks_side_effect(check_id: str, stored_facts: list[str], prompt_builder) -> JudgmentResult:
        prompt_builder(stored_facts)
        v = verdicts.get(check_id, Verdict.NO)
        return _make_judgment(check_id, v)

    judge.judge = AsyncMock(side_effect=_judge_side_effect)
    judge.judge_across_chunks = AsyncMock(side_effect=_chunks_side_effect)
    return judge


# ---------------------------------------------------------------------------
# Tests — DBUDimension (new MetricDimension)
# ---------------------------------------------------------------------------


class TestDBUDimension:
    """Tests for the new binary-verdict DBUDimension scorer."""

    def test_class_attributes(self) -> None:
        """DBUDimension has correct name and description."""
        dim = DBUDimension()
        assert dim.name == "DBU"
        assert "update" in dim.description.lower()
        assert "belief" in dim.description.lower()

    def test_repr(self) -> None:
        dim = DBUDimension()
        assert "DBU" in repr(dim)

    def test_no_belief_changes(self) -> None:
        """When there are no belief changes, score should be 1.0."""
        adapter = _make_adapter()
        judge = _make_judge({})
        gt = _make_ground_truth(changes=[])

        result = asyncio.get_event_loop().run_until_complete(DBUDimension().score(adapter, gt, judge))

        assert isinstance(result, DimensionResult)
        assert result.dimension_name == "DBU"
        assert result.score == 1.0
        assert result.passed_checks == 0
        assert result.total_checks == 0
        assert result.details == []
        # Judge should not have been called at all
        judge.judge.assert_not_called()

    def test_all_checks_pass(self) -> None:
        """All belief changes pass → score = 1.0."""
        changes = [
            BeliefChange(
                fact="job title",
                old_value="Engineer",
                new_value="Manager",
                query_topic="current occupation",
                changed_around_msg=10,
            ),
            BeliefChange(
                fact="city",
                old_value="Boston",
                new_value="San Francisco",
                query_topic="current city",
                changed_around_msg=20,
            ),
        ]

        adapter = _make_adapter(["Works as a Manager", "Lives in San Francisco"])
        judge = _make_judge(
            {
                "dbu_recency_0": Verdict.YES,
                "dbu_staleness_0": Verdict.NO,
                "dbu_recency_1": Verdict.YES,
                "dbu_staleness_1": Verdict.NO,
            }
        )
        gt = _make_ground_truth(changes=changes)

        result = asyncio.get_event_loop().run_until_complete(DBUDimension().score(adapter, gt, judge))

        assert result.score == 1.0
        assert result.passed_checks == 2
        assert result.total_checks == 2
        assert len(result.details) == 2

        # Verify detail structure
        d0 = result.details[0]
        assert d0["belief_change"] == "job title"
        assert d0["old_value"] == "Engineer"
        assert d0["new_value"] == "Manager"
        assert d0["recency_verdict"] == "YES"
        assert d0["staleness_verdict"] == "NO"
        assert d0["passed"] is True
        assert d0["recency_check_id"] == "dbu_recency_0"
        assert d0["staleness_check_id"] == "dbu_staleness_0"

    def test_all_checks_fail_recency(self) -> None:
        """All recency checks fail → score = 0.0."""
        changes = [
            BeliefChange(
                fact="job",
                old_value="A",
                new_value="B",
                query_topic="job",
                changed_around_msg=5,
            ),
        ]

        adapter = _make_adapter(["Still works as A"])
        judge = _make_judge(
            {
                "dbu_recency_0": Verdict.NO,  # new value NOT reflected
                "dbu_staleness_0": Verdict.NO,  # old value not asserted
            }
        )
        gt = _make_ground_truth(changes=changes)

        result = asyncio.get_event_loop().run_until_complete(DBUDimension().score(adapter, gt, judge))

        # Fails because recency = NO
        assert result.score == 0.0
        assert result.passed_checks == 0
        assert result.total_checks == 1
        assert result.details[0]["passed"] is False

    def test_all_checks_fail_staleness(self) -> None:
        """Old value still asserted as current (staleness = YES) → fail."""
        changes = [
            BeliefChange(
                fact="diet",
                old_value="vegetarian",
                new_value="vegan",
                query_topic="dietary preference",
                changed_around_msg=15,
            ),
        ]

        adapter = _make_adapter(["Is vegan", "Is vegetarian"])
        judge = _make_judge(
            {
                "dbu_recency_0": Verdict.YES,  # new value present
                "dbu_staleness_0": Verdict.YES,  # old value STILL asserted
            }
        )
        gt = _make_ground_truth(changes=changes)

        result = asyncio.get_event_loop().run_until_complete(DBUDimension().score(adapter, gt, judge))

        assert result.score == 0.0
        assert result.passed_checks == 0
        assert result.total_checks == 1
        assert result.details[0]["passed"] is False
        assert result.details[0]["recency_verdict"] == "YES"
        assert result.details[0]["staleness_verdict"] == "YES"

    def test_mixed_results(self) -> None:
        """Some pass, some fail → fractional score."""
        changes = [
            BeliefChange(
                fact="hobby",
                old_value="chess",
                new_value="go",
                query_topic="hobbies",
                changed_around_msg=10,
            ),
            BeliefChange(
                fact="pet",
                old_value="cat",
                new_value="dog",
                query_topic="pets",
                changed_around_msg=20,
            ),
            BeliefChange(
                fact="color",
                old_value="red",
                new_value="blue",
                query_topic="favorite color",
                changed_around_msg=30,
            ),
        ]

        adapter = _make_adapter(["Plays go", "Has a cat and dog", "Likes blue"])
        judge = _make_judge(
            {
                # change 0: PASS (recency YES, staleness NO)
                "dbu_recency_0": Verdict.YES,
                "dbu_staleness_0": Verdict.NO,
                # change 1: FAIL (staleness YES — still asserts old)
                "dbu_recency_1": Verdict.YES,
                "dbu_staleness_1": Verdict.YES,
                # change 2: PASS
                "dbu_recency_2": Verdict.YES,
                "dbu_staleness_2": Verdict.NO,
            }
        )
        gt = _make_ground_truth(changes=changes)

        result = asyncio.get_event_loop().run_until_complete(DBUDimension().score(adapter, gt, judge))

        assert result.passed_checks == 2
        assert result.total_checks == 3
        assert abs(result.score - 2.0 / 3.0) < 1e-9

        # Check individual details
        assert result.details[0]["passed"] is True
        assert result.details[1]["passed"] is False
        assert result.details[2]["passed"] is True

    def test_adapter_query_called_with_correct_topics(self) -> None:
        """Adapter.retrieve is called with each belief change's query_topic."""
        changes = [
            BeliefChange(
                fact="f1",
                old_value="a",
                new_value="b",
                query_topic="topic_alpha",
                changed_around_msg=1,
            ),
            BeliefChange(
                fact="f2",
                old_value="c",
                new_value="d",
                query_topic="topic_beta",
                changed_around_msg=2,
            ),
        ]

        adapter = _make_adapter([])
        judge = _make_judge(
            {
                "dbu_recency_0": Verdict.NO,
                "dbu_staleness_0": Verdict.NO,
                "dbu_recency_1": Verdict.NO,
                "dbu_staleness_1": Verdict.NO,
            }
        )
        gt = _make_ground_truth(changes=changes)

        asyncio.get_event_loop().run_until_complete(DBUDimension().score(adapter, gt, judge))

        # Verify adapter.query was called with each topic
        calls = [c.args[0] for c in adapter.retrieve.call_args_list]
        assert calls == ["topic_alpha", "topic_beta"]

    def test_judge_called_with_correct_check_ids(self) -> None:
        """Judge is called with dbu_recency_N and dbu_staleness_N check IDs."""
        changes = [
            BeliefChange(
                fact="f1",
                old_value="a",
                new_value="b",
                query_topic="t1",
                changed_around_msg=1,
            ),
        ]

        adapter = _make_adapter(["fact b"])
        judge = _make_judge(
            {
                "dbu_recency_0": Verdict.YES,
                "dbu_staleness_0": Verdict.NO,
            }
        )
        gt = _make_ground_truth(changes=changes)

        asyncio.get_event_loop().run_until_complete(DBUDimension().score(adapter, gt, judge))

        check_ids = [c.args[0] for c in judge.judge_across_chunks.call_args_list]
        assert check_ids == ["dbu_recency_0", "dbu_staleness_0"]

    def test_judge_receives_rubric_prompts(self) -> None:
        """Judge receives prompts generated by dbu_recency_check and dbu_staleness_check."""
        changes = [
            BeliefChange(
                fact="job title",
                old_value="Engineer",
                new_value="Manager",
                query_topic="occupation",
                changed_around_msg=10,
            ),
        ]

        adapter = _make_adapter(["Works as Manager"])
        judge = _make_judge(
            {
                "dbu_recency_0": Verdict.YES,
                "dbu_staleness_0": Verdict.NO,
            }
        )
        gt = _make_ground_truth(changes=changes)

        asyncio.get_event_loop().run_until_complete(DBUDimension().score(adapter, gt, judge))

        # Check that the prompts contain expected content
        # judge_across_chunks receives (check_id, stored_facts, prompt_builder)
        # Build the prompts from the captured prompt_builder calls
        calls = judge.judge_across_chunks.call_args_list
        recency_call = calls[0]
        staleness_call = calls[1]

        recency_prompt = recency_call.args[2](recency_call.args[1])
        staleness_prompt = staleness_call.args[2](staleness_call.args[1])

        # Recency prompt should mention the new value
        assert "Manager" in recency_prompt
        assert "job title" in recency_prompt

        # Staleness prompt should mention the old value
        assert "Engineer" in staleness_prompt
        assert "job title" in staleness_prompt

    def test_both_fail_conditions(self) -> None:
        """recency=NO AND staleness=YES → definitely fails."""
        changes = [
            BeliefChange(
                fact="language",
                old_value="Python",
                new_value="Rust",
                query_topic="programming",
                changed_around_msg=5,
            ),
        ]

        adapter = _make_adapter(["Writes Python code"])
        judge = _make_judge(
            {
                "dbu_recency_0": Verdict.NO,  # new value NOT found
                "dbu_staleness_0": Verdict.YES,  # old value STILL asserted
            }
        )
        gt = _make_ground_truth(changes=changes)

        result = asyncio.get_event_loop().run_until_complete(DBUDimension().score(adapter, gt, judge))

        assert result.score == 0.0
        assert result.details[0]["passed"] is False
        assert result.details[0]["recency_verdict"] == "NO"
        assert result.details[0]["staleness_verdict"] == "YES"

    def test_single_belief_change_pass(self) -> None:
        """Single belief change that passes produces score 1.0."""
        changes = [
            BeliefChange(
                fact="email",
                old_value="old@example.com",
                new_value="new@example.com",
                query_topic="email address",
                changed_around_msg=8,
            ),
        ]

        adapter = _make_adapter(["new@example.com"])
        judge = _make_judge(
            {
                "dbu_recency_0": Verdict.YES,
                "dbu_staleness_0": Verdict.NO,
            }
        )
        gt = _make_ground_truth(changes=changes)

        result = asyncio.get_event_loop().run_until_complete(DBUDimension().score(adapter, gt, judge))

        assert result.score == 1.0
        assert result.passed_checks == 1
        assert result.total_checks == 1

    def test_detail_structure_completeness(self) -> None:
        """Each detail dict has all required keys."""
        changes = [
            BeliefChange(
                fact="status",
                old_value="single",
                new_value="married",
                query_topic="relationship status",
                changed_around_msg=50,
            ),
        ]

        adapter = _make_adapter(["married"])
        judge = _make_judge(
            {
                "dbu_recency_0": Verdict.YES,
                "dbu_staleness_0": Verdict.NO,
            }
        )
        gt = _make_ground_truth(changes=changes)

        result = asyncio.get_event_loop().run_until_complete(DBUDimension().score(adapter, gt, judge))

        expected_keys = {
            "belief_change",
            "old_value",
            "new_value",
            "recency_verdict",
            "staleness_verdict",
            "passed",
            "recency_check_id",
            "staleness_check_id",
        }
        assert set(result.details[0].keys()) == expected_keys

    def test_empty_adapter_response(self) -> None:
        """When adapter returns no facts, recency should fail."""
        changes = [
            BeliefChange(
                fact="name",
                old_value="Alice",
                new_value="Bob",
                query_topic="user name",
                changed_around_msg=1,
            ),
        ]

        adapter = _make_adapter([])  # No facts stored
        judge = _make_judge(
            {
                "dbu_recency_0": Verdict.NO,
                "dbu_staleness_0": Verdict.NO,
            }
        )
        gt = _make_ground_truth(changes=changes)

        result = asyncio.get_event_loop().run_until_complete(DBUDimension().score(adapter, gt, judge))

        assert result.score == 0.0
        assert result.details[0]["passed"] is False
