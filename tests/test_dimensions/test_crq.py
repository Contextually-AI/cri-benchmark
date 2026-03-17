"""Tests for the CRQ dimension scorer.

Tests both the new ``CRQDimension`` (binary verdict, MetricDimension) and the

"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from cri.models import (
    ConflictScenario,
    DimensionResult,
    GroundTruth,
    JudgmentResult,
    StoredFact,
    Verdict,
)
from cri.scoring.dimensions.crq import CRQDimension

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ground_truth(conflicts: list[ConflictScenario] | None = None) -> GroundTruth:
    """Create a minimal GroundTruth with the given conflict scenarios."""
    return GroundTruth(
        final_profile={},
        changes=[],
        noise_examples=[],
        signal_examples=[],
        conflicts=conflicts or [],
        temporal_facts=[],
        query_relevance_pairs=[],
    )


def _make_conflict(
    conflict_id: str = "c1",
    topic: str = "dietary preference",
    correct_resolution: str = "The user is vegan",
    resolution_type: str = "explicit_correction",
) -> ConflictScenario:
    """Create a ConflictScenario for testing."""
    return ConflictScenario(
        conflict_id=conflict_id,
        topic=topic,
        conflicting_statements=[
            "The user is vegetarian",
            "The user is vegan",
        ],
        correct_resolution=correct_resolution,
        resolution_type=resolution_type,
        introduced_at_messages=[10, 20],
    )


def _make_judgment(
    check_id: str,
    verdict: Verdict,
    unanimous: bool = True,
) -> JudgmentResult:
    """Create a JudgmentResult for testing."""
    if unanimous:
        votes = [verdict, verdict, verdict]
    else:
        opposite = Verdict.NO if verdict is Verdict.YES else Verdict.YES
        votes = [verdict, verdict, opposite]
    return JudgmentResult(
        check_id=check_id,
        verdict=verdict,
        votes=votes,
        unanimous=unanimous,
        prompt="test prompt",
        raw_responses=["YES" if verdict is Verdict.YES else "NO"] * 3,
    )


def _make_mock_adapter(facts: list[str] | None = None) -> MagicMock:
    """Create a mock adapter that returns the given facts for any query."""
    adapter = MagicMock()
    stored_facts = [StoredFact(text=f) for f in (facts or [])]
    adapter.retrieve.return_value = stored_facts
    adapter.get_events.return_value = stored_facts
    return adapter


def _make_mock_judge(verdicts: list[Verdict]) -> MagicMock:
    """Create a mock BinaryJudge that returns verdicts in order."""
    judge = MagicMock()
    results = [_make_judgment(f"crq-c{i}", v) for i, v in enumerate(verdicts, start=1)]
    judge.judge.side_effect = results
    return judge


# ---------------------------------------------------------------------------
# Tests — CRQDimension (new binary-verdict implementation)
# ---------------------------------------------------------------------------


class TestCRQDimension:
    """Tests for the new MetricDimension-based CRQ scorer."""

    def test_class_attributes(self) -> None:
        dim = CRQDimension()
        assert dim.name == "CRQ"
        assert "conflict" in dim.description.lower()

    def test_repr(self) -> None:
        dim = CRQDimension()
        assert "CRQ" in repr(dim)

    def test_no_conflicts_returns_perfect_score(self) -> None:
        """When there are no conflict scenarios, score should be 1.0."""
        dim = CRQDimension()
        adapter = _make_mock_adapter()
        judge = MagicMock()
        gt = _make_ground_truth(conflicts=[])

        result = asyncio.get_event_loop().run_until_complete(dim.score(adapter, gt, judge))

        assert isinstance(result, DimensionResult)
        assert result.dimension_name == "CRQ"
        assert result.score == 1.0
        assert result.passed_checks == 0
        assert result.total_checks == 0
        assert result.details == []
        # Judge should not have been called
        judge.judge.assert_not_called()

    def test_all_conflicts_resolved(self) -> None:
        """All conflicts resolved correctly → score = 1.0."""
        dim = CRQDimension()
        conflicts = [
            _make_conflict(conflict_id="c1", topic="diet"),
            _make_conflict(conflict_id="c2", topic="job"),
        ]
        gt = _make_ground_truth(conflicts=conflicts)
        adapter = _make_mock_adapter(facts=["User is vegan", "User works at Acme"])
        judge = _make_mock_judge([Verdict.YES, Verdict.YES])

        result = asyncio.get_event_loop().run_until_complete(dim.score(adapter, gt, judge))

        assert result.score == 1.0
        assert result.passed_checks == 2
        assert result.total_checks == 2
        assert len(result.details) == 2
        assert all(d["passed"] for d in result.details)

    def test_no_conflicts_resolved(self) -> None:
        """No conflicts resolved → score = 0.0."""
        dim = CRQDimension()
        conflicts = [
            _make_conflict(conflict_id="c1"),
            _make_conflict(conflict_id="c2"),
        ]
        gt = _make_ground_truth(conflicts=conflicts)
        adapter = _make_mock_adapter()
        judge = _make_mock_judge([Verdict.NO, Verdict.NO])

        result = asyncio.get_event_loop().run_until_complete(dim.score(adapter, gt, judge))

        assert result.score == 0.0
        assert result.passed_checks == 0
        assert result.total_checks == 2
        assert not any(d["passed"] for d in result.details)

    def test_partial_resolution(self) -> None:
        """Some conflicts resolved → score = passed / total."""
        dim = CRQDimension()
        conflicts = [
            _make_conflict(conflict_id="c1"),
            _make_conflict(conflict_id="c2"),
            _make_conflict(conflict_id="c3"),
            _make_conflict(conflict_id="c4"),
        ]
        gt = _make_ground_truth(conflicts=conflicts)
        adapter = _make_mock_adapter(facts=["some fact"])
        judge = _make_mock_judge([Verdict.YES, Verdict.NO, Verdict.YES, Verdict.NO])

        result = asyncio.get_event_loop().run_until_complete(dim.score(adapter, gt, judge))

        assert result.score == pytest.approx(0.5)
        assert result.passed_checks == 2
        assert result.total_checks == 4

    def test_single_conflict_pass(self) -> None:
        """Single conflict resolved → score = 1.0."""
        dim = CRQDimension()
        conflicts = [_make_conflict(conflict_id="c1")]
        gt = _make_ground_truth(conflicts=conflicts)
        adapter = _make_mock_adapter(facts=["User is vegan"])
        judge = _make_mock_judge([Verdict.YES])

        result = asyncio.get_event_loop().run_until_complete(dim.score(adapter, gt, judge))

        assert result.score == 1.0
        assert result.passed_checks == 1
        assert result.total_checks == 1

    def test_single_conflict_fail(self) -> None:
        """Single conflict not resolved → score = 0.0."""
        dim = CRQDimension()
        conflicts = [_make_conflict(conflict_id="c1")]
        gt = _make_ground_truth(conflicts=conflicts)
        adapter = _make_mock_adapter()
        judge = _make_mock_judge([Verdict.NO])

        result = asyncio.get_event_loop().run_until_complete(dim.score(adapter, gt, judge))

        assert result.score == 0.0
        assert result.passed_checks == 0
        assert result.total_checks == 1

    def test_details_contain_expected_fields(self) -> None:
        """Each detail dict should contain all expected fields."""
        dim = CRQDimension()
        conflict = _make_conflict(
            conflict_id="conflict-42",
            topic="programming language",
            correct_resolution="User prefers Python",
            resolution_type="recency",
        )
        gt = _make_ground_truth(conflicts=[conflict])
        adapter = _make_mock_adapter(facts=["User likes Python"])
        judge = _make_mock_judge([Verdict.YES])

        result = asyncio.get_event_loop().run_until_complete(dim.score(adapter, gt, judge))

        assert len(result.details) == 1
        detail = result.details[0]
        assert detail["check_id"] == "crq-conflict-42"
        assert detail["conflict_id"] == "conflict-42"
        assert detail["topic"] == "programming language"
        assert detail["resolution_type"] == "recency"
        assert detail["correct_resolution"] == "User prefers Python"
        assert detail["stored_facts_count"] == 1
        assert detail["verdict"] == "YES"
        assert detail["passed"] is True
        assert "unanimous" in detail

    def test_adapter_queried_with_correct_topic(self) -> None:
        """The adapter should be queried with each conflict's topic."""
        dim = CRQDimension()
        conflicts = [
            _make_conflict(conflict_id="c1", topic="favorite food"),
            _make_conflict(conflict_id="c2", topic="home city"),
        ]
        gt = _make_ground_truth(conflicts=conflicts)
        adapter = _make_mock_adapter()
        judge = _make_mock_judge([Verdict.YES, Verdict.NO])

        asyncio.get_event_loop().run_until_complete(dim.score(adapter, gt, judge))

        assert adapter.retrieve.call_count == 2
        topics = [call.args[0] for call in adapter.retrieve.call_args_list]
        assert topics == ["favorite food", "home city"]

    def test_judge_called_with_correct_check_ids(self) -> None:
        """Judge should be called with check_id = 'crq-{conflict_id}'."""
        dim = CRQDimension()
        conflicts = [
            _make_conflict(conflict_id="alpha"),
            _make_conflict(conflict_id="beta"),
        ]
        gt = _make_ground_truth(conflicts=conflicts)
        adapter = _make_mock_adapter()
        judge = _make_mock_judge([Verdict.YES, Verdict.YES])

        asyncio.get_event_loop().run_until_complete(dim.score(adapter, gt, judge))

        assert judge.judge.call_count == 2
        check_ids = [call.kwargs["check_id"] for call in judge.judge.call_args_list]
        assert check_ids == ["crq-alpha", "crq-beta"]

    def test_empty_stored_facts(self) -> None:
        """When the adapter returns no facts, the judge should still be called."""
        dim = CRQDimension()
        conflicts = [_make_conflict(conflict_id="c1")]
        gt = _make_ground_truth(conflicts=conflicts)
        adapter = _make_mock_adapter(facts=[])
        judge = _make_mock_judge([Verdict.NO])

        result = asyncio.get_event_loop().run_until_complete(dim.score(adapter, gt, judge))

        assert result.score == 0.0
        assert result.total_checks == 1
        assert judge.judge.call_count == 1
