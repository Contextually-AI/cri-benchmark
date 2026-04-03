"""Tests for the QRP (Query Relevance Precision) dimension scorer.

Tests both the new :class:`QRPDimension` (MetricDimension-based) and the


Test strategy follows the standard CRI pattern:
1. Create mock adapter returning known facts
2. Create sample ground truth with query-relevance pairs
3. Use mock judge with predetermined verdicts
4. Verify score matches expected value
5. Verify DimensionResult structure

QRP Algorithm recap:
- For each QueryRelevancePair:
  - Relevance checks: YES = found (pass), NO = not found (fail)
  - Irrelevance checks: YES = incorrectly included (fail), NO = excluded (pass)
  - recall = relevant_found / total_relevant (1.0 if none)
  - precision = irrelevant_excluded / total_irrelevant (1.0 if none)
  - pair_score = 0.5 * recall + 0.5 * precision
- Final score = mean of all pair_scores
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from cri.models import (
    DimensionResult,
    GroundTruth,
    JudgmentResult,
    QueryRelevancePair,
    StoredFact,
    Verdict,
)
from cri.scoring.dimensions.qrp import QRPDimension

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


_DEFAULT_FACTS = [StoredFact(text="placeholder fact for testing")]


class MockQRPAdapter:
    """Mock adapter that returns predetermined facts per query string.

    Facts are configured via the *facts_by_query* mapping. Queries not
    present in the mapping return a default non-empty fact list so that
    the judge evaluation path is exercised.  Use ``EmptyQRPAdapter`` to
    test the zero-facts-returned case.
    """

    def __init__(self, facts_by_query: dict[str, list[StoredFact]] | None = None) -> None:
        self.facts_by_query = facts_by_query or {}
        self.queries: list[str] = []

    def retrieve(self, topic: str) -> list[StoredFact]:
        self.queries.append(topic)
        return self.facts_by_query.get(topic, list(_DEFAULT_FACTS))

    def get_events(self) -> list[StoredFact]:
        all_facts: list[StoredFact] = []
        for facts in self.facts_by_query.values():
            all_facts.extend(facts)
        return all_facts or list(_DEFAULT_FACTS)

    def ingest(self, messages: list[Any]) -> None:
        pass


class EmptyQRPAdapter:
    """Mock adapter that always returns no facts — for zero-facts tests."""

    def __init__(self) -> None:
        self.queries: list[str] = []

    def retrieve(self, topic: str) -> list[StoredFact]:
        self.queries.append(topic)
        return []

    def get_events(self) -> list[StoredFact]:
        return []

    def ingest(self, messages: list[Any]) -> None:
        pass


class MockQRPJudge:
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


def _make_ground_truth(
    pairs: list[QueryRelevancePair] | None = None,
) -> GroundTruth:
    """Build a minimal GroundTruth with the given query-relevance pairs."""
    return GroundTruth(
        final_profile={},
        changes=[],
        noise_examples=[],
        signal_examples=[],
        conflicts=[],
        temporal_facts=[],
        query_relevance_pairs=pairs or [],
    )


def _make_pair(
    query_id: str = "qrp-1",
    query: str = "What is the user's job?",
    relevant: list[str] | None = None,
    irrelevant: list[str] | None = None,
) -> QueryRelevancePair:
    """Create a QueryRelevancePair for testing."""
    return QueryRelevancePair(
        query_id=query_id,
        query=query,
        expected_relevant_facts=relevant or ["Software Engineer"],
        expected_irrelevant_facts=irrelevant or ["likes photography"],
    )


def _run(coro):
    """Helper to run an async coroutine synchronously."""
    return asyncio.get_event_loop().run_until_complete(coro)


# ---------------------------------------------------------------------------
# Tests — QRPDimension (new MetricDimension-based scorer)
# ---------------------------------------------------------------------------


class TestQRPDimension:
    """Tests for the new binary-verdict QRPDimension scorer."""

    # -- Class attributes --------------------------------------------------

    def test_class_attributes(self) -> None:
        """QRPDimension has correct name and description."""
        dim = QRPDimension()
        assert dim.name == "QRP"
        assert "precision" in dim.description.lower() or "relevance" in dim.description.lower()
        assert isinstance(dim.description, str)
        assert len(dim.description) > 20

    def test_repr(self) -> None:
        dim = QRPDimension()
        assert "QRP" in repr(dim)
        assert repr(dim) == "QRPDimension(name='QRP')"

    # -- Empty ground truth ------------------------------------------------

    def test_no_query_relevance_pairs(self) -> None:
        """When there are no pairs, score should be 0.0."""
        dim = QRPDimension()
        adapter = MockQRPAdapter()
        judge = MockQRPJudge()
        gt = _make_ground_truth(pairs=[])

        result = _run(dim.score(adapter, gt, judge))

        assert isinstance(result, DimensionResult)
        assert result.dimension_name == "QRP"
        assert result.score == 1.0
        assert result.passed_checks == 0
        assert result.total_checks == 0
        assert result.details == []
        assert len(judge.call_log) == 0

    # -- Perfect score (all pass) ------------------------------------------

    def test_all_checks_pass(self) -> None:
        """All relevant found (YES), all irrelevant excluded (NO) → score 1.0.

        For one pair:
          recall = 2/2 = 1.0 (relevant facts found)
          precision = 2/2 = 1.0 (irrelevant facts excluded)
          pair_score = 0.5 * 1.0 + 0.5 * 1.0 = 1.0
        """
        pair = _make_pair(
            query_id="qrp-1",
            query="What is the user's job?",
            relevant=["Software Engineer", "Senior role"],
            irrelevant=["likes photography", "uses Android"],
        )
        adapter = MockQRPAdapter(
            facts_by_query={
                "What is the user's job?": [
                    StoredFact(text="User is a Senior Software Engineer"),
                ],
            }
        )
        judge = MockQRPJudge(
            overrides={
                "qrp_rel_qrp-1_0": Verdict.YES,
                "qrp_rel_qrp-1_1": Verdict.YES,
                "qrp_irr_qrp-1_0": Verdict.NO,
                "qrp_irr_qrp-1_1": Verdict.NO,
            },
        )
        gt = _make_ground_truth(pairs=[pair])

        result = _run(QRPDimension().score(adapter, gt, judge))

        assert result.score == 1.0
        assert result.passed_checks == 4  # 2 relevance + 2 irrelevance
        assert result.total_checks == 4
        assert len(result.details) == 4

    # -- Zero score (all fail) ---------------------------------------------

    def test_all_checks_fail(self) -> None:
        """All relevant not found (NO), all irrelevant included (YES) → score 0.0.

        recall = 0/2 = 0.0
        precision = 0/2 = 0.0
        pair_score = 0.5 * 0.0 + 0.5 * 0.0 = 0.0
        """
        pair = _make_pair(
            query_id="qrp-1",
            relevant=["fact A", "fact B"],
            irrelevant=["noise X", "noise Y"],
        )
        adapter = MockQRPAdapter()
        judge = MockQRPJudge(
            overrides={
                "qrp_rel_qrp-1_0": Verdict.NO,  # relevant not found
                "qrp_rel_qrp-1_1": Verdict.NO,  # relevant not found
                "qrp_irr_qrp-1_0": Verdict.YES,  # irrelevant included
                "qrp_irr_qrp-1_1": Verdict.YES,  # irrelevant included
            },
        )
        gt = _make_ground_truth(pairs=[pair])

        result = _run(QRPDimension().score(adapter, gt, judge))

        assert result.score == 0.0
        assert result.passed_checks == 0
        assert result.total_checks == 4

    # -- Mixed results (partial pass) --------------------------------------

    def test_mixed_relevance_and_irrelevance(self) -> None:
        """Partial recall and precision → intermediate score.

        recall = 1/2 = 0.5 (one relevant found, one not)
        precision = 1/2 = 0.5 (one irrelevant excluded, one included)
        pair_score = 0.5 * 0.5 + 0.5 * 0.5 = 0.5
        """
        pair = _make_pair(
            query_id="qrp-1",
            relevant=["fact A", "fact B"],
            irrelevant=["noise X", "noise Y"],
        )
        adapter = MockQRPAdapter()
        judge = MockQRPJudge(
            overrides={
                "qrp_rel_qrp-1_0": Verdict.YES,  # found
                "qrp_rel_qrp-1_1": Verdict.NO,  # not found
                "qrp_irr_qrp-1_0": Verdict.NO,  # excluded (pass)
                "qrp_irr_qrp-1_1": Verdict.YES,  # included (fail)
            },
        )
        gt = _make_ground_truth(pairs=[pair])

        result = _run(QRPDimension().score(adapter, gt, judge))

        assert result.score == pytest.approx(0.5)
        assert result.passed_checks == 2  # 1 relevance + 1 irrelevance
        assert result.total_checks == 4

    def test_perfect_recall_zero_precision(self) -> None:
        """All relevant found but all irrelevant included.

        recall = 1/1 = 1.0
        precision = 0/1 = 0.0
        pair_score = 0.5 * 1.0 + 0.5 * 0.0 = 0.5
        """
        pair = _make_pair(
            query_id="qrp-1",
            relevant=["fact A"],
            irrelevant=["noise X"],
        )
        adapter = MockQRPAdapter()
        judge = MockQRPJudge(
            overrides={
                "qrp_rel_qrp-1_0": Verdict.YES,
                "qrp_irr_qrp-1_0": Verdict.YES,  # irrelevant included = fail
            },
        )
        gt = _make_ground_truth(pairs=[pair])

        result = _run(QRPDimension().score(adapter, gt, judge))

        assert result.score == pytest.approx(0.5)
        assert result.passed_checks == 1
        assert result.total_checks == 2

    def test_zero_recall_perfect_precision(self) -> None:
        """No relevant found but all irrelevant excluded.

        recall = 0/1 = 0.0
        precision = 1/1 = 1.0
        pair_score = 0.5 * 0.0 + 0.5 * 1.0 = 0.5
        """
        pair = _make_pair(
            query_id="qrp-1",
            relevant=["fact A"],
            irrelevant=["noise X"],
        )
        adapter = MockQRPAdapter()
        judge = MockQRPJudge(
            overrides={
                "qrp_rel_qrp-1_0": Verdict.NO,  # not found
                "qrp_irr_qrp-1_0": Verdict.NO,  # excluded = pass
            },
        )
        gt = _make_ground_truth(pairs=[pair])

        result = _run(QRPDimension().score(adapter, gt, judge))

        assert result.score == pytest.approx(0.5)
        assert result.passed_checks == 1
        assert result.total_checks == 2

    # -- Multiple pairs ----------------------------------------------------

    def test_multiple_pairs_averaged(self) -> None:
        """Score is average of pair scores.

        Pair 1: recall=1/1=1.0, precision=1/1=1.0 → pair_score=1.0
        Pair 2: recall=0/1=0.0, precision=0/1=0.0 → pair_score=0.0
        Final = (1.0 + 0.0) / 2 = 0.5
        """
        pair1 = _make_pair(
            query_id="qrp-1",
            query="user's job?",
            relevant=["Engineer"],
            irrelevant=["photography"],
        )
        pair2 = _make_pair(
            query_id="qrp-2",
            query="user's hobbies?",
            relevant=["biking"],
            irrelevant=["New York"],
        )
        adapter = MockQRPAdapter()
        judge = MockQRPJudge(
            overrides={
                # Pair 1: perfect
                "qrp_rel_qrp-1_0": Verdict.YES,
                "qrp_irr_qrp-1_0": Verdict.NO,
                # Pair 2: all fail
                "qrp_rel_qrp-2_0": Verdict.NO,
                "qrp_irr_qrp-2_0": Verdict.YES,
            },
        )
        gt = _make_ground_truth(pairs=[pair1, pair2])

        result = _run(QRPDimension().score(adapter, gt, judge))

        assert result.score == pytest.approx(0.5)
        assert result.total_checks == 4

    def test_three_pairs_varied_scores(self) -> None:
        """Three pairs with varied scores.

        Pair 1: recall=1.0, precision=1.0 → 1.0
        Pair 2: recall=0.5, precision=1.0 → 0.75
        Pair 3: recall=1.0, precision=0.0 → 0.5
        Mean = (1.0 + 0.75 + 0.5) / 3 = 0.75
        """
        pair1 = _make_pair(query_id="p1", relevant=["a"], irrelevant=["x"])
        pair2 = _make_pair(query_id="p2", relevant=["a", "b"], irrelevant=["x"])
        pair3 = _make_pair(query_id="p3", relevant=["a"], irrelevant=["x"])

        adapter = MockQRPAdapter()
        judge = MockQRPJudge(
            overrides={
                # Pair 1: perfect
                "qrp_rel_p1_0": Verdict.YES,
                "qrp_irr_p1_0": Verdict.NO,
                # Pair 2: one relevant found, one not; irrelevant excluded
                "qrp_rel_p2_0": Verdict.YES,
                "qrp_rel_p2_1": Verdict.NO,
                "qrp_irr_p2_0": Verdict.NO,
                # Pair 3: relevant found; irrelevant included
                "qrp_rel_p3_0": Verdict.YES,
                "qrp_irr_p3_0": Verdict.YES,
            },
        )
        gt = _make_ground_truth(pairs=[pair1, pair2, pair3])

        result = _run(QRPDimension().score(adapter, gt, judge))

        assert result.score == pytest.approx(0.75, abs=0.001)

    # -- Only relevant facts (no irrelevant) -------------------------------

    def test_only_relevant_facts_all_found(self) -> None:
        """Pair with only relevant facts (no irrelevant).

        recall = 2/2 = 1.0
        precision = 1.0 (no irrelevant facts → defaults to 1.0)
        pair_score = 0.5 * 1.0 + 0.5 * 1.0 = 1.0
        """
        pair = QueryRelevancePair(
            query_id="qrp-1",
            query="What is the user's job?",
            expected_relevant_facts=["fact A", "fact B"],
            expected_irrelevant_facts=[],
        )
        adapter = MockQRPAdapter()
        judge = MockQRPJudge(
            overrides={
                "qrp_rel_qrp-1_0": Verdict.YES,
                "qrp_rel_qrp-1_1": Verdict.YES,
            },
        )
        gt = _make_ground_truth(pairs=[pair])

        result = _run(QRPDimension().score(adapter, gt, judge))

        assert result.score == 1.0
        assert result.passed_checks == 2
        assert result.total_checks == 2

    def test_only_relevant_facts_none_found(self) -> None:
        """Only relevant facts, none found.

        recall = 0/2 = 0.0
        precision = 1.0 (default)
        pair_score = 0.5 * 0.0 + 0.5 * 1.0 = 0.5
        """
        pair = QueryRelevancePair(
            query_id="qrp-1",
            query="What is the user's job?",
            expected_relevant_facts=["fact A", "fact B"],
            expected_irrelevant_facts=[],
        )
        adapter = MockQRPAdapter()
        judge = MockQRPJudge(default_verdict=Verdict.NO)
        gt = _make_ground_truth(pairs=[pair])

        result = _run(QRPDimension().score(adapter, gt, judge))

        assert result.score == pytest.approx(0.5)

    # -- Only irrelevant facts (no relevant) -------------------------------

    def test_only_irrelevant_facts_all_excluded(self) -> None:
        """Only irrelevant facts, all excluded.

        recall = 1.0 (no relevant facts → defaults to 1.0)
        precision = 2/2 = 1.0
        pair_score = 1.0
        """
        pair = QueryRelevancePair(
            query_id="qrp-1",
            query="What is the user's job?",
            expected_relevant_facts=[],
            expected_irrelevant_facts=["noise A", "noise B"],
        )
        adapter = MockQRPAdapter()
        judge = MockQRPJudge(default_verdict=Verdict.NO)  # all excluded
        gt = _make_ground_truth(pairs=[pair])

        result = _run(QRPDimension().score(adapter, gt, judge))

        assert result.score == 1.0
        assert result.passed_checks == 2
        assert result.total_checks == 2

    def test_only_irrelevant_facts_all_included(self) -> None:
        """Only irrelevant facts, all incorrectly included.

        recall = 1.0 (default)
        precision = 0/2 = 0.0
        pair_score = 0.5 * 1.0 + 0.5 * 0.0 = 0.5
        """
        pair = QueryRelevancePair(
            query_id="qrp-1",
            query="What is the user's job?",
            expected_relevant_facts=[],
            expected_irrelevant_facts=["noise A", "noise B"],
        )
        adapter = MockQRPAdapter()
        judge = MockQRPJudge(default_verdict=Verdict.YES)  # all included (fail)
        gt = _make_ground_truth(pairs=[pair])

        result = _run(QRPDimension().score(adapter, gt, judge))

        assert result.score == pytest.approx(0.5)
        assert result.passed_checks == 0
        assert result.total_checks == 2

    # -- Detail structure verification -------------------------------------

    def test_details_contain_expected_fields_relevance(self) -> None:
        """Relevance check detail dicts contain all expected fields."""
        pair = QueryRelevancePair(
            query_id="qrp-check",
            query="What is the user's job?",
            expected_relevant_facts=["Software Engineer"],
            expected_irrelevant_facts=[],
        )
        adapter = MockQRPAdapter(
            facts_by_query={
                "What is the user's job?": [StoredFact(text="User is an engineer")],
            }
        )
        judge = MockQRPJudge(default_verdict=Verdict.YES)
        gt = _make_ground_truth(pairs=[pair])

        result = _run(QRPDimension().score(adapter, gt, judge))

        assert len(result.details) == 1
        detail = result.details[0]
        assert detail["check_id"] == "qrp_rel_qrp-check_0"
        assert detail["query_id"] == "qrp-check"
        assert detail["query"] == "What is the user's job?"
        assert detail["check_type"] == "relevance"
        assert detail["expected_fact"] == "Software Engineer"
        assert detail["verdict"] == "YES"
        assert detail["passed"] is True
        assert detail["num_returned_facts"] == 1

    def test_details_contain_expected_fields_irrelevance(self) -> None:
        """Irrelevance check detail dicts contain all expected fields."""
        pair = QueryRelevancePair(
            query_id="qrp-check",
            query="What is the user's job?",
            expected_relevant_facts=[],
            expected_irrelevant_facts=["likes photography"],
        )
        adapter = MockQRPAdapter(
            facts_by_query={
                "What is the user's job?": [StoredFact(text="User is an engineer")],
            }
        )
        judge = MockQRPJudge(default_verdict=Verdict.NO)  # excluded = pass
        gt = _make_ground_truth(pairs=[pair])

        result = _run(QRPDimension().score(adapter, gt, judge))

        assert len(result.details) == 1
        detail = result.details[0]
        assert detail["check_id"] == "qrp_irr_qrp-check_0"
        assert detail["query_id"] == "qrp-check"
        assert detail["query"] == "What is the user's job?"
        assert detail["check_type"] == "irrelevance"
        assert detail["expected_fact"] == "likes photography"
        assert detail["verdict"] == "NO"
        assert detail["passed"] is True
        assert detail["num_returned_facts"] == 1

    def test_details_order_relevance_then_irrelevance(self) -> None:
        """Within a pair, relevance checks come before irrelevance checks."""
        pair = _make_pair(
            query_id="qrp-1",
            relevant=["rel A", "rel B"],
            irrelevant=["irr X", "irr Y"],
        )
        adapter = MockQRPAdapter()
        judge = MockQRPJudge(default_verdict=Verdict.YES)
        gt = _make_ground_truth(pairs=[pair])

        result = _run(QRPDimension().score(adapter, gt, judge))

        assert len(result.details) == 4
        check_types = [d["check_type"] for d in result.details]
        assert check_types == ["relevance", "relevance", "irrelevance", "irrelevance"]

    # -- Adapter interaction verification ----------------------------------

    def test_adapter_queried_with_correct_queries(self) -> None:
        """Adapter.retrieve is called with each pair's query string."""
        pair1 = _make_pair(query_id="p1", query="What is the user's job?")
        pair2 = _make_pair(query_id="p2", query="What are the hobbies?")
        adapter = MockQRPAdapter()
        judge = MockQRPJudge(default_verdict=Verdict.YES)
        gt = _make_ground_truth(pairs=[pair1, pair2])

        _run(QRPDimension().score(adapter, gt, judge))

        assert adapter.queries == ["What is the user's job?", "What are the hobbies?"]

    def test_empty_adapter_response(self) -> None:
        """When adapter returns no facts, pair scores 0 — no useful response."""
        pair = _make_pair(query_id="qrp-1", relevant=["A"], irrelevant=["B"])
        adapter = EmptyQRPAdapter()
        judge = MockQRPJudge()
        gt = _make_ground_truth(pairs=[pair])

        result = _run(QRPDimension().score(adapter, gt, judge))

        assert result.total_checks == 2
        # 0 facts returned → recall=0, precision=0 → pair_score=0
        assert result.score == pytest.approx(0.0)
        assert result.details[0]["num_returned_facts"] == 0
        assert result.details[1]["num_returned_facts"] == 0

    # -- Judge interaction verification ------------------------------------

    def test_judge_called_with_correct_check_ids(self) -> None:
        """Judge is called with check_id following the pattern.

        Expected: 'qrp_rel_{pair_id}_{idx}' and 'qrp_irr_{pair_id}_{idx}'.
        """
        pair = _make_pair(
            query_id="test-pair",
            relevant=["fact A", "fact B"],
            irrelevant=["noise X"],
        )
        adapter = MockQRPAdapter()
        judge = MockQRPJudge(default_verdict=Verdict.YES)
        gt = _make_ground_truth(pairs=[pair])

        _run(QRPDimension().score(adapter, gt, judge))

        check_ids = [call["check_id"] for call in judge.call_log]
        assert check_ids == [
            "qrp_rel_test-pair_0",
            "qrp_rel_test-pair_1",
            "qrp_irr_test-pair_0",
        ]

    def test_judge_receives_correct_prompts(self) -> None:
        """Judge prompts should contain the query and expected facts."""
        pair = _make_pair(
            query_id="qrp-prompt",
            query="What does the user do?",
            relevant=["Software Engineer"],
            irrelevant=["likes biking"],
        )
        adapter = MockQRPAdapter(
            facts_by_query={
                "What does the user do?": [StoredFact(text="User is an engineer")],
            }
        )
        judge = MockQRPJudge(default_verdict=Verdict.YES)
        gt = _make_ground_truth(pairs=[pair])

        _run(QRPDimension().score(adapter, gt, judge))

        assert len(judge.call_log) == 2

        # Relevance prompt should contain query and expected fact
        rel_prompt = judge.call_log[0]["prompt"]
        assert "What does the user do?" in rel_prompt
        assert "Software Engineer" in rel_prompt
        assert "User is an engineer" in rel_prompt

        # Irrelevance prompt should contain query and irrelevant fact
        irr_prompt = judge.call_log[1]["prompt"]
        assert "What does the user do?" in irr_prompt
        assert "likes biking" in irr_prompt

    # -- Score rounding ----------------------------------------------------

    def test_score_is_rounded(self) -> None:
        """Score should be rounded to 4 decimal places."""
        # 2 pairs:
        # Pair 1: recall=1/1=1.0, precision=1/1=1.0 → 1.0
        # Pair 2: recall=1/1=1.0, precision=0/1=0.0 → 0.5
        # Pair 3: recall=0/1=0.0, precision=1/1=1.0 → 0.5
        # Mean = (1.0 + 0.5 + 0.5) / 3 = 0.6667
        pair1 = _make_pair(query_id="p1", relevant=["a"], irrelevant=["x"])
        pair2 = _make_pair(query_id="p2", relevant=["a"], irrelevant=["x"])
        pair3 = _make_pair(query_id="p3", relevant=["a"], irrelevant=["x"])
        adapter = MockQRPAdapter()
        judge = MockQRPJudge(
            overrides={
                "qrp_rel_p1_0": Verdict.YES,
                "qrp_irr_p1_0": Verdict.NO,
                "qrp_rel_p2_0": Verdict.YES,
                "qrp_irr_p2_0": Verdict.YES,  # fail
                "qrp_rel_p3_0": Verdict.NO,  # fail
                "qrp_irr_p3_0": Verdict.NO,
            },
        )
        gt = _make_ground_truth(pairs=[pair1, pair2, pair3])

        result = _run(QRPDimension().score(adapter, gt, judge))

        assert result.score == round(2.0 / 3.0, 4)

    # -- Edge cases --------------------------------------------------------

    def test_single_pair_single_relevant_pass(self) -> None:
        """Single pair with one relevant fact found → score 1.0."""
        pair = QueryRelevancePair(
            query_id="qrp-1",
            query="What is the user's job?",
            expected_relevant_facts=["fact"],
            expected_irrelevant_facts=[],
        )
        adapter = MockQRPAdapter()
        judge = MockQRPJudge(default_verdict=Verdict.YES)
        gt = _make_ground_truth(pairs=[pair])

        result = _run(QRPDimension().score(adapter, gt, judge))

        assert result.score == 1.0
        assert result.passed_checks == 1
        assert result.total_checks == 1

    def test_many_pairs(self) -> None:
        """Scorer handles many pairs correctly."""
        n = 10
        pairs = [
            _make_pair(
                query_id=f"p{i}",
                query=f"query {i}",
                relevant=[f"rel-{i}"],
                irrelevant=[f"irr-{i}"],
            )
            for i in range(n)
        ]
        # All pass
        overrides = {}
        for i in range(n):
            overrides[f"qrp_rel_p{i}_0"] = Verdict.YES
            overrides[f"qrp_irr_p{i}_0"] = Verdict.NO
        adapter = MockQRPAdapter()
        judge = MockQRPJudge(overrides=overrides)
        gt = _make_ground_truth(pairs=pairs)

        result = _run(QRPDimension().score(adapter, gt, judge))

        assert result.score == 1.0
        assert result.passed_checks == n * 2
        assert result.total_checks == n * 2
        assert len(result.details) == n * 2

    def test_many_relevant_and_irrelevant_facts(self) -> None:
        """Pair with many relevant and irrelevant facts."""
        pair = _make_pair(
            query_id="qrp-big",
            relevant=[f"rel-{i}" for i in range(5)],
            irrelevant=[f"irr-{i}" for i in range(5)],
        )
        # 3 out of 5 relevant found, 4 out of 5 irrelevant excluded
        overrides = {
            "qrp_rel_qrp-big_0": Verdict.YES,
            "qrp_rel_qrp-big_1": Verdict.YES,
            "qrp_rel_qrp-big_2": Verdict.YES,
            "qrp_rel_qrp-big_3": Verdict.NO,
            "qrp_rel_qrp-big_4": Verdict.NO,
            "qrp_irr_qrp-big_0": Verdict.NO,
            "qrp_irr_qrp-big_1": Verdict.NO,
            "qrp_irr_qrp-big_2": Verdict.NO,
            "qrp_irr_qrp-big_3": Verdict.NO,
            "qrp_irr_qrp-big_4": Verdict.YES,  # included = fail
        }
        adapter = MockQRPAdapter()
        judge = MockQRPJudge(overrides=overrides)
        gt = _make_ground_truth(pairs=[pair])

        result = _run(QRPDimension().score(adapter, gt, judge))

        # recall = 3/5 = 0.6
        # precision = 4/5 = 0.8
        # pair_score = 0.5 * 0.6 + 0.5 * 0.8 = 0.7
        assert result.score == pytest.approx(0.7, abs=0.001)
        assert result.passed_checks == 7  # 3 relevance + 4 irrelevance
        assert result.total_checks == 10

    def test_dimension_result_type(self) -> None:
        """Result should always be a DimensionResult instance."""
        pair = _make_pair(query_id="qrp-1")
        adapter = MockQRPAdapter()
        judge = MockQRPJudge(default_verdict=Verdict.YES)
        gt = _make_ground_truth(pairs=[pair])

        result = _run(QRPDimension().score(adapter, gt, judge))

        assert isinstance(result, DimensionResult)
        assert result.dimension_name == "QRP"

    def test_empty_relevant_and_irrelevant_pair(self) -> None:
        """Pair with no relevant and no irrelevant facts.

        recall = 1.0 (default, no relevant)
        precision = 1.0 (default, no irrelevant)
        pair_score = 1.0
        """
        pair = QueryRelevancePair(
            query_id="qrp-empty",
            query="What is the user's job?",
            expected_relevant_facts=[],
            expected_irrelevant_facts=[],
        )
        adapter = MockQRPAdapter()
        judge = MockQRPJudge()
        gt = _make_ground_truth(pairs=[pair])

        result = _run(QRPDimension().score(adapter, gt, judge))

        assert result.score == 1.0
        assert result.passed_checks == 0
        assert result.total_checks == 0

    def test_details_passed_false_on_relevance_fail(self) -> None:
        """Relevance check failing should have passed=False in details."""
        pair = QueryRelevancePair(
            query_id="qrp-1",
            query="What is the user's job?",
            expected_relevant_facts=["fact A"],
            expected_irrelevant_facts=[],
        )
        adapter = MockQRPAdapter()
        judge = MockQRPJudge(default_verdict=Verdict.NO)
        gt = _make_ground_truth(pairs=[pair])

        result = _run(QRPDimension().score(adapter, gt, judge))

        assert len(result.details) == 1
        assert result.details[0]["passed"] is False
        assert result.details[0]["verdict"] == "NO"
        assert result.details[0]["check_type"] == "relevance"

    def test_details_passed_false_on_irrelevance_fail(self) -> None:
        """Irrelevance check failing (YES) should have passed=False."""
        pair = QueryRelevancePair(
            query_id="qrp-1",
            query="What is the user's job?",
            expected_relevant_facts=[],
            expected_irrelevant_facts=["noise X"],
        )
        adapter = MockQRPAdapter()
        judge = MockQRPJudge(default_verdict=Verdict.YES)  # included = fail
        gt = _make_ground_truth(pairs=[pair])

        result = _run(QRPDimension().score(adapter, gt, judge))

        assert len(result.details) == 1
        assert result.details[0]["passed"] is False
        assert result.details[0]["verdict"] == "YES"
        assert result.details[0]["check_type"] == "irrelevance"
