"""Integration tests for the CRI Benchmark full pipeline.

Tests the complete evaluation pipeline end-to-end using:
- NoMemoryAdapter: lower bound baseline — discards everything
- FullContextAdapter: upper recall bound — stores all user messages

These tests use mock BinaryJudge implementations to avoid real LLM calls
while still exercising the full scoring pipeline — from adapter querying
through dimension scoring to composite CRI computation.

Key insight about scoring semantics:
- Some dimension checks are "positive" (YES = pass, NO = fail):
  PAS profile checks, DBU recency, CRQ resolution, QRP relevance, MEI coverage
- Some dimension checks are "negative" (YES = fail, NO = pass):
  DBU staleness, TC expired, QRP irrelevance

This means an adapter that stores nothing (NoMemoryAdapter) can still
"pass" negative checks — it correctly avoids storing noise, stale facts,
and irrelevant results.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from cri.adapter import MemoryAdapter
from cri.models import (
    BeliefChange,
    BenchmarkResult,
    ConflictScenario,
    ConversationDataset,
    CRIResult,
    DatasetMetadata,
    DimensionResult,
    GroundTruth,
    JudgmentResult,
    Message,
    NoiseExample,
    PerformanceProfile,
    ProfileDimension,
    QueryRelevancePair,
    SignalExample,
    TemporalFact,
    Verdict,
)
from cri.scoring.engine import ScoringEngine

# Import baseline adapters from examples
sys.path.insert(0, str(Path(__file__).parent.parent / "examples" / "adapters"))
from full_context_adapter import FullContextAdapter  # noqa: E402
from no_memory_adapter import NoMemoryAdapter  # noqa: E402

# ---------------------------------------------------------------------------
# Mock judge variants
# ---------------------------------------------------------------------------


class AlwaysNoJudge:
    """A judge that always returns NO verdicts.

    For positive checks (PAS, DBU recency, signal, relevance, CRQ):
      NO → the check fails.
    For negative checks (staleness, noise, stale, irrelevance, TC expired):
      NO → the check passes (correctly not found).
    """

    def __init__(self) -> None:
        self._log: list[JudgmentResult] = []
        self.num_runs: int = 3

    async def judge(self, check_id: str, prompt: str) -> JudgmentResult:
        result = JudgmentResult(
            check_id=check_id,
            verdict=Verdict.NO,
            votes=[Verdict.NO, Verdict.NO, Verdict.NO],
            unanimous=True,
            prompt=prompt,
            raw_responses=["NO", "NO", "NO"],
        )
        self._log.append(result)
        return result

    async def judge_across_chunks(self, check_id: str, stored_facts: list[str], prompt_builder) -> JudgmentResult:
        prompt = prompt_builder(stored_facts)
        return await self.judge(check_id, prompt)

    def get_log(self) -> list[JudgmentResult]:
        return list(self._log)


class SmartMockJudge:
    """A judge that returns YES when stored facts are present in the prompt.

    Inspects the prompt for the "(no facts provided)" marker:
    - If found → NO (no facts to evaluate)
    - If not found (facts are present) → YES

    This simulates a realistic judge: when facts are present,
    positive checks pass (YES) but negative checks also get YES (fail).
    """

    def __init__(self) -> None:
        self._log: list[JudgmentResult] = []
        self.num_runs: int = 3

    async def judge(self, check_id: str, prompt: str) -> JudgmentResult:
        prompt_upper = prompt.upper()

        # If no facts provided, verdict is NO
        verdict = Verdict.NO if "(NO FACTS PROVIDED)" in prompt_upper else Verdict.YES

        result = JudgmentResult(
            check_id=check_id,
            verdict=verdict,
            votes=[verdict, verdict, verdict],
            unanimous=True,
            prompt=prompt,
            raw_responses=[verdict.value] * 3,
        )
        self._log.append(result)
        return result

    async def judge_across_chunks(self, check_id: str, stored_facts: list[str], prompt_builder) -> JudgmentResult:
        prompt = prompt_builder(stored_facts)
        return await self.judge(check_id, prompt)

    async def judge_coverage(self, _check_id: str, prompt: str) -> set[int]:
        """Return all GT fact indices as covered when facts are present."""
        if "(NO FACTS PROVIDED)" in prompt.upper():
            return set()
        # Extract the number of GT facts from the prompt by counting "[N]" lines.
        import re

        indices = {int(m) for m in re.findall(r"\[(\d+)\]", prompt)}
        return indices

    def get_log(self) -> list[JudgmentResult]:
        return list(self._log)


class AlwaysYesJudge:
    """A judge that always returns YES verdicts."""

    def __init__(self) -> None:
        self._log: list[JudgmentResult] = []
        self.num_runs: int = 3

    async def judge(self, check_id: str, prompt: str) -> JudgmentResult:
        result = JudgmentResult(
            check_id=check_id,
            verdict=Verdict.YES,
            votes=[Verdict.YES, Verdict.YES, Verdict.YES],
            unanimous=True,
            prompt=prompt,
            raw_responses=["YES", "YES", "YES"],
        )
        self._log.append(result)
        return result

    async def judge_across_chunks(self, check_id: str, stored_facts: list[str], prompt_builder) -> JudgmentResult:
        prompt = prompt_builder(stored_facts)
        return await self.judge(check_id, prompt)

    def get_log(self) -> list[JudgmentResult]:
        return list(self._log)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_messages() -> list[Message]:
    """A realistic set of conversation messages for integration testing."""
    return [
        Message(
            message_id=1,
            role="user",
            content="Hi, I'm Alice. I'm a 30-year-old software engineer in New York.",
            timestamp="2026-01-01T10:00:00Z",
            session_id="sess-001",
            day=1,
        ),
        Message(
            message_id=2,
            role="assistant",
            content="Nice to meet you, Alice! Software engineering in NYC sounds exciting.",
            timestamp="2026-01-01T10:00:30Z",
            session_id="sess-001",
            day=1,
        ),
        Message(
            message_id=3,
            role="user",
            content="I love hiking and photography. I go to the Catskills every weekend.",
            timestamp="2026-01-01T10:02:00Z",
            session_id="sess-001",
            day=1,
        ),
        Message(
            message_id=4,
            role="assistant",
            content="The Catskills are beautiful! Do you prefer landscape or nature photography?",
            timestamp="2026-01-01T10:02:30Z",
            session_id="sess-001",
            day=1,
        ),
        Message(
            message_id=5,
            role="user",
            content="Mostly landscape. I just got a new Fujifilm X-T5 camera.",
            timestamp="2026-01-02T14:00:00Z",
            session_id="sess-002",
            day=2,
        ),
        Message(
            message_id=6,
            role="assistant",
            content="Great choice! The X-T5 is excellent for landscapes.",
            timestamp="2026-01-02T14:00:20Z",
            session_id="sess-002",
            day=2,
        ),
        Message(
            message_id=7,
            role="user",
            content="By the way, I recently switched from iOS to Android. Using a Pixel 8 now.",
            timestamp="2026-01-05T09:30:00Z",
            session_id="sess-003",
            day=5,
        ),
        Message(
            message_id=8,
            role="assistant",
            content="How are you finding the switch to Android?",
            timestamp="2026-01-05T09:30:15Z",
            session_id="sess-003",
            day=5,
        ),
        Message(
            message_id=9,
            role="user",
            content="Actually I love it. The Pixel camera is amazing for my photography hobby.",
            timestamp="2026-01-05T09:31:00Z",
            session_id="sess-003",
            day=5,
        ),
        Message(
            message_id=10,
            role="user",
            content="Oh, I should mention — I just got promoted to Senior Engineer!",
            timestamp="2026-01-10T11:00:00Z",
            session_id="sess-004",
            day=10,
        ),
        Message(
            message_id=11,
            role="assistant",
            content="Congratulations on the promotion, Alice! Well deserved.",
            timestamp="2026-01-10T11:00:10Z",
            session_id="sess-004",
            day=10,
        ),
        Message(
            message_id=12,
            role="user",
            content="Thanks! Also, I no longer go hiking every weekend — moved to biking instead.",
            timestamp="2026-01-15T16:00:00Z",
            session_id="sess-005",
            day=15,
        ),
    ]


@pytest.fixture
def sample_ground_truth() -> GroundTruth:
    """A fully populated ground truth matching the sample messages."""
    return GroundTruth(
        final_profile={
            "occupation": ProfileDimension(
                dimension_name="occupation",
                value="Senior Software Engineer",
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
            "hobbies": ProfileDimension(
                dimension_name="hobbies",
                value=["biking", "photography"],
                query_topic="What are the user's hobbies?",
                category="interests",
            ),
            "phone_os": ProfileDimension(
                dimension_name="phone_os",
                value="Android",
                query_topic="What phone OS does the user use?",
                category="preferences",
            ),
        },
        changes=[
            BeliefChange(
                fact="phone_os",
                old_value="iOS",
                new_value="Android (Pixel 8)",
                query_topic="What phone does the user use?",
                changed_around_msg=7,
                key_messages=[7, 9],
            ),
            BeliefChange(
                fact="occupation_level",
                old_value="Software Engineer",
                new_value="Senior Software Engineer",
                query_topic="What is the user's job title?",
                changed_around_msg=10,
                key_messages=[10],
            ),
            BeliefChange(
                fact="primary_outdoor_activity",
                old_value="hiking",
                new_value="biking",
                query_topic="What outdoor activity does the user prefer?",
                changed_around_msg=12,
                key_messages=[12],
            ),
        ],
        noise_examples=[
            NoiseExample(
                text="How are you doing today?",
                reason="Generic greeting with no factual content",
            ),
            NoiseExample(
                text="That's interesting, tell me more.",
                reason="Conversational filler with no user-specific info",
            ),
        ],
        signal_examples=[
            SignalExample(
                text="I'm a 30-year-old software engineer in New York.",
                target_fact="occupation: Software Engineer; age: 30; location: New York",
            ),
            SignalExample(
                text="I recently switched from iOS to Android.",
                target_fact="phone_os: Android (updated from iOS)",
            ),
            SignalExample(
                text="I just got promoted to Senior Engineer!",
                target_fact="occupation_level: Senior Engineer",
            ),
        ],
        conflicts=[
            ConflictScenario(
                conflict_id="conflict-hiking-biking",
                topic="primary_outdoor_activity",
                conflicting_statements=[
                    "I go to the Catskills every weekend for hiking.",
                    "I no longer go hiking every weekend — moved to biking instead.",
                ],
                correct_resolution="User now prefers biking over hiking.",
                resolution_type="recency",
                introduced_at_messages=[3, 12],
            ),
            ConflictScenario(
                conflict_id="conflict-phone-os",
                topic="phone_os",
                conflicting_statements=[
                    "Implied iOS user (pre-switch context).",
                    "I recently switched from iOS to Android. Using a Pixel 8 now.",
                ],
                correct_resolution="User now uses Android (Pixel 8).",
                resolution_type="explicit_correction",
                introduced_at_messages=[1, 7],
            ),
        ],
        temporal_facts=[
            TemporalFact(
                fact_id="tf-job-engineer",
                description="Software Engineer role",
                value="Software Engineer",
                valid_from="2026-01-01",
                valid_until="2026-01-10",
                query_topic="What was the user's previous job title?",
                should_be_current=False,
            ),
            TemporalFact(
                fact_id="tf-job-senior",
                description="Senior Software Engineer role",
                value="Senior Software Engineer",
                valid_from="2026-01-10",
                valid_until=None,
                query_topic="What is the user's current job title?",
                should_be_current=True,
            ),
            TemporalFact(
                fact_id="tf-hiking-hobby",
                description="Hiking as primary outdoor hobby",
                value="hiking",
                valid_from="2026-01-01",
                valid_until="2026-01-15",
                query_topic="Did the user used to hike?",
                should_be_current=False,
            ),
        ],
        query_relevance_pairs=[
            QueryRelevancePair(
                query_id="qrp-occupation",
                query="What does the user do for a living?",
                expected_relevant_facts=[
                    "Senior Software Engineer",
                    "promoted from Software Engineer",
                ],
                expected_irrelevant_facts=[
                    "uses Android phone",
                    "likes photography",
                ],
            ),
            QueryRelevancePair(
                query_id="qrp-hobbies",
                query="What are the user's current hobbies?",
                expected_relevant_facts=[
                    "biking",
                    "photography",
                ],
                expected_irrelevant_facts=[
                    "software engineer",
                    "New York",
                ],
            ),
        ],
    )


@pytest.fixture
def sample_dataset(
    sample_messages: list[Message],
    sample_ground_truth: GroundTruth,
) -> ConversationDataset:
    """A complete ConversationDataset for integration tests."""
    return ConversationDataset(
        metadata=DatasetMetadata(
            dataset_id="integration-test-001",
            persona_id="test-persona-alice",
            message_count=len(sample_messages),
            simulated_days=15,
            version="1.0.0",
            seed=42,
        ),
        messages=sample_messages,
        ground_truth=sample_ground_truth,
    )


# ---------------------------------------------------------------------------
# NoMemoryAdapter integration tests
# ---------------------------------------------------------------------------


class TestNoMemoryAdapterIntegration:
    """Integration tests using NoMemoryAdapter (lower bound baseline).

    NoMemoryAdapter discards all input and returns nothing on queries.
    With an AlwaysNoJudge:

    - Positive checks (PAS, DBU recency, signal, relevance, CRQ) → NO → fail
    - Negative checks (staleness, noise, stale, irrelevance, TC expired) → NO → pass

    Expected per-dimension scores:
    - PAS: 0.0 (all positive checks fail)
    - DBU: 0.0 (recency=NO → fail for each belief change)
    - MEI: 0.0 (no facts stored → efficiency=0)
    - TC: ~0.6667 (2 of 3 temporal facts have should_be_current=False → pass)
    - CRQ: 0.0 (all resolution checks fail)
    - QRP: 0.5 (relevance=0, irrelevance=1.0 → 0.5*0 + 0.5*1 per pair)
    """

    @pytest.mark.asyncio
    async def test_no_memory_protocol_compliance(self) -> None:
        """NoMemoryAdapter should satisfy the MemoryAdapter protocol."""
        adapter = NoMemoryAdapter()
        assert isinstance(adapter, MemoryAdapter)

    @pytest.mark.asyncio
    async def test_no_memory_positive_dimensions_zero(
        self,
        sample_messages: list[Message],
        sample_ground_truth: GroundTruth,
    ) -> None:
        """NoMemoryAdapter should score 0.0 on purely positive dimensions (PAS, DBU, CRQ)."""
        adapter = NoMemoryAdapter()
        adapter.ingest(sample_messages)

        judge = AlwaysNoJudge()
        engine = ScoringEngine(
            ground_truth=sample_ground_truth,
            judge=judge,  # type: ignore[arg-type]
        )

        result = await engine.run(adapter, "no-memory")  # type: ignore[arg-type]

        assert isinstance(result, BenchmarkResult)
        # Purely positive dimensions should be 0.0
        assert result.cri_result.pas == 0.0
        assert result.cri_result.dbu == 0.0
        assert result.cri_result.crq == 0.0

    @pytest.mark.asyncio
    async def test_no_memory_mei_zero_when_nothing_stored(
        self,
        sample_messages: list[Message],
        sample_ground_truth: GroundTruth,
    ) -> None:
        """MEI should be 0.0 when no facts are stored."""
        adapter = NoMemoryAdapter()
        adapter.ingest(sample_messages)
        judge = AlwaysNoJudge()
        engine = ScoringEngine(
            ground_truth=sample_ground_truth,
            judge=judge,  # type: ignore[arg-type]
        )

        result = await engine.run(adapter, "no-memory")  # type: ignore[arg-type]
        assert result.cri_result.mei == 0.0

    @pytest.mark.asyncio
    async def test_no_memory_tc_zero(
        self,
        sample_messages: list[Message],
        sample_ground_truth: GroundTruth,
    ) -> None:
        """TC = 0 because no-memory stores nothing — no temporal reasoning possible.

        With 0 stored facts, all temporal checks fail regardless of direction.
        """
        adapter = NoMemoryAdapter()
        adapter.ingest(sample_messages)
        judge = AlwaysNoJudge()
        engine = ScoringEngine(
            ground_truth=sample_ground_truth,
            judge=judge,  # type: ignore[arg-type]
        )

        result = await engine.run(adapter, "no-memory")  # type: ignore[arg-type]
        assert result.cri_result.tc == 0.0

    @pytest.mark.asyncio
    async def test_no_memory_qrp_zero(
        self,
        sample_messages: list[Message],
        sample_ground_truth: GroundTruth,
    ) -> None:
        """QRP = 0 because no-memory returns nothing — no useful response.

        With 0 facts returned, both recall and precision are 0 per pair.
        """
        adapter = NoMemoryAdapter()
        adapter.ingest(sample_messages)
        judge = AlwaysNoJudge()
        engine = ScoringEngine(
            ground_truth=sample_ground_truth,
            judge=judge,  # type: ignore[arg-type]
        )

        result = await engine.run(adapter, "no-memory")  # type: ignore[arg-type]
        assert result.cri_result.qrp == 0.0

    @pytest.mark.asyncio
    async def test_no_memory_composite_cri(
        self,
        sample_messages: list[Message],
        sample_ground_truth: GroundTruth,
    ) -> None:
        """Composite CRI is 0.0 for no-memory — all dimensions score 0.

        No facts stored or returned means no dimension can produce a positive score.
        """
        adapter = NoMemoryAdapter()
        adapter.ingest(sample_messages)
        judge = AlwaysNoJudge()
        engine = ScoringEngine(
            ground_truth=sample_ground_truth,
            judge=judge,  # type: ignore[arg-type]
        )

        result = await engine.run(adapter, "no-memory")  # type: ignore[arg-type]
        assert result.cri_result.cri == 0.0

    @pytest.mark.asyncio
    async def test_no_memory_empty_facts(
        self,
        sample_messages: list[Message],
    ) -> None:
        """NoMemoryAdapter should have empty facts after ingestion."""
        adapter = NoMemoryAdapter()
        adapter.ingest(sample_messages)
        assert adapter.get_events() == []
        assert adapter.retrieve("anything") == []

    @pytest.mark.asyncio
    async def test_no_memory_benchmark_result_complete(
        self,
        sample_messages: list[Message],
        sample_ground_truth: GroundTruth,
    ) -> None:
        """BenchmarkResult should be complete even for a no-memory adapter."""
        adapter = NoMemoryAdapter()
        adapter.ingest(sample_messages)
        judge = AlwaysNoJudge()
        engine = ScoringEngine(
            ground_truth=sample_ground_truth,
            judge=judge,  # type: ignore[arg-type]
        )

        result = await engine.run(adapter, "no-memory")  # type: ignore[arg-type]

        # All required fields present
        assert result.run_id != ""
        assert result.adapter_name == "no-memory"
        assert result.started_at is not None
        assert result.completed_at is not None
        assert isinstance(result.cri_result, CRIResult)
        assert isinstance(result.performance_profile, PerformanceProfile)
        assert isinstance(result.judge_log, list)
        # Details should have all 6 dimensions
        assert len(result.cri_result.details) == 6

    @pytest.mark.asyncio
    async def test_no_memory_all_dimension_details_populated(
        self,
        sample_messages: list[Message],
        sample_ground_truth: GroundTruth,
    ) -> None:
        """Each dimension in the details should be a DimensionResult."""
        adapter = NoMemoryAdapter()
        adapter.ingest(sample_messages)
        judge = AlwaysNoJudge()
        engine = ScoringEngine(
            ground_truth=sample_ground_truth,
            judge=judge,  # type: ignore[arg-type]
        )

        result = await engine.run(adapter, "no-memory")  # type: ignore[arg-type]

        for dim_name in ["PAS", "DBU", "MEI", "TC", "CRQ", "QRP"]:
            assert dim_name in result.cri_result.details
            detail = result.cri_result.details[dim_name]
            assert isinstance(detail, DimensionResult)
            assert detail.dimension_name == dim_name

    @pytest.mark.asyncio
    async def test_no_memory_judge_was_called(
        self,
        sample_messages: list[Message],
        sample_ground_truth: GroundTruth,
    ) -> None:
        """The judge should have been called during the pipeline."""
        adapter = NoMemoryAdapter()
        adapter.ingest(sample_messages)
        judge = AlwaysNoJudge()
        engine = ScoringEngine(
            ground_truth=sample_ground_truth,
            judge=judge,  # type: ignore[arg-type]
        )

        result = await engine.run(adapter, "no-memory")  # type: ignore[arg-type]

        # Judge should have been called at least once per check
        assert len(result.judge_log) > 0
        assert result.performance_profile.judge_api_calls > 0

    @pytest.mark.asyncio
    async def test_no_memory_pas_zero_all_checks_fail(
        self,
        sample_messages: list[Message],
        sample_ground_truth: GroundTruth,
    ) -> None:
        """PAS should have zero passed checks with NoMemoryAdapter."""
        adapter = NoMemoryAdapter()
        adapter.ingest(sample_messages)
        judge = AlwaysNoJudge()
        engine = ScoringEngine(
            ground_truth=sample_ground_truth,
            judge=judge,  # type: ignore[arg-type]
        )

        result = await engine.run(adapter, "no-memory")  # type: ignore[arg-type]

        pas_detail = result.cri_result.details["PAS"]
        assert pas_detail.passed_checks == 0
        # total_checks should be > 0 (there are profile dimensions to check)
        assert pas_detail.total_checks > 0

    @pytest.mark.asyncio
    async def test_no_memory_serializable(
        self,
        sample_messages: list[Message],
        sample_ground_truth: GroundTruth,
    ) -> None:
        """Result should be JSON-serializable."""
        adapter = NoMemoryAdapter()
        adapter.ingest(sample_messages)
        judge = AlwaysNoJudge()
        engine = ScoringEngine(
            ground_truth=sample_ground_truth,
            judge=judge,  # type: ignore[arg-type]
        )

        result = await engine.run(adapter, "no-memory")  # type: ignore[arg-type]

        json_str = result.model_dump_json()
        assert isinstance(json_str, str)
        loaded = BenchmarkResult.model_validate_json(json_str)
        assert loaded.cri_result.cri == result.cri_result.cri


# ---------------------------------------------------------------------------
# FullContextAdapter integration tests
# ---------------------------------------------------------------------------


class TestFullContextAdapterIntegration:
    """Integration tests using FullContextAdapter (upper recall bound).

    FullContextAdapter stores every user message verbatim and returns all
    facts for every query. With SmartMockJudge (YES when facts are present),
    positive checks pass but negative checks also fail (YES = failure for
    noise/staleness/irrelevance).
    """

    @pytest.mark.asyncio
    async def test_full_context_protocol_compliance(self) -> None:
        """FullContextAdapter should satisfy the MemoryAdapter protocol."""
        adapter = FullContextAdapter()
        assert isinstance(adapter, MemoryAdapter)

    @pytest.mark.asyncio
    async def test_full_context_stores_user_messages(
        self,
        sample_messages: list[Message],
    ) -> None:
        """FullContextAdapter should store all user messages."""
        adapter = FullContextAdapter()
        adapter.ingest(sample_messages)

        all_facts = adapter.get_events()
        user_msgs = [m for m in sample_messages if m.role == "user"]
        assert len(all_facts) == len(user_msgs)

    @pytest.mark.asyncio
    async def test_full_context_query_returns_all_facts(
        self,
        sample_messages: list[Message],
    ) -> None:
        """FullContextAdapter.retrieve() should return all stored facts."""
        adapter = FullContextAdapter()
        adapter.ingest(sample_messages)

        results = adapter.retrieve("any topic")
        assert len(results) == len(adapter.get_events())

    @pytest.mark.asyncio
    async def test_full_context_scores_higher_than_no_memory_overall(
        self,
        sample_messages: list[Message],
        sample_ground_truth: GroundTruth,
    ) -> None:
        """FullContextAdapter with AlwaysYesJudge should score higher composite
        CRI than NoMemoryAdapter with AlwaysNoJudge.

        Note: FullContextAdapter + AlwaysYesJudge means all positive checks pass
        (good) but all negative checks also "pass" as YES (which means failure
        for the system). This trade-off still produces a higher overall CRI than
        NoMemoryAdapter which gets 0 on all positive checks.
        """
        # FullContextAdapter with AlwaysYesJudge
        full_adapter = FullContextAdapter()
        full_adapter.ingest(sample_messages)
        full_judge = AlwaysYesJudge()
        full_engine = ScoringEngine(
            ground_truth=sample_ground_truth,
            judge=full_judge,  # type: ignore[arg-type]
        )
        full_result = await full_engine.run(full_adapter, "full-context")  # type: ignore[arg-type]

        # NoMemoryAdapter with AlwaysNoJudge
        no_adapter = NoMemoryAdapter()
        no_adapter.ingest(sample_messages)
        no_judge = AlwaysNoJudge()
        no_engine = ScoringEngine(
            ground_truth=sample_ground_truth,
            judge=no_judge,  # type: ignore[arg-type]
        )
        no_result = await no_engine.run(no_adapter, "no-memory")  # type: ignore[arg-type]

        # FullContext should have positive PAS, DBU, CRQ where NoMemory has 0
        assert full_result.cri_result.pas > no_result.cri_result.pas
        assert full_result.cri_result.crq > no_result.cri_result.crq

    @pytest.mark.asyncio
    async def test_full_context_pas_score_positive(
        self,
        sample_messages: list[Message],
        sample_ground_truth: GroundTruth,
    ) -> None:
        """FullContextAdapter should achieve positive PAS score.

        SmartMockJudge returns YES when facts are in the prompt.
        """
        adapter = FullContextAdapter()
        adapter.ingest(sample_messages)
        judge = SmartMockJudge()
        engine = ScoringEngine(
            ground_truth=sample_ground_truth,
            judge=judge,  # type: ignore[arg-type]
        )

        result = await engine.run(adapter, "full-context")  # type: ignore[arg-type]

        # PAS should be > 0 because the adapter stores all user messages
        assert result.cri_result.pas > 0.0

    @pytest.mark.asyncio
    async def test_full_context_crq_score_positive(
        self,
        sample_messages: list[Message],
        sample_ground_truth: GroundTruth,
    ) -> None:
        """FullContextAdapter should achieve positive CRQ score."""
        adapter = FullContextAdapter()
        adapter.ingest(sample_messages)
        judge = SmartMockJudge()
        engine = ScoringEngine(
            ground_truth=sample_ground_truth,
            judge=judge,  # type: ignore[arg-type]
        )

        result = await engine.run(adapter, "full-context")  # type: ignore[arg-type]
        assert result.cri_result.crq > 0.0

    @pytest.mark.asyncio
    async def test_full_context_composite_positive(
        self,
        sample_messages: list[Message],
        sample_ground_truth: GroundTruth,
    ) -> None:
        """Composite CRI should be positive for FullContextAdapter."""
        adapter = FullContextAdapter()
        adapter.ingest(sample_messages)
        judge = SmartMockJudge()
        engine = ScoringEngine(
            ground_truth=sample_ground_truth,
            judge=judge,  # type: ignore[arg-type]
        )

        result = await engine.run(adapter, "full-context")  # type: ignore[arg-type]
        assert result.cri_result.cri > 0.0

    @pytest.mark.asyncio
    async def test_full_context_benchmark_result_complete(
        self,
        sample_messages: list[Message],
        sample_ground_truth: GroundTruth,
    ) -> None:
        """BenchmarkResult should be complete for FullContextAdapter."""
        adapter = FullContextAdapter()
        adapter.ingest(sample_messages)
        judge = SmartMockJudge()
        engine = ScoringEngine(
            ground_truth=sample_ground_truth,
            judge=judge,  # type: ignore[arg-type]
        )

        result = await engine.run(adapter, "full-context")  # type: ignore[arg-type]

        assert isinstance(result, BenchmarkResult)
        assert result.run_id != ""
        assert result.adapter_name == "full-context"
        assert isinstance(result.cri_result, CRIResult)
        assert isinstance(result.performance_profile, PerformanceProfile)
        assert isinstance(result.judge_log, list)
        assert len(result.judge_log) > 0

        # All dimensions should be present in details
        for dim_name in ["PAS", "DBU", "MEI", "TC", "CRQ", "QRP"]:
            assert dim_name in result.cri_result.details

    @pytest.mark.asyncio
    async def test_full_context_judge_called_many_times(
        self,
        sample_messages: list[Message],
        sample_ground_truth: GroundTruth,
    ) -> None:
        """The judge should be called multiple times across all dimensions."""
        adapter = FullContextAdapter()
        adapter.ingest(sample_messages)
        judge = SmartMockJudge()
        engine = ScoringEngine(
            ground_truth=sample_ground_truth,
            judge=judge,  # type: ignore[arg-type]
        )

        result = await engine.run(adapter, "full-context")  # type: ignore[arg-type]

        # With our ground truth, we expect many judge calls:
        # PAS: 6 checks, DBU: 6 checks, MEI: coverage, TC: 3, CRQ: 2, QRP: many
        assert result.performance_profile.judge_api_calls > 10

    @pytest.mark.asyncio
    async def test_full_context_dimension_details_have_checks(
        self,
        sample_messages: list[Message],
        sample_ground_truth: GroundTruth,
    ) -> None:
        """Each dimension detail should have non-zero total_checks."""
        adapter = FullContextAdapter()
        adapter.ingest(sample_messages)
        judge = SmartMockJudge()
        engine = ScoringEngine(
            ground_truth=sample_ground_truth,
            judge=judge,  # type: ignore[arg-type]
        )

        result = await engine.run(adapter, "full-context")  # type: ignore[arg-type]

        for dim_name in ["PAS", "DBU", "CRQ"]:
            detail = result.cri_result.details[dim_name]
            assert detail.total_checks > 0, f"{dim_name} should have checks"

    @pytest.mark.asyncio
    async def test_full_context_dbu_has_belief_change_details(
        self,
        sample_messages: list[Message],
        sample_ground_truth: GroundTruth,
    ) -> None:
        """DBU details should contain per-belief-change records."""
        adapter = FullContextAdapter()
        adapter.ingest(sample_messages)
        judge = SmartMockJudge()
        engine = ScoringEngine(
            ground_truth=sample_ground_truth,
            judge=judge,  # type: ignore[arg-type]
        )

        result = await engine.run(adapter, "full-context")  # type: ignore[arg-type]

        dbu_detail = result.cri_result.details["DBU"]
        # There are 3 belief changes in ground truth
        assert dbu_detail.total_checks == 3
        assert len(dbu_detail.details) == 3

        # Each detail should have belief_change info
        for detail in dbu_detail.details:
            assert "belief_change" in detail
            assert "old_value" in detail
            assert "new_value" in detail
            assert "recency_verdict" in detail
            assert "staleness_verdict" in detail
            assert "passed" in detail

    @pytest.mark.asyncio
    async def test_full_context_pas_has_profile_details(
        self,
        sample_messages: list[Message],
        sample_ground_truth: GroundTruth,
    ) -> None:
        """PAS details should contain per-profile-dimension records."""
        adapter = FullContextAdapter()
        adapter.ingest(sample_messages)
        judge = SmartMockJudge()
        engine = ScoringEngine(
            ground_truth=sample_ground_truth,
            judge=judge,  # type: ignore[arg-type]
        )

        result = await engine.run(adapter, "full-context")  # type: ignore[arg-type]

        pas_detail = result.cri_result.details["PAS"]
        # 5 profile dims: occupation, location, age, hobbies (2 values), phone_os = 6 checks
        assert pas_detail.total_checks == 6
        assert len(pas_detail.details) == 6

        for detail in pas_detail.details:
            assert "check_id" in detail
            assert "expected_value" in detail
            assert "verdict" in detail
            assert "passed" in detail

    @pytest.mark.asyncio
    async def test_full_context_serializable(
        self,
        sample_messages: list[Message],
        sample_ground_truth: GroundTruth,
    ) -> None:
        """Result from FullContextAdapter should be JSON-serializable."""
        adapter = FullContextAdapter()
        adapter.ingest(sample_messages)
        judge = SmartMockJudge()
        engine = ScoringEngine(
            ground_truth=sample_ground_truth,
            judge=judge,  # type: ignore[arg-type]
        )

        result = await engine.run(adapter, "full-context")  # type: ignore[arg-type]

        json_str = result.model_dump_json()
        assert isinstance(json_str, str)
        loaded = BenchmarkResult.model_validate_json(json_str)
        assert loaded.cri_result.cri == result.cri_result.cri
        assert loaded.adapter_name == "full-context"

    @pytest.mark.asyncio
    async def test_full_context_dbu_behavior_with_smart_judge(
        self,
        sample_messages: list[Message],
        sample_ground_truth: GroundTruth,
    ) -> None:
        """SmartMockJudge returns YES for both recency and staleness checks
        (since facts are present). DBU requires recency=YES AND staleness=NO,
        so with SmartMockJudge all DBU checks fail (staleness=YES means failure).
        """
        adapter = FullContextAdapter()
        adapter.ingest(sample_messages)
        judge = SmartMockJudge()
        engine = ScoringEngine(
            ground_truth=sample_ground_truth,
            judge=judge,  # type: ignore[arg-type]
        )

        result = await engine.run(adapter, "full-context")  # type: ignore[arg-type]

        # SmartMockJudge → YES for both recency and staleness → DBU fails
        # (staleness=YES means old value is still asserted as current → failure)
        assert result.cri_result.dbu == 0.0


# ---------------------------------------------------------------------------
# Comparative integration tests
# ---------------------------------------------------------------------------


class TestAdapterComparison:
    """Tests comparing NoMemoryAdapter and FullContextAdapter behavior."""

    @pytest.mark.asyncio
    async def test_both_adapters_produce_valid_benchmark_results(
        self,
        sample_messages: list[Message],
        sample_ground_truth: GroundTruth,
    ) -> None:
        """Both adapters should produce structurally valid BenchmarkResults."""
        for adapter_class, judge_class, name in [
            (NoMemoryAdapter, AlwaysNoJudge, "no-memory"),
            (FullContextAdapter, SmartMockJudge, "full-context"),
        ]:
            adapter = adapter_class()
            adapter.ingest(sample_messages)
            judge = judge_class()
            engine = ScoringEngine(
                ground_truth=sample_ground_truth,
                judge=judge,  # type: ignore[arg-type]
            )
            result = await engine.run(adapter, name)  # type: ignore[arg-type]

            assert isinstance(result, BenchmarkResult)
            assert 0.0 <= result.cri_result.cri <= 1.0
            assert 0.0 <= result.cri_result.pas <= 1.0
            assert 0.0 <= result.cri_result.dbu <= 1.0
            assert 0.0 <= result.cri_result.mei <= 1.0
            assert 0.0 <= result.cri_result.tc <= 1.0
            assert 0.0 <= result.cri_result.crq <= 1.0
            assert 0.0 <= result.cri_result.qrp <= 1.0
            assert len(result.cri_result.details) == 6

    @pytest.mark.asyncio
    async def test_full_context_pas_better_than_no_memory(
        self,
        sample_messages: list[Message],
        sample_ground_truth: GroundTruth,
    ) -> None:
        """FullContextAdapter should have strictly better PAS than NoMemoryAdapter.

        PAS is a purely positive metric — the adapter that stores facts will
        always outperform one that stores nothing.
        """
        # FullContext with SmartMockJudge (YES when facts present)
        full_adapter = FullContextAdapter()
        full_adapter.ingest(sample_messages)
        full_judge = SmartMockJudge()
        full_engine = ScoringEngine(
            ground_truth=sample_ground_truth,
            judge=full_judge,  # type: ignore[arg-type]
        )
        full_result = await full_engine.run(full_adapter, "full-context")  # type: ignore[arg-type]

        # NoMemory with AlwaysNoJudge (NO for everything)
        no_adapter = NoMemoryAdapter()
        no_adapter.ingest(sample_messages)
        no_judge = AlwaysNoJudge()
        no_engine = ScoringEngine(
            ground_truth=sample_ground_truth,
            judge=no_judge,  # type: ignore[arg-type]
        )
        no_result = await no_engine.run(no_adapter, "no-memory")  # type: ignore[arg-type]

        assert full_result.cri_result.pas > no_result.cri_result.pas
        assert no_result.cri_result.pas == 0.0

    @pytest.mark.asyncio
    async def test_concurrent_runs_independent(
        self,
        sample_messages: list[Message],
        sample_ground_truth: GroundTruth,
    ) -> None:
        """Multiple benchmark runs should be independent (different run_ids)."""
        adapter = FullContextAdapter()
        adapter.ingest(sample_messages)
        judge = SmartMockJudge()
        engine = ScoringEngine(
            ground_truth=sample_ground_truth,
            judge=judge,  # type: ignore[arg-type]
        )

        result1 = await engine.run(adapter, "run-1")  # type: ignore[arg-type]
        result2 = await engine.run(adapter, "run-2")  # type: ignore[arg-type]

        assert result1.run_id != result2.run_id
        assert result1.adapter_name == "run-1"
        assert result2.adapter_name == "run-2"

    @pytest.mark.asyncio
    async def test_scores_within_valid_range(
        self,
        sample_messages: list[Message],
        sample_ground_truth: GroundTruth,
    ) -> None:
        """All scores should be within [0.0, 1.0] for both adapters."""
        for adapter_class, judge_class in [
            (NoMemoryAdapter, AlwaysNoJudge),
            (FullContextAdapter, SmartMockJudge),
            (FullContextAdapter, AlwaysYesJudge),
        ]:
            adapter = adapter_class()
            adapter.ingest(sample_messages)
            judge = judge_class()
            engine = ScoringEngine(
                ground_truth=sample_ground_truth,
                judge=judge,  # type: ignore[arg-type]
            )
            result = await engine.run(adapter, "test")  # type: ignore[arg-type]

            for dim_name in ["PAS", "DBU", "MEI", "TC", "CRQ", "QRP"]:
                detail = result.cri_result.details[dim_name]
                assert 0.0 <= detail.score <= 1.0, f"{dim_name} score {detail.score} out of range for {adapter_class.__name__}/{judge_class.__name__}"
            assert 0.0 <= result.cri_result.cri <= 1.0


# ---------------------------------------------------------------------------
# Pipeline with canonical dataset
# ---------------------------------------------------------------------------


class TestCanonicalDatasetPipeline:
    """Integration tests using a canonical dataset from the datasets folder."""

    @pytest.fixture
    def canonical_ground_truth(self) -> GroundTruth | None:
        """Load the persona-1-base ground truth if available."""
        gt_path = Path(__file__).parent.parent / "datasets" / "canonical" / "persona-1-base" / "ground_truth.json"
        if not gt_path.exists():
            return None
        data = json.loads(gt_path.read_text())
        return GroundTruth(**data)

    @pytest.fixture
    def canonical_messages(self) -> list[Message] | None:
        """Load persona-1-base conversation messages if available."""
        conv_path = Path(__file__).parent.parent / "datasets" / "canonical" / "persona-1-base" / "conversations.jsonl"
        if not conv_path.exists():
            return None
        messages = []
        for line in conv_path.read_text().splitlines():
            if line.strip():
                data = json.loads(line)
                messages.append(Message(**data))
        return messages

    @pytest.mark.asyncio
    async def test_canonical_no_memory_positive_dims_zero(
        self,
        canonical_ground_truth: GroundTruth | None,
        canonical_messages: list[Message] | None,
    ) -> None:
        """NoMemoryAdapter should score 0.0 on positive dimensions with canonical dataset."""
        if canonical_ground_truth is None or canonical_messages is None:
            pytest.skip("Canonical dataset not available")

        adapter = NoMemoryAdapter()
        adapter.ingest(canonical_messages)
        judge = AlwaysNoJudge()
        engine = ScoringEngine(
            ground_truth=canonical_ground_truth,
            judge=judge,  # type: ignore[arg-type]
        )

        result = await engine.run(adapter, "no-memory-canonical")  # type: ignore[arg-type]
        assert result.cri_result.pas == 0.0
        assert result.cri_result.dbu == 0.0
        assert result.cri_result.crq == 0.0

    @pytest.mark.asyncio
    async def test_canonical_full_context_positive(
        self,
        canonical_ground_truth: GroundTruth | None,
        canonical_messages: list[Message] | None,
    ) -> None:
        """FullContextAdapter should score > 0 on positive dimensions with canonical dataset."""
        if canonical_ground_truth is None or canonical_messages is None:
            pytest.skip("Canonical dataset not available")

        adapter = FullContextAdapter()
        adapter.ingest(canonical_messages)
        judge = SmartMockJudge()
        engine = ScoringEngine(
            ground_truth=canonical_ground_truth,
            judge=judge,  # type: ignore[arg-type]
        )

        result = await engine.run(adapter, "full-context-canonical")  # type: ignore[arg-type]
        assert result.cri_result.pas > 0.0
        assert result.cri_result.crq > 0.0
        # Should have many checks given the canonical dataset is rich
        total_checks = sum(d.total_checks for d in result.cri_result.details.values())
        assert total_checks > 10

    @pytest.mark.asyncio
    async def test_canonical_dataset_dimensions_exercised(
        self,
        canonical_ground_truth: GroundTruth | None,
        canonical_messages: list[Message] | None,
    ) -> None:
        """All dimensions should be exercised with the canonical dataset."""
        if canonical_ground_truth is None or canonical_messages is None:
            pytest.skip("Canonical dataset not available")

        adapter = FullContextAdapter()
        adapter.ingest(canonical_messages)
        judge = SmartMockJudge()
        engine = ScoringEngine(
            ground_truth=canonical_ground_truth,
            judge=judge,  # type: ignore[arg-type]
        )

        result = await engine.run(adapter, "full-context-canonical")  # type: ignore[arg-type]

        # All dimensions should have been evaluated
        for dim_name in ["PAS", "DBU", "MEI", "TC", "CRQ", "QRP"]:
            assert dim_name in result.cri_result.details
            detail = result.cri_result.details[dim_name]
            assert detail.total_checks > 0, f"Dimension {dim_name} should have checks in canonical dataset"


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge case tests for unusual inputs."""

    @pytest.mark.asyncio
    async def test_empty_ground_truth_minimal(self) -> None:
        """Engine should handle a minimal ground truth with empty annotations."""
        gt = GroundTruth(
            final_profile={},
            changes=[],
            noise_examples=[],
            signal_examples=[],
            conflicts=[],
            temporal_facts=[],
            query_relevance_pairs=[],
        )
        adapter = NoMemoryAdapter()
        judge = AlwaysNoJudge()
        engine = ScoringEngine(
            ground_truth=gt,
            judge=judge,  # type: ignore[arg-type]
        )

        result = await engine.run(adapter, "empty-gt")  # type: ignore[arg-type]
        assert isinstance(result, BenchmarkResult)
        # With no checks to fail, scores default to 0.0 or 1.0 depending on dimension logic
        assert 0.0 <= result.cri_result.cri <= 1.0

    @pytest.mark.asyncio
    async def test_no_messages_ingested(
        self,
        sample_ground_truth: GroundTruth,
    ) -> None:
        """Running the pipeline without ingesting messages should still work."""
        adapter = FullContextAdapter()
        # Don't ingest anything
        judge = AlwaysNoJudge()
        engine = ScoringEngine(
            ground_truth=sample_ground_truth,
            judge=judge,  # type: ignore[arg-type]
        )

        result = await engine.run(adapter, "empty-ingest")  # type: ignore[arg-type]
        assert isinstance(result, BenchmarkResult)
        # No facts stored → positive checks fail
        assert result.cri_result.pas == 0.0

    @pytest.mark.asyncio
    async def test_adapter_reset_between_runs(
        self,
        sample_messages: list[Message],
        sample_ground_truth: GroundTruth,
    ) -> None:
        """Resetting an adapter between runs should yield independent results."""
        adapter = FullContextAdapter()

        # First run: ingest messages
        adapter.ingest(sample_messages)
        judge1 = SmartMockJudge()
        engine1 = ScoringEngine(
            ground_truth=sample_ground_truth,
            judge=judge1,  # type: ignore[arg-type]
        )
        result1 = await engine1.run(adapter, "run-1")  # type: ignore[arg-type]

        # Reset adapter
        adapter.reset()
        assert len(adapter.get_events()) == 0

        # Second run: no messages ingested after reset
        judge2 = AlwaysNoJudge()
        engine2 = ScoringEngine(
            ground_truth=sample_ground_truth,
            judge=judge2,  # type: ignore[arg-type]
        )
        result2 = await engine2.run(adapter, "run-2-reset")  # type: ignore[arg-type]

        # After reset, PAS should be 0 (no facts to match)
        assert result2.cri_result.pas == 0.0
        # First run should have had positive PAS
        assert result1.cri_result.pas > 0.0

    @pytest.mark.asyncio
    async def test_single_message_dataset(
        self,
        sample_ground_truth: GroundTruth,
    ) -> None:
        """Pipeline should work with a single message."""
        messages = [
            Message(
                message_id=1,
                role="user",
                content="I am a software engineer in New York.",
                timestamp="2026-01-01T10:00:00Z",
            ),
        ]
        adapter = FullContextAdapter()
        adapter.ingest(messages)
        judge = SmartMockJudge()
        engine = ScoringEngine(
            ground_truth=sample_ground_truth,
            judge=judge,  # type: ignore[arg-type]
        )

        result = await engine.run(adapter, "single-msg")  # type: ignore[arg-type]
        assert isinstance(result, BenchmarkResult)
        assert len(adapter.get_events()) == 1

    @pytest.mark.asyncio
    async def test_assistant_only_messages(
        self,
        sample_ground_truth: GroundTruth,
    ) -> None:
        """FullContextAdapter should store nothing if only assistant messages."""
        messages = [
            Message(
                message_id=1,
                role="assistant",
                content="Hello! How can I help you?",
                timestamp="2026-01-01T10:00:00Z",
            ),
            Message(
                message_id=2,
                role="assistant",
                content="I can help with many things.",
                timestamp="2026-01-01T10:01:00Z",
            ),
        ]
        adapter = FullContextAdapter()
        adapter.ingest(messages)

        # No user messages → no facts stored
        assert len(adapter.get_events()) == 0

        judge = AlwaysNoJudge()
        engine = ScoringEngine(
            ground_truth=sample_ground_truth,
            judge=judge,  # type: ignore[arg-type]
        )
        result = await engine.run(adapter, "assistant-only")  # type: ignore[arg-type]
        assert isinstance(result, BenchmarkResult)
        assert result.cri_result.pas == 0.0
