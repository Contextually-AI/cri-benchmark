"""Tests for the SSI (Scale Sensitivity Index) meta-metric.

Covers:
- Identical CRI scores across scales → SSI = 1.0 (perfect stability)
- Varying CRI scores across scales → SSI < 1.0
- Function signature and return type
- Edge cases: empty messages, single scale
"""

from __future__ import annotations

import pytest

from cri.models import (
    DimensionResult,
    GroundTruth,
    JudgmentResult,
    Message,
    ScoringConfig,
    StoredFact,
    Verdict,
)
from cri.scoring.ssi import compute_ssi

# ---------------------------------------------------------------------------
# Test helpers — lightweight mocks
# ---------------------------------------------------------------------------


class MockBinaryJudge:
    """Mock binary judge satisfying BinaryJudge's interface.

    Returns a fixed verdict for all checks and records a call log,
    compatible with ScoringEngine's ``judge.get_log()`` call.
    """

    def __init__(self, default_verdict: Verdict = Verdict.YES) -> None:
        self.default_verdict = default_verdict
        self._log: list[JudgmentResult] = []
        self.num_runs: int = 3

    async def judge(self, check_id: str, prompt: str) -> JudgmentResult:
        result = JudgmentResult(
            check_id=check_id,
            verdict=self.default_verdict,
            votes=[self.default_verdict] * 3,
            unanimous=True,
            prompt=prompt,
            raw_responses=[self.default_verdict.value] * 3,
        )
        self._log.append(result)
        return result

    async def judge_across_chunks(self, check_id: str, stored_facts: list[str], prompt_builder) -> JudgmentResult:
        prompt = prompt_builder(stored_facts)
        return await self.judge(check_id, prompt)

    async def judge_coverage(self, check_id: str, prompt: str) -> set[int]:
        return set()

    def get_log(self) -> list[JudgmentResult]:
        return list(self._log)


class MockAdapter:
    """Minimal adapter satisfying the MemoryAdapter protocol."""

    def __init__(self) -> None:
        self._facts: list[StoredFact] = []

    def ingest(self, messages: list[Message]) -> None:
        for msg in messages:
            if msg.role == "user":
                self._facts.append(StoredFact(text=msg.content))

    def retrieve(self, topic: str) -> list[StoredFact]:
        return list(self._facts)

    def get_events(self) -> list[StoredFact]:
        return list(self._facts)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_messages(n: int = 4) -> list[Message]:
    """Create a list of n test messages alternating user/assistant."""
    messages = []
    for i in range(n):
        role = "user" if i % 2 == 0 else "assistant"
        messages.append(
            Message(
                message_id=i + 1,
                role=role,
                content=f"Message {i + 1} content.",
                timestamp=f"2026-01-{i + 1:02d}T10:00:00Z",
                session_id="sess-001",
                day=i + 1,
            )
        )
    return messages


def _make_simple_ground_truth() -> GroundTruth:
    """Create a minimal GroundTruth with an empty profile."""
    return GroundTruth(
        final_profile={},
        changes=[],
        noise_examples=[],
        signal_examples=[],
        conflicts=[],
        temporal_facts=[],
        query_relevance_pairs=[],
    )


def _make_config() -> ScoringConfig:
    """Create a minimal ScoringConfig with a single dimension to keep tests fast."""
    return ScoringConfig.from_dimensions(["PAS"])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSSIReturnType:
    """Verify the function signature and return type."""

    @pytest.mark.asyncio
    async def test_returns_dimension_result(self) -> None:
        """compute_ssi returns a DimensionResult."""
        ground_truth = _make_simple_ground_truth()
        messages = _make_messages(4)

        result = await compute_ssi(
            adapter_factory=MockAdapter,
            messages=messages,
            ground_truth=ground_truth,
            judge_factory=MockBinaryJudge,
            config=_make_config(),
        )

        assert isinstance(result, DimensionResult)

    @pytest.mark.asyncio
    async def test_dimension_name_is_ssi(self) -> None:
        ground_truth = _make_simple_ground_truth()
        messages = _make_messages(4)

        result = await compute_ssi(
            adapter_factory=MockAdapter,
            messages=messages,
            ground_truth=ground_truth,
            judge_factory=MockBinaryJudge,
            config=_make_config(),
        )

        assert result.dimension_name == "SSI"

    @pytest.mark.asyncio
    async def test_score_in_valid_range(self) -> None:
        ground_truth = _make_simple_ground_truth()
        messages = _make_messages(4)

        result = await compute_ssi(
            adapter_factory=MockAdapter,
            messages=messages,
            ground_truth=ground_truth,
            judge_factory=MockBinaryJudge,
            config=_make_config(),
        )

        assert 0.0 <= result.score <= 1.0


class TestSSIEmptyMessages:
    """Edge case: empty message list → vacuous SSI = 1.0."""

    @pytest.mark.asyncio
    async def test_empty_messages_returns_one(self) -> None:
        ground_truth = _make_simple_ground_truth()

        result = await compute_ssi(
            adapter_factory=MockAdapter,
            messages=[],
            ground_truth=ground_truth,
            judge_factory=MockBinaryJudge,
            config=_make_config(),
        )

        assert result.score == 1.0
        assert result.dimension_name == "SSI"

    @pytest.mark.asyncio
    async def test_empty_messages_zero_checks(self) -> None:
        ground_truth = _make_simple_ground_truth()

        result = await compute_ssi(
            adapter_factory=MockAdapter,
            messages=[],
            ground_truth=ground_truth,
            judge_factory=MockBinaryJudge,
            config=_make_config(),
        )

        assert result.passed_checks == 0
        assert result.total_checks == 0
        assert result.details == []


class TestSSIPerfectStability:
    """Identical scores across all scales → SSI = 1.0."""

    @pytest.mark.asyncio
    async def test_identical_scores_give_ssi_one(self) -> None:
        """When CRI is the same at all scale points, degradation_rate=0 and SSI=1.0."""
        ground_truth = _make_simple_ground_truth()
        messages = _make_messages(8)

        # A judge that always returns YES produces the same score regardless of scale.
        result = await compute_ssi(
            adapter_factory=MockAdapter,
            messages=messages,
            ground_truth=ground_truth,
            judge_factory=MockBinaryJudge,
            config=_make_config(),
        )

        # With empty final_profile, PAS returns 0.0 at every scale → no degradation
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_total_checks_equals_number_of_scales(self) -> None:
        ground_truth = _make_simple_ground_truth()
        messages = _make_messages(4)
        scales = [0.25, 0.50, 0.75, 1.00]

        result = await compute_ssi(
            adapter_factory=MockAdapter,
            messages=messages,
            ground_truth=ground_truth,
            judge_factory=MockBinaryJudge,
            config=_make_config(),
            scales=scales,
        )

        assert result.total_checks == len(scales)


class TestSSISingleScale:
    """Edge case: single scale → no degradation possible, SSI=1.0."""

    @pytest.mark.asyncio
    async def test_single_scale_returns_one(self) -> None:
        ground_truth = _make_simple_ground_truth()
        messages = _make_messages(4)

        result = await compute_ssi(
            adapter_factory=MockAdapter,
            messages=messages,
            ground_truth=ground_truth,
            judge_factory=MockBinaryJudge,
            config=_make_config(),
            scales=[1.0],
        )

        # With a single scale, smallest == largest → degradation_rate = 0 → SSI = 1.0
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_single_scale_total_checks_one(self) -> None:
        ground_truth = _make_simple_ground_truth()
        messages = _make_messages(4)

        result = await compute_ssi(
            adapter_factory=MockAdapter,
            messages=messages,
            ground_truth=ground_truth,
            judge_factory=MockBinaryJudge,
            config=_make_config(),
            scales=[1.0],
        )

        assert result.total_checks == 1


class TestSSIDetails:
    """Detail records are structured correctly."""

    @pytest.mark.asyncio
    async def test_details_contain_scale_records(self) -> None:
        ground_truth = _make_simple_ground_truth()
        messages = _make_messages(4)
        scales = [0.5, 1.0]

        result = await compute_ssi(
            adapter_factory=MockAdapter,
            messages=messages,
            ground_truth=ground_truth,
            judge_factory=MockBinaryJudge,
            config=_make_config(),
            scales=scales,
        )

        scale_records = [d for d in result.details if not d.get("summary")]
        assert len(scale_records) == len(scales)
        for record in scale_records:
            assert "scale" in record
            assert "message_count" in record
            assert "cri" in record

    @pytest.mark.asyncio
    async def test_details_contain_summary(self) -> None:
        ground_truth = _make_simple_ground_truth()
        messages = _make_messages(4)

        result = await compute_ssi(
            adapter_factory=MockAdapter,
            messages=messages,
            ground_truth=ground_truth,
            judge_factory=MockBinaryJudge,
            config=_make_config(),
        )

        summaries = [d for d in result.details if d.get("summary")]
        assert len(summaries) == 1
        summary = summaries[0]
        assert "cri_at_smallest_scale" in summary
        assert "cri_at_largest_scale" in summary
        assert "degradation_rate" in summary
        assert "ssi" in summary

    @pytest.mark.asyncio
    async def test_custom_scales_reflected_in_details(self) -> None:
        ground_truth = _make_simple_ground_truth()
        messages = _make_messages(4)
        custom_scales = [0.5, 1.0]

        result = await compute_ssi(
            adapter_factory=MockAdapter,
            messages=messages,
            ground_truth=ground_truth,
            judge_factory=MockBinaryJudge,
            config=_make_config(),
            scales=custom_scales,
        )

        scale_values = {d["scale"] for d in result.details if not d.get("summary")}
        assert scale_values == {0.5, 1.0}

    @pytest.mark.asyncio
    async def test_fresh_adapter_used_per_scale(self) -> None:
        """A new adapter is created for each scale — factory is called multiple times."""
        ground_truth = _make_simple_ground_truth()
        messages = _make_messages(4)
        call_count = 0

        def counting_factory() -> MockAdapter:
            nonlocal call_count
            call_count += 1
            return MockAdapter()

        scales = [0.25, 0.5, 0.75, 1.0]
        await compute_ssi(
            adapter_factory=counting_factory,
            messages=messages,
            ground_truth=ground_truth,
            judge_factory=MockBinaryJudge,
            config=_make_config(),
            scales=scales,
        )

        assert call_count == len(scales)

    @pytest.mark.asyncio
    async def test_fresh_judge_used_per_scale(self) -> None:
        """A new judge is created for each scale — factory is called multiple times."""
        ground_truth = _make_simple_ground_truth()
        messages = _make_messages(4)
        call_count = 0

        def counting_judge_factory() -> MockBinaryJudge:
            nonlocal call_count
            call_count += 1
            return MockBinaryJudge()

        scales = [0.5, 1.0]
        await compute_ssi(
            adapter_factory=MockAdapter,
            messages=messages,
            ground_truth=ground_truth,
            judge_factory=counting_judge_factory,
            config=_make_config(),
            scales=scales,
        )

        assert call_count == len(scales)
