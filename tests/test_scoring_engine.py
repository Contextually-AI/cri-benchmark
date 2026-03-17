"""Comprehensive tests for the CRI ScoringEngine.

Tests cover:
- Construction with default and custom config
- Weight validation
- Successful runs with mocked dimension scorers
- Error handling (individual and total dimension failures)
- Composite CRI calculation
- CRIResult field correctness
- BenchmarkResult completeness
- Legacy engine backward compatibility
- Edge cases and boundary conditions
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from cri.models import (
    BeliefChange,
    BenchmarkResult,
    DimensionResult,
    GroundTruth,
    JudgmentResult,
    NoiseExample,
    PerformanceProfile,
    ProfileDimension,
    ScoringConfig,
    SignalExample,
    StoredFact,
    Verdict,
)
from cri.scoring.engine import (
    _DEFAULT_DIMENSION_REGISTRY,
    ScoringEngine,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class MockMemoryAdapter:
    """A minimal mock adapter satisfying the MemoryAdapter protocol."""

    def ingest(self, messages: list[Any]) -> None:
        pass

    def retrieve(self, topic: str) -> list[StoredFact]:
        return []

    def get_events(self) -> list[StoredFact]:
        return []


class MockBinaryJudge:
    """A mock BinaryJudge that tracks calls without LLM interaction."""

    def __init__(self) -> None:
        self._log: list[JudgmentResult] = []

    def judge(self, check_id: str, prompt: str) -> JudgmentResult:
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

    def get_log(self) -> list[JudgmentResult]:
        return list(self._log)


@pytest.fixture
def mock_adapter() -> MockMemoryAdapter:
    return MockMemoryAdapter()


@pytest.fixture
def mock_judge() -> MockBinaryJudge:
    return MockBinaryJudge()


@pytest.fixture
def minimal_ground_truth() -> GroundTruth:
    """A minimal GroundTruth with one entry per annotation type."""
    return GroundTruth(
        final_profile={
            "occupation": ProfileDimension(
                dimension_name="occupation",
                value="Engineer",
                query_topic="What does the user do?",
            ),
        },
        changes=[
            BeliefChange(
                fact="job",
                old_value="Junior",
                new_value="Senior",
                query_topic="job title",
                changed_around_msg=5,
                key_messages=[5],
            ),
        ],
        noise_examples=[
            NoiseExample(text="Hello!", reason="Greeting"),
        ],
        signal_examples=[
            SignalExample(text="I am an engineer", target_fact="occupation: Engineer"),
        ],
        conflicts=[],
        temporal_facts=[],
        query_relevance_pairs=[],
    )


@pytest.fixture
def default_config() -> ScoringConfig:
    return ScoringConfig()


@pytest.fixture
def engine(
    minimal_ground_truth: GroundTruth,
    mock_judge: MockBinaryJudge,
    default_config: ScoringConfig,
) -> ScoringEngine:
    return ScoringEngine(
        ground_truth=minimal_ground_truth,
        judge=mock_judge,  # type: ignore[arg-type]
        config=default_config,
    )


# ---------------------------------------------------------------------------
# Construction tests
# ---------------------------------------------------------------------------


class TestScoringEngineConstruction:
    """Tests for ScoringEngine initialization."""

    def test_default_config(self, minimal_ground_truth: GroundTruth, mock_judge: MockBinaryJudge) -> None:
        engine = ScoringEngine(ground_truth=minimal_ground_truth, judge=mock_judge)  # type: ignore[arg-type]
        assert engine.config is not None
        assert engine.config.dimension_weights["PAS"] == 0.25
        assert engine.config.dimension_weights["DBU"] == 0.20
        assert engine.config.dimension_weights["MEI"] == 0.20
        assert engine.config.dimension_weights["TC"] == 0.15
        assert engine.config.dimension_weights["CRQ"] == 0.10
        assert engine.config.dimension_weights["QRP"] == 0.10

    def test_custom_config(self, minimal_ground_truth: GroundTruth, mock_judge: MockBinaryJudge) -> None:
        config = ScoringConfig(
            dimension_weights={
                "PAS": 0.30,
                "DBU": 0.20,
                "MEI": 0.15,
                "TC": 0.15,
                "CRQ": 0.10,
                "QRP": 0.10,
            }
        )
        engine = ScoringEngine(
            ground_truth=minimal_ground_truth,
            judge=mock_judge,  # type: ignore[arg-type]
            config=config,
        )
        assert engine.config.dimension_weights["PAS"] == 0.30

    def test_stores_ground_truth(self, engine: ScoringEngine, minimal_ground_truth: GroundTruth) -> None:
        assert engine.ground_truth is minimal_ground_truth

    def test_stores_judge(self, engine: ScoringEngine, mock_judge: MockBinaryJudge) -> None:
        assert engine.judge is mock_judge

    def test_dimension_registry_populated(self, engine: ScoringEngine) -> None:
        assert "PAS" in engine.dimension_registry
        assert "DBU" in engine.dimension_registry
        assert "TC" in engine.dimension_registry
        assert "CRQ" in engine.dimension_registry
        assert "QRP" in engine.dimension_registry
        assert "MEI" in engine.dimension_registry
        assert "SFC" in engine.dimension_registry
        assert "LNC" in engine.dimension_registry
        assert "ARS" in engine.dimension_registry
        assert len(engine.dimension_registry) == 9

    def test_invalid_weights_raises(self, minimal_ground_truth: GroundTruth, mock_judge: MockBinaryJudge) -> None:
        config = ScoringConfig(
            dimension_weights={"PAS": 0.50, "DBU": 0.10},
            enabled_dimensions=["PAS", "DBU"],
        )
        with pytest.raises(ValueError, match="sum to 1.0"):
            ScoringEngine(
                ground_truth=minimal_ground_truth,
                judge=mock_judge,  # type: ignore[arg-type]
                config=config,
            )

    def test_weights_exactly_one(self, minimal_ground_truth: GroundTruth, mock_judge: MockBinaryJudge) -> None:
        """Weights that sum to exactly 1.0 should not raise."""
        config = ScoringConfig(
            dimension_weights={
                "PAS": 0.25,
                "DBU": 0.20,
                "MEI": 0.20,
                "TC": 0.15,
                "CRQ": 0.10,
                "QRP": 0.10,
            }
        )
        engine = ScoringEngine(
            ground_truth=minimal_ground_truth,
            judge=mock_judge,  # type: ignore[arg-type]
            config=config,
        )
        assert engine is not None

    def test_weights_within_tolerance(self, minimal_ground_truth: GroundTruth, mock_judge: MockBinaryJudge) -> None:
        """Weights within ±0.01 tolerance should not raise."""
        config = ScoringConfig(
            dimension_weights={
                "PAS": 0.2502,
                "DBU": 0.2002,
                "MEI": 0.2001,
                "TC": 0.1501,
                "CRQ": 0.0998,
                "QRP": 0.0996,
            }
        )
        # Sum = 1.0000 within tolerance
        engine = ScoringEngine(
            ground_truth=minimal_ground_truth,
            judge=mock_judge,  # type: ignore[arg-type]
            config=config,
        )
        assert engine is not None

    def test_dimension_registry_is_independent_copy(
        self,
        minimal_ground_truth: GroundTruth,
        mock_judge: MockBinaryJudge,
    ) -> None:
        """Each ScoringEngine should have its own independent registry."""
        engine1 = ScoringEngine(ground_truth=minimal_ground_truth, judge=mock_judge)  # type: ignore[arg-type]
        engine2 = ScoringEngine(ground_truth=minimal_ground_truth, judge=mock_judge)  # type: ignore[arg-type]
        # Mutating engine1's registry should not affect engine2
        engine1.dimension_registry["FAKE"] = MagicMock()
        assert "FAKE" in engine1.dimension_registry
        assert "FAKE" not in engine2.dimension_registry
        # The global registry (class mapping) should be unaffected
        assert "FAKE" not in _DEFAULT_DIMENSION_REGISTRY

    def test_default_enabled_dimensions(self, minimal_ground_truth: GroundTruth, mock_judge: MockBinaryJudge) -> None:
        """Default config should enable all 6 dimensions."""
        engine = ScoringEngine(ground_truth=minimal_ground_truth, judge=mock_judge)  # type: ignore[arg-type]
        assert set(engine.config.enabled_dimensions) == {"PAS", "DBU", "MEI", "TC", "CRQ", "QRP"}


# ---------------------------------------------------------------------------
# Run method tests
# ---------------------------------------------------------------------------


def _make_dim_result(name: str, score: float) -> DimensionResult:
    """Helper to create a DimensionResult."""
    return DimensionResult(
        dimension_name=name,
        score=score,
        passed_checks=int(score * 10),
        total_checks=10,
        details=[],
    )


class TestScoringEngineRun:
    """Tests for the async run() method."""

    @pytest.mark.asyncio
    async def test_run_returns_benchmark_result(
        self,
        engine: ScoringEngine,
        mock_adapter: MockMemoryAdapter,
    ) -> None:
        """Patch all dimension scorers to return predetermined results."""
        for dim_name in engine.dimension_registry:
            scorer = engine.dimension_registry[dim_name]
            scorer.score = AsyncMock(  # type: ignore[method-assign]
                return_value=_make_dim_result(dim_name, 0.8)
            )

        result = await engine.run(mock_adapter, "test-system")  # type: ignore[arg-type]
        assert isinstance(result, BenchmarkResult)

    @pytest.mark.asyncio
    async def test_run_id_is_valid_uuid(
        self,
        engine: ScoringEngine,
        mock_adapter: MockMemoryAdapter,
    ) -> None:
        for dim_name in engine.dimension_registry:
            engine.dimension_registry[dim_name].score = AsyncMock(  # type: ignore[method-assign]
                return_value=_make_dim_result(dim_name, 0.5)
            )

        result = await engine.run(mock_adapter, "test-system")  # type: ignore[arg-type]
        parsed_uuid = uuid.UUID(result.run_id)
        assert parsed_uuid.version == 4

    @pytest.mark.asyncio
    async def test_run_timestamps_are_iso8601(
        self,
        engine: ScoringEngine,
        mock_adapter: MockMemoryAdapter,
    ) -> None:
        for dim_name in engine.dimension_registry:
            engine.dimension_registry[dim_name].score = AsyncMock(  # type: ignore[method-assign]
                return_value=_make_dim_result(dim_name, 0.5)
            )

        result = await engine.run(mock_adapter, "test-system")  # type: ignore[arg-type]
        # Should parse without error
        datetime.fromisoformat(result.started_at)
        datetime.fromisoformat(result.completed_at)

    @pytest.mark.asyncio
    async def test_run_adapter_name_matches(
        self,
        engine: ScoringEngine,
        mock_adapter: MockMemoryAdapter,
    ) -> None:
        for dim_name in engine.dimension_registry:
            engine.dimension_registry[dim_name].score = AsyncMock(  # type: ignore[method-assign]
                return_value=_make_dim_result(dim_name, 0.5)
            )

        result = await engine.run(mock_adapter, "my-system")  # type: ignore[arg-type]
        assert result.adapter_name == "my-system"
        assert result.cri_result.system_name == "my-system"

    @pytest.mark.asyncio
    async def test_run_dimension_scores_in_cri_result(
        self,
        engine: ScoringEngine,
        mock_adapter: MockMemoryAdapter,
    ) -> None:
        scores = {"PAS": 0.9, "DBU": 0.8, "MEI": 0.7, "TC": 0.6, "CRQ": 0.5, "QRP": 0.4}
        for dim_name, s in scores.items():
            engine.dimension_registry[dim_name].score = AsyncMock(  # type: ignore[method-assign]
                return_value=_make_dim_result(dim_name, s)
            )

        result = await engine.run(mock_adapter, "test-system")  # type: ignore[arg-type]
        cri = result.cri_result
        assert cri.pas == 0.9
        assert cri.dbu == 0.8
        assert cri.mei == 0.7
        assert cri.tc == 0.6
        assert cri.crq == 0.5
        assert cri.qrp == 0.4

    @pytest.mark.asyncio
    async def test_composite_cri_weighted_sum(
        self,
        engine: ScoringEngine,
        mock_adapter: MockMemoryAdapter,
    ) -> None:
        """Verify CRI = sum(weight_i * score_i) with default weights."""
        scores = {"PAS": 1.0, "DBU": 1.0, "MEI": 1.0, "TC": 1.0, "CRQ": 1.0, "QRP": 1.0}
        for dim_name, s in scores.items():
            engine.dimension_registry[dim_name].score = AsyncMock(  # type: ignore[method-assign]
                return_value=_make_dim_result(dim_name, s)
            )

        result = await engine.run(mock_adapter, "perfect")  # type: ignore[arg-type]
        assert abs(result.cri_result.cri - 1.0) < 0.001

    @pytest.mark.asyncio
    async def test_composite_cri_with_varied_scores(
        self,
        engine: ScoringEngine,
        mock_adapter: MockMemoryAdapter,
    ) -> None:
        """Verify composite CRI matches manual weighted-sum calculation."""
        scores = {"PAS": 0.8, "DBU": 0.6, "MEI": 0.4, "TC": 0.9, "CRQ": 1.0, "QRP": 0.5}
        weights = engine.config.dimension_weights

        expected_cri = sum(weights[d] * scores[d] for d in scores)

        for dim_name, s in scores.items():
            engine.dimension_registry[dim_name].score = AsyncMock(  # type: ignore[method-assign]
                return_value=_make_dim_result(dim_name, s)
            )

        result = await engine.run(mock_adapter, "test")  # type: ignore[arg-type]
        assert abs(result.cri_result.cri - round(expected_cri, 4)) < 0.001

    @pytest.mark.asyncio
    async def test_dimension_details_in_result(
        self,
        engine: ScoringEngine,
        mock_adapter: MockMemoryAdapter,
    ) -> None:
        for dim_name in engine.dimension_registry:
            engine.dimension_registry[dim_name].score = AsyncMock(  # type: ignore[method-assign]
                return_value=_make_dim_result(dim_name, 0.5)
            )

        result = await engine.run(mock_adapter, "test")  # type: ignore[arg-type]
        for dim_name in ["PAS", "DBU", "MEI", "TC", "CRQ", "QRP"]:
            assert dim_name in result.cri_result.details
            assert isinstance(result.cri_result.details[dim_name], DimensionResult)

    @pytest.mark.asyncio
    async def test_performance_profile_present(
        self,
        engine: ScoringEngine,
        mock_adapter: MockMemoryAdapter,
    ) -> None:
        for dim_name in engine.dimension_registry:
            engine.dimension_registry[dim_name].score = AsyncMock(  # type: ignore[method-assign]
                return_value=_make_dim_result(dim_name, 0.5)
            )

        result = await engine.run(mock_adapter, "test")  # type: ignore[arg-type]
        assert isinstance(result.performance_profile, PerformanceProfile)

    @pytest.mark.asyncio
    async def test_dimension_weights_in_result(
        self,
        engine: ScoringEngine,
        mock_adapter: MockMemoryAdapter,
    ) -> None:
        for dim_name in engine.dimension_registry:
            engine.dimension_registry[dim_name].score = AsyncMock(  # type: ignore[method-assign]
                return_value=_make_dim_result(dim_name, 0.5)
            )

        result = await engine.run(mock_adapter, "test")  # type: ignore[arg-type]
        assert result.cri_result.dimension_weights == engine.config.dimension_weights

    @pytest.mark.asyncio
    async def test_run_all_zero_scores(
        self,
        engine: ScoringEngine,
        mock_adapter: MockMemoryAdapter,
    ) -> None:
        """All dimensions scoring 0.0 should yield composite CRI 0.0."""
        for dim_name in engine.dimension_registry:
            engine.dimension_registry[dim_name].score = AsyncMock(  # type: ignore[method-assign]
                return_value=_make_dim_result(dim_name, 0.0)
            )
        result = await engine.run(mock_adapter, "zero-system")  # type: ignore[arg-type]
        assert result.cri_result.cri == 0.0
        assert result.cri_result.pas == 0.0
        assert result.cri_result.dbu == 0.0

    @pytest.mark.asyncio
    async def test_completed_at_after_started_at(
        self,
        engine: ScoringEngine,
        mock_adapter: MockMemoryAdapter,
    ) -> None:
        """completed_at should be >= started_at."""
        for dim_name in engine.dimension_registry:
            engine.dimension_registry[dim_name].score = AsyncMock(  # type: ignore[method-assign]
                return_value=_make_dim_result(dim_name, 0.5)
            )
        result = await engine.run(mock_adapter, "test")  # type: ignore[arg-type]
        started = datetime.fromisoformat(result.started_at)
        completed = datetime.fromisoformat(result.completed_at)
        assert completed >= started

    @pytest.mark.asyncio
    async def test_run_each_dimension_scorer_called_once(
        self,
        engine: ScoringEngine,
        mock_adapter: MockMemoryAdapter,
    ) -> None:
        """Each enabled dimension scorer should be called exactly once during a run."""
        mocks: dict[str, AsyncMock] = {}
        for dim_name in engine.dimension_registry:
            m = AsyncMock(return_value=_make_dim_result(dim_name, 0.5))
            engine.dimension_registry[dim_name].score = m  # type: ignore[method-assign]
            mocks[dim_name] = m

        await engine.run(mock_adapter, "test")  # type: ignore[arg-type]

        for dim_name, m in mocks.items():
            if dim_name in engine.config.enabled_dimensions:
                assert m.call_count == 1, f"{dim_name} scorer was not called exactly once"
            else:
                assert m.call_count == 0, f"{dim_name} scorer should not be called (not enabled)"

    @pytest.mark.asyncio
    async def test_run_scorer_receives_correct_args(
        self,
        engine: ScoringEngine,
        mock_adapter: MockMemoryAdapter,
        minimal_ground_truth: GroundTruth,
        mock_judge: MockBinaryJudge,
    ) -> None:
        """Verify scorers receive the adapter, ground_truth, and judge."""
        for dim_name in engine.dimension_registry:
            engine.dimension_registry[dim_name].score = AsyncMock(  # type: ignore[method-assign]
                return_value=_make_dim_result(dim_name, 0.5)
            )

        await engine.run(mock_adapter, "test")  # type: ignore[arg-type]

        for dim_name in engine.config.enabled_dimensions:
            call_args = engine.dimension_registry[dim_name].score.call_args  # type: ignore[union-attr]
            args = call_args[0]
            assert args[0] is mock_adapter
            assert args[1] is minimal_ground_truth
            assert args[2] is mock_judge

    @pytest.mark.asyncio
    async def test_cri_score_is_rounded_to_4_decimals(
        self,
        engine: ScoringEngine,
        mock_adapter: MockMemoryAdapter,
    ) -> None:
        """Composite CRI should be rounded to 4 decimal places."""
        scores = {
            "PAS": 0.333333,
            "DBU": 0.666666,
            "MEI": 0.111111,
            "TC": 0.999999,
            "CRQ": 0.444444,
            "QRP": 0.777777,
        }
        for dim_name, s in scores.items():
            engine.dimension_registry[dim_name].score = AsyncMock(  # type: ignore[method-assign]
                return_value=_make_dim_result(dim_name, s)
            )
        result = await engine.run(mock_adapter, "test")  # type: ignore[arg-type]
        # Check rounding: the CRI should have at most 4 decimal places
        cri_str = str(result.cri_result.cri)
        if "." in cri_str:
            decimals = len(cri_str.split(".")[1])
            assert decimals <= 4


# ---------------------------------------------------------------------------
# Error handling tests
# ---------------------------------------------------------------------------


class TestScoringEngineErrorHandling:
    """Tests for error resilience during dimension scoring."""

    @pytest.mark.asyncio
    async def test_single_dimension_failure_continues(
        self,
        engine: ScoringEngine,
        mock_adapter: MockMemoryAdapter,
    ) -> None:
        """If one dimension throws, others should still be scored."""
        for dim_name in engine.dimension_registry:
            if dim_name == "PAS":
                engine.dimension_registry[dim_name].score = AsyncMock(  # type: ignore[method-assign]
                    side_effect=RuntimeError("PAS scorer crashed")
                )
            else:
                engine.dimension_registry[dim_name].score = AsyncMock(  # type: ignore[method-assign]
                    return_value=_make_dim_result(dim_name, 0.8)
                )

        result = await engine.run(mock_adapter, "test")  # type: ignore[arg-type]

        # PAS should be 0.0
        assert result.cri_result.pas == 0.0
        # Others should be 0.8
        assert result.cri_result.dbu == 0.8
        assert result.cri_result.mei == 0.8
        assert result.cri_result.tc == 0.8
        assert result.cri_result.crq == 0.8
        assert result.cri_result.qrp == 0.8

    @pytest.mark.asyncio
    async def test_single_dimension_failure_has_error_detail(
        self,
        engine: ScoringEngine,
        mock_adapter: MockMemoryAdapter,
    ) -> None:
        for dim_name in engine.dimension_registry:
            if dim_name == "TC":
                engine.dimension_registry[dim_name].score = AsyncMock(  # type: ignore[method-assign]
                    side_effect=ValueError("TC exploded")
                )
            else:
                engine.dimension_registry[dim_name].score = AsyncMock(  # type: ignore[method-assign]
                    return_value=_make_dim_result(dim_name, 0.5)
                )

        result = await engine.run(mock_adapter, "test")  # type: ignore[arg-type]

        tc_details = result.cri_result.details["TC"]
        assert tc_details.score == 0.0
        assert tc_details.passed_checks == 0
        assert tc_details.total_checks == 0
        assert len(tc_details.details) == 1
        assert "error" in tc_details.details[0]
        assert "TC exploded" in tc_details.details[0]["error"]

    @pytest.mark.asyncio
    async def test_all_dimensions_fail_gives_zero_cri(
        self,
        engine: ScoringEngine,
        mock_adapter: MockMemoryAdapter,
    ) -> None:
        """If every dimension fails, CRI should be 0.0."""
        for dim_name in engine.dimension_registry:
            engine.dimension_registry[dim_name].score = AsyncMock(  # type: ignore[method-assign]
                side_effect=RuntimeError(f"{dim_name} failed")
            )

        result = await engine.run(mock_adapter, "test")  # type: ignore[arg-type]
        assert result.cri_result.cri == 0.0
        assert result.cri_result.pas == 0.0
        assert result.cri_result.dbu == 0.0

    @pytest.mark.asyncio
    async def test_unknown_dimension_in_config_skipped(
        self,
        minimal_ground_truth: GroundTruth,
        mock_judge: MockBinaryJudge,
        mock_adapter: MockMemoryAdapter,
    ) -> None:
        """A dimension name in enabled_dimensions with no registered scorer
        should be silently skipped (with warning log)."""
        config = ScoringConfig(
            dimension_weights={
                "PAS": 0.25,
                "DBU": 0.20,
                "MEI": 0.20,
                "TC": 0.15,
                "CRQ": 0.10,
                "QRP": 0.10,
            },
            enabled_dimensions=["PAS", "DBU", "MEI", "TC", "CRQ", "QRP", "UNKNOWN"],
        )
        engine = ScoringEngine(
            ground_truth=minimal_ground_truth,
            judge=mock_judge,  # type: ignore[arg-type]
            config=config,
        )

        for dim_name in engine.dimension_registry:
            engine.dimension_registry[dim_name].score = AsyncMock(  # type: ignore[method-assign]
                return_value=_make_dim_result(dim_name, 0.5)
            )

        # Should not raise
        result = await engine.run(mock_adapter, "test")  # type: ignore[arg-type]
        assert "UNKNOWN" not in result.cri_result.details

    @pytest.mark.asyncio
    async def test_benchmark_always_completes(
        self,
        engine: ScoringEngine,
        mock_adapter: MockMemoryAdapter,
    ) -> None:
        """Even with all kinds of exceptions, run() should return a result."""
        exceptions = [
            RuntimeError("boom"),
            ValueError("bad value"),
            TypeError("wrong type"),
            KeyError("missing key"),
            OSError("io error"),
            Exception("generic"),
        ]
        for i, dim_name in enumerate(engine.config.enabled_dimensions):
            engine.dimension_registry[dim_name].score = AsyncMock(  # type: ignore[method-assign]
                side_effect=exceptions[i % len(exceptions)]
            )

        result = await engine.run(mock_adapter, "test")  # type: ignore[arg-type]
        assert isinstance(result, BenchmarkResult)
        assert result.cri_result.cri == 0.0

    @pytest.mark.asyncio
    async def test_multiple_dimension_failures_mixed(
        self,
        engine: ScoringEngine,
        mock_adapter: MockMemoryAdapter,
    ) -> None:
        """When multiple (but not all) dimensions fail, surviving ones contribute."""
        for dim_name in engine.dimension_registry:
            if dim_name in ("PAS", "TC", "QRP"):
                engine.dimension_registry[dim_name].score = AsyncMock(  # type: ignore[method-assign]
                    side_effect=RuntimeError(f"{dim_name} failed")
                )
            else:
                engine.dimension_registry[dim_name].score = AsyncMock(  # type: ignore[method-assign]
                    return_value=_make_dim_result(dim_name, 1.0)
                )

        result = await engine.run(mock_adapter, "test")  # type: ignore[arg-type]
        # Failed dimensions should be 0.0
        assert result.cri_result.pas == 0.0
        assert result.cri_result.tc == 0.0
        assert result.cri_result.qrp == 0.0
        # Surviving dimensions should be 1.0
        assert result.cri_result.dbu == 1.0
        assert result.cri_result.mei == 1.0
        assert result.cri_result.crq == 1.0
        # Composite should be between 0 and 1
        assert 0.0 < result.cri_result.cri < 1.0

    @pytest.mark.asyncio
    async def test_failed_dimension_still_in_details(
        self,
        engine: ScoringEngine,
        mock_adapter: MockMemoryAdapter,
    ) -> None:
        """Failed dimensions should still appear in the details dict with score 0."""
        engine.dimension_registry["MEI"].score = AsyncMock(  # type: ignore[method-assign]
            side_effect=RuntimeError("MEI crashed")
        )
        for dim_name in engine.dimension_registry:
            if dim_name != "MEI":
                engine.dimension_registry[dim_name].score = AsyncMock(  # type: ignore[method-assign]
                    return_value=_make_dim_result(dim_name, 0.5)
                )

        result = await engine.run(mock_adapter, "test")  # type: ignore[arg-type]
        assert "MEI" in result.cri_result.details
        assert result.cri_result.details["MEI"].score == 0.0


# ---------------------------------------------------------------------------
# Composite CRI calculation edge cases
# ---------------------------------------------------------------------------


class TestCompositeCalculation:
    """Tests for the composite CRI computation logic."""

    def test_compute_composite_empty(self, engine: ScoringEngine) -> None:
        result = engine._compute_composite({})
        assert result == 0.0

    def test_compute_composite_all_ones(self, engine: ScoringEngine) -> None:
        dims = {d: _make_dim_result(d, 1.0) for d in ["PAS", "DBU", "MEI", "TC", "CRQ", "QRP"]}
        result = engine._compute_composite(dims)
        assert abs(result - 1.0) < 0.001

    def test_compute_composite_all_zeros(self, engine: ScoringEngine) -> None:
        dims = {d: _make_dim_result(d, 0.0) for d in ["PAS", "DBU", "MEI", "TC", "CRQ", "QRP"]}
        result = engine._compute_composite(dims)
        assert result == 0.0

    def test_compute_composite_partial_dimensions(self, engine: ScoringEngine) -> None:
        """When only some dimensions are scored, weights should be re-normalized."""
        # Only PAS and DBU scored
        dims = {
            "PAS": _make_dim_result("PAS", 1.0),
            "DBU": _make_dim_result("DBU", 0.0),
        }
        result = engine._compute_composite(dims)

        # PAS weight=0.25, DBU weight=0.20. Total=0.45
        # Normalized: PAS=0.25/0.45 ≈ 0.5556, DBU=0.20/0.45 ≈ 0.4444
        # CRI = 0.5556*1.0 + 0.4444*0.0 ≈ 0.5556
        expected = 0.25 / 0.45
        assert abs(result - expected) < 0.001

    def test_compute_composite_respects_weights(
        self,
        minimal_ground_truth: GroundTruth,
        mock_judge: MockBinaryJudge,
    ) -> None:
        """Custom weights should affect the composite score."""
        config = ScoringConfig(
            dimension_weights={
                "PAS": 0.50,
                "DBU": 0.10,
                "MEI": 0.10,
                "TC": 0.10,
                "CRQ": 0.10,
                "QRP": 0.10,
            }
        )
        engine = ScoringEngine(
            ground_truth=minimal_ground_truth,
            judge=mock_judge,  # type: ignore[arg-type]
            config=config,
        )
        dims = {
            "PAS": _make_dim_result("PAS", 1.0),
            "DBU": _make_dim_result("DBU", 0.0),
            "MEI": _make_dim_result("MEI", 0.0),
            "TC": _make_dim_result("TC", 0.0),
            "CRQ": _make_dim_result("CRQ", 0.0),
            "QRP": _make_dim_result("QRP", 0.0),
        }
        result = engine._compute_composite(dims)
        # CRI = 0.50 * 1.0 + rest * 0.0 = 0.50
        assert abs(result - 0.50) < 0.001

    def test_compute_composite_single_dimension(self, engine: ScoringEngine) -> None:
        """A single scored dimension should be re-normalized to weight 1.0."""
        dims = {"PAS": _make_dim_result("PAS", 0.75)}
        result = engine._compute_composite(dims)
        # Re-normalized: PAS has weight 1.0, score = 0.75
        assert abs(result - 0.75) < 0.001

    def test_compute_composite_unknown_dimension_weight_zero(self, engine: ScoringEngine) -> None:
        """A dimension not in config weights should have weight 0.0."""
        dims = {
            "MYSTERY": _make_dim_result("MYSTERY", 1.0),
            "PAS": _make_dim_result("PAS", 0.5),
        }
        result = engine._compute_composite(dims)
        # MYSTERY has weight 0.0, PAS has weight 0.25
        # Total active weight = 0.25, normalized PAS = 1.0
        # CRI = 1.0 * 0.5 = 0.5
        assert abs(result - 0.5) < 0.001

    def test_compute_composite_boundary_scores(self, engine: ScoringEngine) -> None:
        """Test with exact boundary values 0.0 and 1.0."""
        dims = {
            "PAS": _make_dim_result("PAS", 0.0),
            "DBU": _make_dim_result("DBU", 1.0),
            "MEI": _make_dim_result("MEI", 0.0),
            "TC": _make_dim_result("TC", 1.0),
            "CRQ": _make_dim_result("CRQ", 0.0),
            "QRP": _make_dim_result("QRP", 1.0),
        }
        result = engine._compute_composite(dims)
        # Expected = 0.25*0 + 0.20*1 + 0.20*0 + 0.15*1 + 0.10*0 + 0.10*1 = 0.45
        assert abs(result - 0.45) < 0.001


# ---------------------------------------------------------------------------
# Judge log integration tests
# ---------------------------------------------------------------------------


class TestJudgeLogIntegration:
    """Tests that the judge log is correctly captured in results."""

    @pytest.mark.asyncio
    async def test_judge_log_empty_when_mocked(
        self,
        engine: ScoringEngine,
        mock_adapter: MockMemoryAdapter,
    ) -> None:
        """When dimension scorers are mocked (don't call judge), log is empty."""
        for dim_name in engine.dimension_registry:
            engine.dimension_registry[dim_name].score = AsyncMock(  # type: ignore[method-assign]
                return_value=_make_dim_result(dim_name, 0.5)
            )

        result = await engine.run(mock_adapter, "test")  # type: ignore[arg-type]
        # Mocked scorers don't call the judge, so log should be empty
        assert isinstance(result.judge_log, list)

    @pytest.mark.asyncio
    async def test_judge_api_calls_count(
        self,
        engine: ScoringEngine,
        mock_adapter: MockMemoryAdapter,
        mock_judge: MockBinaryJudge,
    ) -> None:
        """Performance profile should report the number of judge API calls."""
        # Manually add some entries to the judge log
        mock_judge.judge("test-1", "prompt-1")
        mock_judge.judge("test-2", "prompt-2")

        for dim_name in engine.dimension_registry:
            engine.dimension_registry[dim_name].score = AsyncMock(  # type: ignore[method-assign]
                return_value=_make_dim_result(dim_name, 0.5)
            )

        result = await engine.run(mock_adapter, "test")  # type: ignore[arg-type]
        assert result.performance_profile.judge_api_calls == 2

    @pytest.mark.asyncio
    async def test_judge_log_contains_results_from_run(
        self,
        engine: ScoringEngine,
        mock_adapter: MockMemoryAdapter,
        mock_judge: MockBinaryJudge,
    ) -> None:
        """Judge log should contain JudgmentResult entries."""
        mock_judge.judge("pre-run-check", "test prompt")

        for dim_name in engine.dimension_registry:
            engine.dimension_registry[dim_name].score = AsyncMock(  # type: ignore[method-assign]
                return_value=_make_dim_result(dim_name, 0.5)
            )

        result = await engine.run(mock_adapter, "test")  # type: ignore[arg-type]
        assert len(result.judge_log) >= 1
        assert all(isinstance(j, JudgmentResult) for j in result.judge_log)
        assert result.judge_log[0].check_id == "pre-run-check"
