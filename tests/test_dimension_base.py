"""Tests for MetricDimension abstract base class.

Verifies the contract, subclass validation, and behavior of the
MetricDimension ABC defined in cri.scoring.dimensions.base.
"""

from __future__ import annotations

import pytest

from cri.models import DimensionResult, GroundTruth
from cri.scoring.dimensions.base import MetricDimension

# ---------------------------------------------------------------------------
# Fixtures — minimal GroundTruth for testing
# ---------------------------------------------------------------------------


@pytest.fixture()
def minimal_ground_truth() -> GroundTruth:
    """Return a minimal GroundTruth instance for test plumbing."""
    return GroundTruth(
        final_profile={},
        changes=[],
        noise_examples=[],
        signal_examples=[],
        conflicts=[],
        temporal_facts=[],
        query_relevance_pairs=[],
    )


# ---------------------------------------------------------------------------
# Concrete subclass for testing
# ---------------------------------------------------------------------------


class StubDimension(MetricDimension):
    """Concrete stub dimension for testing the ABC contract."""

    name = "STUB"
    description = "A stub dimension for testing purposes."

    async def score(self, adapter, ground_truth, judge) -> DimensionResult:
        return DimensionResult(
            dimension_name=self.name,
            score=0.75,
            passed_checks=3,
            total_checks=4,
            details=[
                {"check": "c1", "passed": True},
                {"check": "c2", "passed": True},
                {"check": "c3", "passed": True},
                {"check": "c4", "passed": False},
            ],
        )


# ---------------------------------------------------------------------------
# Tests — ABC contract
# ---------------------------------------------------------------------------


class TestMetricDimensionABC:
    """Test that MetricDimension enforces the abstract contract."""

    def test_cannot_instantiate_abc_directly(self) -> None:
        """MetricDimension itself cannot be instantiated."""
        with pytest.raises(TypeError):
            MetricDimension()  # type: ignore[abstract]

    def test_must_implement_score(self) -> None:
        """A subclass without score() cannot be instantiated."""
        with pytest.raises(TypeError):
            # Defining a class without implementing score should work,
            # but instantiating it should fail.
            class IncompleteDimension(MetricDimension):
                name = "INCOMPLETE"
                description = "Missing score method."

            IncompleteDimension()  # type: ignore[abstract]

    def test_concrete_subclass_instantiates(self) -> None:
        """A properly defined concrete subclass can be instantiated."""
        dim = StubDimension()
        assert dim.name == "STUB"
        assert dim.description == "A stub dimension for testing purposes."


# ---------------------------------------------------------------------------
# Tests — class-level attribute validation
# ---------------------------------------------------------------------------


class TestMetricDimensionValidation:
    """Test __init_subclass__ validation of name and description."""

    def test_missing_name_raises(self) -> None:
        """A concrete subclass without 'name' raises TypeError at definition time."""
        with pytest.raises(TypeError, match="name"):

            class NoName(MetricDimension):
                description = "Has description but no name."

                async def score(self, adapter, ground_truth, judge) -> DimensionResult:
                    return DimensionResult(
                        dimension_name="X",
                        score=0.0,
                        passed_checks=0,
                        total_checks=0,
                    )

    def test_missing_description_raises(self) -> None:
        """A concrete subclass without 'description' raises TypeError at definition time."""
        with pytest.raises(TypeError, match="description"):

            class NoDesc(MetricDimension):
                name = "NODESC"

                async def score(self, adapter, ground_truth, judge) -> DimensionResult:
                    return DimensionResult(
                        dimension_name="NODESC",
                        score=0.0,
                        passed_checks=0,
                        total_checks=0,
                    )

    def test_empty_name_raises(self) -> None:
        """An empty string for 'name' raises TypeError at definition time."""
        with pytest.raises(TypeError, match="name"):

            class EmptyName(MetricDimension):
                name = ""
                description = "Non-empty description."

                async def score(self, adapter, ground_truth, judge) -> DimensionResult:
                    return DimensionResult(
                        dimension_name="",
                        score=0.0,
                        passed_checks=0,
                        total_checks=0,
                    )

    def test_empty_description_raises(self) -> None:
        """An empty string for 'description' raises TypeError at definition time."""
        with pytest.raises(TypeError, match="description"):

            class EmptyDesc(MetricDimension):
                name = "EDESC"
                description = ""

                async def score(self, adapter, ground_truth, judge) -> DimensionResult:
                    return DimensionResult(
                        dimension_name="EDESC",
                        score=0.0,
                        passed_checks=0,
                        total_checks=0,
                    )

    def test_intermediate_abstract_subclass_allowed(self) -> None:
        """An intermediate abstract subclass (no score impl) is allowed
        without name/description because it is still abstract."""

        class IntermediateDimension(MetricDimension):
            """Still abstract — does not implement score."""

        # It should have abstractmethods so it can't be instantiated
        with pytest.raises(TypeError):
            IntermediateDimension()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# Tests — score method
# ---------------------------------------------------------------------------


class TestMetricDimensionScore:
    """Test the score method on a concrete subclass."""

    @pytest.mark.asyncio
    async def test_score_returns_dimension_result(self, minimal_ground_truth: GroundTruth) -> None:
        """score() returns a DimensionResult with expected values."""
        dim = StubDimension()
        result = await dim.score(
            adapter=None,  # type: ignore[arg-type]
            ground_truth=minimal_ground_truth,
            judge=None,  # type: ignore[arg-type]
        )

        assert isinstance(result, DimensionResult)
        assert result.dimension_name == "STUB"
        assert result.score == 0.75
        assert result.passed_checks == 3
        assert result.total_checks == 4
        assert len(result.details) == 4

    @pytest.mark.asyncio
    async def test_score_details_contain_check_info(self, minimal_ground_truth: GroundTruth) -> None:
        """The details list contains per-check records."""
        dim = StubDimension()
        result = await dim.score(
            adapter=None,  # type: ignore[arg-type]
            ground_truth=minimal_ground_truth,
            judge=None,  # type: ignore[arg-type]
        )
        passed_count = sum(1 for d in result.details if d.get("passed"))
        assert passed_count == 3


# ---------------------------------------------------------------------------
# Tests — repr
# ---------------------------------------------------------------------------


class TestMetricDimensionRepr:
    """Test the __repr__ method."""

    def test_repr(self) -> None:
        dim = StubDimension()
        assert repr(dim) == "StubDimension(name='STUB')"


# ---------------------------------------------------------------------------
# Tests — multiple concrete subclasses coexist
# ---------------------------------------------------------------------------


class TestMultipleDimensions:
    """Verify that multiple concrete subclasses can coexist independently."""

    def test_two_dimensions_with_different_names(self) -> None:

        class AlphaDimension(MetricDimension):
            name = "ALPHA"
            description = "First test dimension."

            async def score(self, adapter, ground_truth, judge) -> DimensionResult:
                return DimensionResult(
                    dimension_name=self.name,
                    score=1.0,
                    passed_checks=5,
                    total_checks=5,
                )

        class BetaDimension(MetricDimension):
            name = "BETA"
            description = "Second test dimension."

            async def score(self, adapter, ground_truth, judge) -> DimensionResult:
                return DimensionResult(
                    dimension_name=self.name,
                    score=0.5,
                    passed_checks=2,
                    total_checks=4,
                )

        alpha = AlphaDimension()
        beta = BetaDimension()

        assert alpha.name == "ALPHA"
        assert beta.name == "BETA"
        assert alpha.name != beta.name
