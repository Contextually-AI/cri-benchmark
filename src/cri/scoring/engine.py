"""Scoring engine for the CRI Benchmark.

:class:`ScoringEngine` orchestrates the full evaluation pipeline. It uses
:class:`~cri.judge.BinaryJudge` with
:class:`~cri.scoring.dimensions.base.MetricDimension` scorers to evaluate
a memory system across the CRI dimensions and compute a weighted
composite CRI score.

It accepts a :class:`~cri.adapter.MemoryAdapter` and returns a fully
populated :class:`~cri.models.BenchmarkResult`.
"""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime

from cri.adapter import MemoryAdapter
from cri.judge import BinaryJudge
from cri.models import (
    BenchmarkResult,
    CRIResult,
    DimensionResult,
    GroundTruth,
    PerformanceProfile,
    ScoringConfig,
)
from cri.scoring.dimensions.ars import ARSDimension
from cri.scoring.dimensions.base import MetricDimension
from cri.scoring.dimensions.crq import CRQDimension
from cri.scoring.dimensions.dbu import DBUDimension
from cri.scoring.dimensions.lnc import LNCDimension
from cri.scoring.dimensions.mei import MEIDimension
from cri.scoring.dimensions.pas import ProfileAccuracyScore
from cri.scoring.dimensions.qrp import QRPDimension
from cri.scoring.dimensions.sfc import SFCDimension
from cri.scoring.dimensions.tc import TCDimension

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default dimension registry — maps dimension code to MetricDimension scorer
# ---------------------------------------------------------------------------

_DEFAULT_DIMENSION_REGISTRY: dict[str, type[MetricDimension]] = {
    "PAS": ProfileAccuracyScore,
    "DBU": DBUDimension,
    "TC": TCDimension,
    "CRQ": CRQDimension,
    "QRP": QRPDimension,
    "MEI": MEIDimension,
    "SFC": SFCDimension,
    "LNC": LNCDimension,
    "ARS": ARSDimension,
}


def _create_default_registry() -> dict[str, MetricDimension]:
    """Create a fresh set of dimension scorer instances.

    Returns a new dict with new instances each time to avoid shared
    mutable state between engine instances.
    """
    return {name: cls() for name, cls in _DEFAULT_DIMENSION_REGISTRY.items()}


# ---------------------------------------------------------------------------
# Primary scoring engine — ScoringEngine
# ---------------------------------------------------------------------------


class ScoringEngine:
    """Orchestrates the full CRI evaluation pipeline.

    The engine iterates through the enabled dimensions specified in the
    :class:`~cri.models.ScoringConfig`, runs each dimension scorer against
    the provided adapter, and computes a weighted composite CRI score.

    **Error resilience**: If any individual dimension scorer raises an
    exception, the engine catches it, logs a warning, records a score of
    ``0.0`` for that dimension, and continues evaluating the remaining
    dimensions.  The benchmark always completes.

    Args:
        ground_truth: The expected outcomes for the benchmark dataset.
        judge: A :class:`~cri.judge.BinaryJudge` instance for semantic
            evaluation of adapter responses.
        config: Scoring configuration controlling weights and enabled
            dimensions.  Defaults to :class:`ScoringConfig` with standard
            weights for all 9 dimensions.

    Raises:
        ValueError: If the configured dimension weights do not sum to
            approximately 1.0 (tolerance ±0.01).

    Example::

        from cri.scoring.engine import ScoringEngine
        from cri.judge import BinaryJudge
        from cri.models import ScoringConfig

        engine = ScoringEngine(
            ground_truth=ground_truth,
            judge=BinaryJudge(),
            config=ScoringConfig(),
        )
        result = await engine.run(adapter, system_name="my-memory-system")
        print(result.cri_result.cri)  # composite CRI score
    """

    def __init__(
        self,
        ground_truth: GroundTruth,
        judge: BinaryJudge,
        config: ScoringConfig | None = None,
    ) -> None:
        self.ground_truth = ground_truth
        self.judge = judge
        self.config = config if config is not None else ScoringConfig()
        self.dimension_registry: dict[str, MetricDimension] = _create_default_registry()
        self._validate_weights()

    def _validate_weights(self) -> None:
        """Ensure configured dimension weights sum to approximately 1.0."""
        total = sum(self.config.dimension_weights.values())
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Dimension weights must sum to 1.0, got {total:.4f}")

    async def run(
        self,
        adapter: MemoryAdapter,
        system_name: str,
    ) -> BenchmarkResult:
        """Execute the full benchmark evaluation pipeline.

        Runs each enabled dimension scorer against the adapter, computes
        the weighted composite CRI score, and packages everything into a
        :class:`~cri.models.BenchmarkResult`.

        Args:
            adapter: The memory system under evaluation.  It must have
                already ingested the benchmark conversation before this
                method is called.
            system_name: Human-readable name for the memory system being
                evaluated (e.g., ``"acme-memory-v2"``).

        Returns:
            A :class:`~cri.models.BenchmarkResult` containing the CRI
            evaluation result, a placeholder performance profile, and the
            full judge log.
        """
        run_id = str(uuid.uuid4())
        started_at = datetime.now(UTC).isoformat()

        # -- Score each enabled dimension ------------------------------------
        dimension_results: dict[str, DimensionResult] = {}

        for dim_name in self.config.enabled_dimensions:
            scorer = self.dimension_registry.get(dim_name)
            if scorer is None:
                logger.warning(
                    "Dimension %r is enabled but has no registered scorer — skipping.",
                    dim_name,
                )
                continue

            try:
                result = await scorer.score(adapter, self.ground_truth, self.judge)
                dimension_results[dim_name] = result
            except Exception as exc:
                logger.warning(
                    "Dimension %r scorer raised %s: %s — recording 0.0",
                    dim_name,
                    type(exc).__name__,
                    exc,
                )
                dimension_results[dim_name] = DimensionResult(
                    dimension_name=dim_name,
                    score=0.0,
                    passed_checks=0,
                    total_checks=0,
                    details=[{"error": str(exc)}],
                )

        # -- Compute composite CRI score ------------------------------------
        composite_cri = self._compute_composite(dimension_results)

        # -- Build CRIResult -------------------------------------------------
        _zero = DimensionResult(
            dimension_name="",
            score=0.0,
            passed_checks=0,
            total_checks=0,
        )
        cri_result = CRIResult(
            system_name=system_name,
            cri=round(composite_cri, 4),
            pas=round(dimension_results.get("PAS", _zero).score, 4),
            dbu=round(dimension_results.get("DBU", _zero).score, 4),
            tc=round(dimension_results.get("TC", _zero).score, 4),
            crq=round(dimension_results.get("CRQ", _zero).score, 4),
            qrp=round(dimension_results.get("QRP", _zero).score, 4),
            mei=round(dimension_results.get("MEI", _zero).score, 4),
            sfc=round(dimension_results.get("SFC", _zero).score, 4),
            lnc=round(dimension_results.get("LNC", _zero).score, 4),
            ars=round(dimension_results.get("ARS", _zero).score, 4),
            dimension_weights=dict(self.config.dimension_weights),
            details=dimension_results,
        )

        # -- Build placeholder PerformanceProfile ---------------------------
        performance_profile = PerformanceProfile(
            ingest_latency_ms=0.0,
            query_latency_avg_ms=0.0,
            query_latency_p95_ms=0.0,
            query_latency_p99_ms=0.0,
            total_facts_stored=0,
            memory_growth_curve=[],
            judge_api_calls=len(self.judge.get_log()),
            judge_total_cost_estimate=None,
        )

        # -- Collect judge log -----------------------------------------------
        judge_log = self.judge.get_log()

        completed_at = datetime.now(UTC).isoformat()

        return BenchmarkResult(
            run_id=run_id,
            adapter_name=system_name,
            dataset_id="",  # populated by caller if needed
            started_at=started_at,
            completed_at=completed_at,
            cri_result=cri_result,
            performance_profile=performance_profile,
            judge_log=judge_log,
        )

    def _compute_composite(
        self,
        dimension_results: dict[str, DimensionResult],
    ) -> float:
        """Compute the weighted composite CRI score.

        Uses the configured dimension weights.  If some dimensions are
        missing from the results, weights are re-normalized so the
        composite is still on a 0.0–1.0 scale.

        Args:
            dimension_results: Scored dimensions keyed by dimension name.

        Returns:
            The composite CRI score (0.0–1.0).
        """
        if not dimension_results:
            return 0.0

        # Collect active weights (only for dimensions that were scored)
        active_weights: dict[str, float] = {}
        for dim_name, _result in dimension_results.items():
            weight = self.config.dimension_weights.get(dim_name, 0.0)
            active_weights[dim_name] = weight

        total_weight = sum(active_weights.values())
        if total_weight == 0:
            return 0.0

        # Re-normalize weights
        normalized: dict[str, float] = {d: w / total_weight for d, w in active_weights.items()}

        composite = sum(normalized[d] * dimension_results[d].score for d in dimension_results if d in normalized)

        return composite
