"""SSI — Scale Sensitivity Index.

A **meta-metric** that measures how composite CRI degrades (or improves)
as the volume of ingested messages increases.  Unlike the six core
dimensions, SSI is *not* a :class:`~cri.scoring.dimensions.base.MetricDimension`
because it needs to control ingestion and run the scoring engine at
multiple scale points.

Usage::

    from cri.scoring.ssi import compute_ssi

    ssi_result = await compute_ssi(
        adapter_factory=lambda: MyAdapter(),
        messages=dataset.messages,
        ground_truth=dataset.ground_truth,
        judge_factory=lambda: BinaryJudge(),
    )
    print(ssi_result.score)  # 0.0–1.0

Formula::

    scales = [0.25, 0.50, 0.75, 1.00]
    CRI_s = composite CRI using the first s × len(messages) messages

    degradation_rate = (CRI_25% − CRI_100%) / CRI_25%
    SSI = 1 − max(0, degradation_rate)

    SSI = 1.0 → no degradation (performance stable or improving)
    SSI = 0.0 → total degradation
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

from cri.models import DimensionResult

if TYPE_CHECKING:
    from cri.adapter import MemoryAdapter
    from cri.judge import BinaryJudge
    from cri.models import GroundTruth, Message, ScoringConfig

logger = logging.getLogger(__name__)

# Default scale fractions at which the benchmark is evaluated.
DEFAULT_SCALES: list[float] = [0.25, 0.50, 0.75, 1.00]


async def compute_ssi(
    adapter_factory: Callable[[], MemoryAdapter],
    messages: list[Message],
    ground_truth: GroundTruth,
    judge_factory: Callable[[], BinaryJudge],
    config: ScoringConfig | None = None,
    scales: list[float] | None = None,
) -> DimensionResult:
    """Run the CRI benchmark at multiple scale points and compute SSI.

    Args:
        adapter_factory: Zero-argument callable that returns a **fresh**
            adapter instance.  A new adapter is created for each scale
            point so that earlier ingestions do not leak into later runs.
        messages: The full ordered list of conversation messages.
        ground_truth: Ground truth for the dataset.
        judge_factory: Zero-argument callable that returns a **fresh**
            :class:`~cri.judge.BinaryJudge`.  A new judge is created per
            scale point to keep logs separate.
        config: Optional :class:`~cri.models.ScoringConfig`.  If ``None``,
            the default config (6 core dimensions) is used.
        scales: Fractions of the dataset at which to evaluate.  Defaults
            to ``[0.25, 0.50, 0.75, 1.00]``.

    Returns:
        A :class:`~cri.models.DimensionResult` with ``dimension_name``
        set to ``"SSI"`` and a score in [0.0, 1.0].
    """
    # Lazy import to avoid circular dependency (engine imports dimensions).
    from cri.scoring.engine import ScoringEngine

    if scales is None:
        scales = list(DEFAULT_SCALES)

    total_messages = len(messages)
    if total_messages == 0:
        logger.info("SSI: no messages — returning 1.0 (vacuous).")
        return DimensionResult(
            dimension_name="SSI",
            score=1.0,
            passed_checks=0,
            total_checks=0,
            details=[],
        )

    # -- Evaluate CRI at each scale point -----------------------------------
    scale_results: list[dict[str, object]] = []
    cri_values: dict[float, float] = {}

    for scale in sorted(scales):
        cutoff = max(1, int(total_messages * scale))
        subset = messages[:cutoff]

        # Fresh adapter and judge for this scale point.
        adapter = adapter_factory()
        judge = judge_factory()

        # Ingest subset.
        adapter.ingest(subset)

        # Run scoring engine (uses only core dimensions, not SSI itself).
        engine = ScoringEngine(
            ground_truth=ground_truth,
            judge=judge,
            config=config,
        )
        result = await engine.run(adapter, system_name=f"ssi-{scale:.0%}")
        cri_score = result.cri_result.cri

        cri_values[scale] = cri_score
        scale_results.append(
            {
                "scale": scale,
                "message_count": cutoff,
                "cri": round(cri_score, 4),
            }
        )

        logger.info(
            "SSI scale %.0f%%: %d messages → CRI=%.4f",
            scale * 100,
            cutoff,
            cri_score,
        )

    # -- Compute SSI --------------------------------------------------------
    smallest_scale = min(scales)
    largest_scale = max(scales)
    cri_small = cri_values[smallest_scale]
    cri_large = cri_values[largest_scale]

    degradation_rate = 0.0 if cri_small == 0 else (cri_small - cri_large) / cri_small

    ssi = 1.0 - max(0.0, degradation_rate)

    details: list[dict[str, object]] = list(scale_results)
    details.append(
        {
            "summary": True,
            "cri_at_smallest_scale": round(cri_small, 4),
            "cri_at_largest_scale": round(cri_large, 4),
            "degradation_rate": round(degradation_rate, 4),
            "ssi": round(ssi, 4),
        }
    )

    logger.info(
        "SSI: CRI@%.0f%%=%.4f CRI@%.0f%%=%.4f degradation=%.4f SSI=%.4f",
        smallest_scale * 100,
        cri_small,
        largest_scale * 100,
        cri_large,
        degradation_rate,
        ssi,
    )

    return DimensionResult(
        dimension_name="SSI",
        score=round(ssi, 4),
        passed_checks=0,
        total_checks=len(scales),
        details=details,
    )
