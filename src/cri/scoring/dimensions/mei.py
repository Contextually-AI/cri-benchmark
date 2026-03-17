"""MEI — Memory Efficiency Index.

Measures global storage efficiency by comparing what the system stored
against what it *should* have stored (ground truth).  A perfect system
stores exactly the N ground-truth facts and nothing more (MEI = 1.0).

Formula::

    efficiency_ratio = covered_gt_facts / total_facts_stored
    coverage_factor  = covered_gt_facts / total_gt_facts
    MEI = harmonic_mean(efficiency_ratio, coverage_factor)

The dimension uses :func:`~cri.adapter.MemoryAdapter.get_events` and
evaluates each ground-truth fact for coverage using the
:class:`~cri.judge.BinaryJudge`.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from cri.models import DimensionResult, Verdict
from cri.scoring.dimensions.base import MetricDimension
from cri.scoring.rubrics import mei_coverage_check

if TYPE_CHECKING:
    from cri.adapter import MemoryAdapter
    from cri.judge import BinaryJudge
    from cri.models import GroundTruth

logger = logging.getLogger(__name__)


class MEIDimension(MetricDimension):
    """Memory Efficiency Index scorer.

    Evaluates how efficiently the memory system stores information by
    measuring the balance between *coverage* (how many ground-truth facts
    are represented) and *efficiency* (how lean the storage is relative
    to the useful facts it covers).
    """

    name: str = "MEI"
    description: str = "Memory Efficiency Index — measures the balance between storage coverage and storage efficiency."

    async def score(
        self,
        adapter: MemoryAdapter,
        ground_truth: GroundTruth,
        judge: BinaryJudge,
    ) -> DimensionResult:
        """Score the adapter's storage efficiency against ground truth.

        Returns a :class:`~cri.models.DimensionResult` with a score in
        [0.0, 1.0] representing the harmonic mean of coverage and
        efficiency.
        """
        # -- Build the list of expected ground-truth facts -------------------
        gt_facts = _build_gt_facts(ground_truth)

        if not gt_facts:
            logger.info("MEI: no ground-truth facts — returning 1.0 (vacuous).")
            return DimensionResult(
                dimension_name=self.name,
                score=1.0,
                passed_checks=0,
                total_checks=0,
                details=[],
            )

        # -- Retrieve all stored events once ---------------------------------
        all_stored = adapter.get_events()
        stored_texts = [sf.text for sf in all_stored]
        total_stored = len(all_stored)

        if total_stored == 0:
            logger.info("MEI: adapter stored 0 facts — returning 0.0.")
            return DimensionResult(
                dimension_name=self.name,
                score=0.0,
                passed_checks=0,
                total_checks=len(gt_facts),
                details=[],
            )

        # -- Coverage checks: is each GT fact represented? -------------------
        covered = 0
        total_gt = len(gt_facts)
        details: list[dict[str, object]] = []

        for idx, (gt_key, gt_value) in enumerate(gt_facts):
            check_id = f"mei-coverage-{idx}"
            prompt = mei_coverage_check(gt_key, gt_value, stored_texts)
            result = judge.judge(check_id, prompt)
            passed = result.verdict is Verdict.YES

            if passed:
                covered += 1

            details.append(
                {
                    "check_id": check_id,
                    "gt_key": gt_key,
                    "gt_value": gt_value,
                    "verdict": result.verdict.value,
                    "passed": passed,
                }
            )

            logger.debug(
                "MEI coverage check %s: key=%r verdict=%s",
                check_id,
                gt_key,
                result.verdict.value,
            )

        # -- Compute MEI -----------------------------------------------------
        coverage = covered / total_gt
        efficiency = covered / total_stored
        mei = _harmonic_mean(efficiency, coverage)

        details.append(
            {
                "summary": True,
                "total_stored_facts": total_stored,
                "total_gt_facts": total_gt,
                "covered_gt_facts": covered,
                "coverage": round(coverage, 4),
                "efficiency": round(efficiency, 4),
                "mei": round(mei, 4),
            }
        )

        logger.info(
            "MEI: covered=%d/%d gt facts, stored=%d total, coverage=%.4f efficiency=%.4f MEI=%.4f",
            covered,
            total_gt,
            total_stored,
            coverage,
            efficiency,
            mei,
        )

        return DimensionResult(
            dimension_name=self.name,
            score=round(mei, 4),
            passed_checks=covered,
            total_checks=total_gt,
            details=details,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_gt_facts(ground_truth: GroundTruth) -> list[tuple[str, str]]:
    """Extract (key, value) pairs from the ground truth's final profile.

    Each profile dimension contributes one or more ground-truth facts.
    Multi-value dimensions (lists) are flattened so each value is a
    separate fact.

    Returns:
        A list of ``(dimension_name, expected_value)`` tuples.
    """
    facts: list[tuple[str, str]] = []
    for dim_name, profile_dim in ground_truth.final_profile.items():
        if isinstance(profile_dim.value, list):
            for v in profile_dim.value:
                facts.append((dim_name, v))
        else:
            facts.append((dim_name, profile_dim.value))
    return facts


def _harmonic_mean(a: float, b: float) -> float:
    """Compute the harmonic mean of two values.

    Returns 0.0 if either value is zero (avoids division by zero).
    """
    if a + b == 0:
        return 0.0
    return 2.0 * a * b / (a + b)
