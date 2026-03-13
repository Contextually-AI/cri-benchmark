"""Conflict Resolution Quality (CRQ) dimension scorer.

Measures how well the memory system handles contradictory information —
whether it identifies conflicts, resolves them appropriately, and
communicates uncertainty when relevant.

This module provides two implementations:

- :class:`CRQDimension` — New ``MetricDimension`` subclass using binary
  verdict evaluation.  This is the **primary** implementation used by the
  current CRI pipeline.

Algorithm
---------
For each :class:`~cri.models.ConflictScenario` in the ground truth:

1. Query the adapter with the scenario's ``topic`` to retrieve stored facts.
2. Build an evaluation prompt using :func:`~cri.scoring.rubrics.crq_resolution_check`.
3. Submit the prompt to a :class:`~cri.judge.BinaryJudge` for a binary verdict.
4. A ``YES`` verdict means the conflict was resolved correctly.

The dimension score is ``passed_checks / total_checks``.  If there are no
conflict scenarios the score defaults to ``1.0`` (vacuously correct).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from cri.models import DimensionResult, Verdict
from cri.scoring.dimensions.base import MetricDimension
from cri.scoring.rubrics import crq_resolution_check

if TYPE_CHECKING:
    from cri.adapter import MemoryAdapter
    from cri.judge import BinaryJudge
    from cri.models import GroundTruth

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# New implementation — MetricDimension
# ---------------------------------------------------------------------------


class CRQDimension(MetricDimension):
    """Binary-verdict scorer for the Conflict Resolution Quality dimension.

    CRQ evaluates how well the memory system handles contradictory
    information:

    - Does the system detect contradictory information?
    - Does it resolve conflicts using appropriate strategies
      (recency, authority, explicit correction)?
    - Does the stored knowledge reflect the *correct* resolution?

    For each :class:`~cri.models.ConflictScenario`, the scorer queries the
    adapter, builds a rubric prompt via
    :func:`~cri.scoring.rubrics.crq_resolution_check`, and asks the
    :class:`~cri.judge.BinaryJudge` whether the stored facts reflect the
    expected resolution.
    """

    name: str = "CRQ"
    description: str = (
        "Measures how well the memory system handles contradictory "
        "information — whether it identifies conflicts, resolves them "
        "appropriately, and reflects the correct resolution."
    )

    async def score(
        self,
        adapter: MemoryAdapter,
        ground_truth: GroundTruth,
        judge: BinaryJudge,
    ) -> DimensionResult:
        """Evaluate conflict resolution quality.

        Args:
            adapter: The memory system under evaluation.
            ground_truth: Expected outcomes including conflict scenarios.
            judge: Binary verdict judge for semantic evaluation.

        Returns:
            A :class:`~cri.models.DimensionResult` with the CRQ score.
        """
        conflicts = ground_truth.conflicts

        # Vacuously correct when there are no conflict scenarios.
        if not conflicts:
            logger.info("CRQ: no conflict scenarios in ground truth; score=1.0")
            return DimensionResult(
                dimension_name=self.name,
                score=1.0,
                passed_checks=0,
                total_checks=0,
                details=[],
            )

        passed = 0
        total = len(conflicts)
        details: list[dict[str, object]] = []

        for scenario in conflicts:
            check_id = f"crq-{scenario.conflict_id}"

            # 1. Query the adapter for facts related to the conflict topic.
            stored_facts = adapter.query(scenario.topic)
            fact_texts = [f.text for f in stored_facts]

            # 2. Build the evaluation prompt.
            prompt = crq_resolution_check(
                topic=scenario.topic,
                correct_resolution=scenario.correct_resolution,
                stored_facts=fact_texts,
            )

            # 3. Judge the resolution.
            result = judge.judge(check_id=check_id, prompt=prompt)
            verdict_passed = result.verdict is Verdict.YES

            if verdict_passed:
                passed += 1

            # 4. Collect per-check detail.
            details.append(
                {
                    "check_id": check_id,
                    "conflict_id": scenario.conflict_id,
                    "topic": scenario.topic,
                    "resolution_type": scenario.resolution_type,
                    "correct_resolution": scenario.correct_resolution,
                    "stored_facts_count": len(fact_texts),
                    "verdict": result.verdict.value,
                    "passed": verdict_passed,
                    "unanimous": result.unanimous,
                }
            )

            logger.debug(
                "CRQ check %s: verdict=%s (passed=%s)",
                check_id,
                result.verdict.value,
                verdict_passed,
            )

        score = passed / total
        logger.info(
            "CRQ: %d/%d conflict resolutions correct (score=%.4f)",
            passed,
            total,
            score,
        )

        return DimensionResult(
            dimension_name=self.name,
            score=score,
            passed_checks=passed,
            total_checks=total,
            details=details,
        )
