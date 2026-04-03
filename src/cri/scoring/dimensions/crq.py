"""Conflict Resolution Quality (CRQ) dimension scorer.

Measures how well the memory system handles contradictory information.
All conflict scenario checks run **concurrently**.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from cri.models import DimensionResult, Verdict
from cri.scoring.dimensions.base import MetricDimension
from cri.scoring.rubrics import crq_resolution_check

if TYPE_CHECKING:
    from cri.adapter import MemoryAdapter
    from cri.judge import BinaryJudge
    from cri.models import ConflictScenario, GroundTruth

logger = logging.getLogger(__name__)


class CRQDimension(MetricDimension):
    """Binary-verdict scorer for Conflict Resolution Quality (concurrent)."""

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
        """Evaluate conflict resolution quality."""
        conflicts = ground_truth.conflicts

        if not conflicts:
            logger.info("CRQ: no conflict scenarios in ground truth; score=1.0")
            return DimensionResult(
                dimension_name=self.name, score=1.0, passed_checks=0, total_checks=0, details=[]
            )

        async def _check_one(scenario: ConflictScenario) -> dict[str, object]:
            check_id = f"crq-{scenario.conflict_id}"
            stored_facts = adapter.retrieve(scenario.topic)
            fact_texts = [f.text for f in stored_facts]

            result = await judge.judge_across_chunks(
                check_id,
                fact_texts,
                lambda chunk, _topic=scenario.topic, _res=scenario.correct_resolution: crq_resolution_check(  # type: ignore[misc]
                    topic=_topic, correct_resolution=_res, stored_facts=chunk,
                ),
            )
            verdict_passed = result.verdict is Verdict.YES

            logger.debug("CRQ check %s: verdict=%s (passed=%s)", check_id, result.verdict.value, verdict_passed)

            return {
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

        # Run all conflict checks concurrently.
        details_list = await asyncio.gather(*[_check_one(s) for s in conflicts])

        passed = sum(1 for d in details_list if d["passed"])
        total = len(conflicts)
        score_val = passed / total

        logger.info("CRQ: %d/%d conflict resolutions correct (score=%.4f)", passed, total, score_val)

        return DimensionResult(
            dimension_name=self.name,
            score=score_val,
            passed_checks=passed,
            total_checks=total,
            details=list(details_list),
        )
