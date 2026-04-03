"""Temporal Consistency (TC) dimension scorer.

Measures how well the memory system handles the temporal dimension
of knowledge.  All temporal fact checks run **concurrently**.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from cri.models import DimensionResult, GroundTruth, TemporalFact, Verdict
from cri.scoring.dimensions.base import MetricDimension
from cri.scoring.rubrics import tc_temporal_validity_check

if TYPE_CHECKING:
    from cri.adapter import MemoryAdapter
    from cri.judge import BinaryJudge

logger = logging.getLogger(__name__)


class TCDimension(MetricDimension):
    """Temporal Consistency dimension scorer (concurrent checks)."""

    name: str = "TC"
    description: str = "Measures how well the system handles temporal evolution of knowledge, distinguishing current from outdated information."

    async def score(
        self,
        adapter: MemoryAdapter,
        ground_truth: GroundTruth,
        judge: BinaryJudge,
    ) -> DimensionResult:
        """Evaluate temporal consistency of the memory system."""
        temporal_facts = ground_truth.temporal_facts

        if not temporal_facts:
            logger.info("TC: no temporal facts — returning 1.0 (vacuous)")
            return DimensionResult(
                dimension_name=self.name, score=1.0, passed_checks=0, total_checks=0, details=[]
            )

        async def _check_one(tf: TemporalFact) -> dict[str, object]:
            check_id = f"tc_{tf.fact_id}"
            stored_facts = adapter.retrieve(tf.query_topic)
            fact_texts = [f.text for f in stored_facts]

            result = await judge.judge_across_chunks(
                check_id,
                fact_texts,
                lambda chunk, _desc=tf.description, _cur=tf.should_be_current: tc_temporal_validity_check(  # type: ignore[misc]
                    fact_description=_desc, expected_current=_cur, stored_facts=chunk,
                ),
            )

            if not fact_texts:
                check_passed = False
            elif tf.should_be_current:
                check_passed = result.verdict is Verdict.YES
            else:
                check_passed = result.verdict is Verdict.NO

            logger.debug(
                "TC check %s: should_be_current=%s verdict=%s passed=%s",
                check_id, tf.should_be_current, result.verdict.value, check_passed,
            )

            return {
                "check_id": check_id,
                "fact_id": tf.fact_id,
                "description": tf.description,
                "should_be_current": tf.should_be_current,
                "verdict": result.verdict.value,
                "passed": check_passed,
                "num_stored_facts": len(fact_texts),
            }

        # Run all temporal fact checks concurrently.
        details_list = await asyncio.gather(*[_check_one(tf) for tf in temporal_facts])

        passed = sum(1 for d in details_list if d["passed"])
        total = len(temporal_facts)
        dimension_score = passed / total if total > 0 else 0.0

        logger.info("TC: %d/%d checks passed — score %.4f", passed, total, dimension_score)

        return DimensionResult(
            dimension_name=self.name,
            score=round(dimension_score, 4),
            passed_checks=passed,
            total_checks=total,
            details=list(details_list),
        )
