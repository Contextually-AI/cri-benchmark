"""Dynamic Belief Updating (DBU) dimension scorer.

Measures how well the memory system updates its beliefs when new
information contradicts or supersedes previous knowledge.  All belief
change checks run **concurrently**.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from cri.models import (
    DimensionResult,
    Verdict,
)
from cri.scoring.dimensions.base import MetricDimension
from cri.scoring.rubrics import dbu_recency_check, dbu_staleness_check

if TYPE_CHECKING:
    from cri.adapter import MemoryAdapter
    from cri.judge import BinaryJudge
    from cri.models import GroundTruth

logger = logging.getLogger(__name__)


class DBUDimension(MetricDimension):
    """Dynamic Belief Updating dimension scorer (concurrent checks)."""

    name: str = "DBU"
    description: str = "Measures how well the system updates its beliefs when new information contradicts or supersedes previous knowledge."

    async def score(
        self,
        adapter: MemoryAdapter,
        ground_truth: GroundTruth,
        judge: BinaryJudge,
    ) -> DimensionResult:
        """Evaluate the memory system on Dynamic Belief Updating."""
        changes = ground_truth.changes

        if not changes:
            logger.info("DBU: no belief changes in ground truth — score defaults to 1.0")
            return DimensionResult(dimension_name=self.name, score=1.0, passed_checks=0, total_checks=0, details=[])

        async def _check_one(idx: int) -> dict[str, object]:
            change = changes[idx]
            stored_facts = adapter.retrieve(change.query_topic)
            fact_texts = [f.text for f in stored_facts]

            recency_check_id = f"dbu_recency_{idx}"
            staleness_check_id = f"dbu_staleness_{idx}"

            # Recency and staleness checks run concurrently.
            recency_result, staleness_result = await asyncio.gather(
                judge.judge_across_chunks(
                    recency_check_id,
                    fact_texts,
                    lambda chunk, _name=change.fact, _new=change.new_value: dbu_recency_check(  # type: ignore[misc]
                        fact_name=_name,
                        new_value=_new,
                        stored_facts=chunk,
                    ),
                ),
                judge.judge_across_chunks(
                    staleness_check_id,
                    fact_texts,
                    lambda chunk, _name=change.fact, _old=change.old_value: dbu_staleness_check(  # type: ignore[misc]
                        fact_name=_name,
                        old_value=_old,
                        stored_facts=chunk,
                    ),
                ),
            )

            passed = recency_result.verdict is Verdict.YES and staleness_result.verdict is Verdict.NO

            logger.debug(
                "DBU check %d (%s): recency=%s, staleness=%s → %s",
                idx,
                change.fact,
                recency_result.verdict.value,
                staleness_result.verdict.value,
                "PASS" if passed else "FAIL",
            )

            return {
                "belief_change": change.fact,
                "old_value": change.old_value,
                "new_value": change.new_value,
                "recency_verdict": recency_result.verdict.value,
                "staleness_verdict": staleness_result.verdict.value,
                "passed": passed,
                "recency_check_id": recency_check_id,
                "staleness_check_id": staleness_check_id,
            }

        # Run all belief-change checks concurrently.
        details_list = await asyncio.gather(*[_check_one(i) for i in range(len(changes))])

        passed_count = sum(1 for d in details_list if d["passed"])
        total_count = len(changes)
        dimension_score = passed_count / total_count

        logger.info("DBU: %d/%d belief changes passed (score=%.4f)", passed_count, total_count, dimension_score)

        return DimensionResult(
            dimension_name=self.name,
            score=dimension_score,
            passed_checks=passed_count,
            total_checks=total_count,
            details=list(details_list),
        )
