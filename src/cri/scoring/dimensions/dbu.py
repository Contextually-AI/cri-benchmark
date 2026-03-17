"""Dynamic Belief Updating (DBU) dimension scorer.

Measures how well the memory system updates its beliefs when new
information contradicts or supersedes previous knowledge. A strong
memory system should seamlessly transition from old to new beliefs
without retaining outdated information as current truth.

This module provides two implementations:

- **DBUDimension** — The primary scorer, extending :class:`MetricDimension`.
  Uses the binary-verdict judge model and ``MemoryAdapter`` protocol.
  Each :class:`~cri.models.BeliefChange` in the ground truth is evaluated
  with a *recency check* (does the system reflect the new value?) and a
  *staleness check* (does the system still assert the old value as current?).
  A belief change passes only when recency = YES **and** staleness = NO.
"""

from __future__ import annotations

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


# ---------------------------------------------------------------------------
# New implementation — MetricDimension
# ---------------------------------------------------------------------------


class DBUDimension(MetricDimension):
    """Dynamic Belief Updating dimension scorer.

    Evaluates how well a memory system updates its beliefs when new
    information contradicts or supersedes previous knowledge.  For every
    :class:`~cri.models.BeliefChange` recorded in the ground truth the
    scorer performs two checks:

    1. **Recency check** — Does the system reflect the *new* value?
       Evaluated via :func:`~cri.scoring.rubrics.dbu_recency_check`.
       Expected verdict: **YES**.

    2. **Staleness check** — Does the system still assert the *old* value
       as the current truth?  Evaluated via
       :func:`~cri.scoring.rubrics.dbu_staleness_check`.
       Expected verdict: **NO** (i.e., the old value should *not* be
       asserted as current).

    A belief change **passes** only when ``recency == YES`` **and**
    ``staleness == NO``.

    The dimension score is the ratio of passed belief changes to the total
    number of belief changes (0.0–1.0).  If there are no belief changes in
    the ground truth, the score defaults to ``1.0`` (nothing to fail).
    """

    name: str = "DBU"
    description: str = "Measures how well the system updates its beliefs when new information contradicts or supersedes previous knowledge."

    async def score(
        self,
        adapter: MemoryAdapter,
        ground_truth: GroundTruth,
        judge: BinaryJudge,
    ) -> DimensionResult:
        """Evaluate the memory system on Dynamic Belief Updating.

        Args:
            adapter: The memory system under evaluation.
            ground_truth: Expected outcomes containing belief changes.
            judge: Binary verdict judge for semantic evaluation.

        Returns:
            A :class:`~cri.models.DimensionResult` with the DBU score,
            check counts, and per-belief-change detail records.
        """
        changes = ground_truth.changes

        if not changes:
            logger.info("DBU: no belief changes in ground truth — score defaults to 1.0")
            return DimensionResult(
                dimension_name=self.name,
                score=1.0,
                passed_checks=0,
                total_checks=0,
                details=[],
            )

        passed_count = 0
        details: list[dict[str, object]] = []

        for idx, change in enumerate(changes):
            # Retrieve facts related to this belief change
            stored_facts = adapter.retrieve(change.query_topic)
            fact_texts = [f.text for f in stored_facts]

            # --- Recency check ---
            recency_check_id = f"dbu_recency_{idx}"
            recency_prompt = dbu_recency_check(
                fact_name=change.fact,
                new_value=change.new_value,
                stored_facts=fact_texts,
            )
            recency_result = judge.judge(recency_check_id, recency_prompt)

            # --- Staleness check ---
            staleness_check_id = f"dbu_staleness_{idx}"
            staleness_prompt = dbu_staleness_check(
                fact_name=change.fact,
                old_value=change.old_value,
                stored_facts=fact_texts,
            )
            staleness_result = judge.judge(staleness_check_id, staleness_prompt)

            # --- Verdict ---
            # Pass requires: recency = YES (new value present)
            #            AND staleness = NO  (old value NOT asserted as current)
            passed = recency_result.verdict is Verdict.YES and staleness_result.verdict is Verdict.NO

            if passed:
                passed_count += 1

            details.append(
                {
                    "belief_change": change.fact,
                    "old_value": change.old_value,
                    "new_value": change.new_value,
                    "recency_verdict": recency_result.verdict.value,
                    "staleness_verdict": staleness_result.verdict.value,
                    "passed": passed,
                    "recency_check_id": recency_check_id,
                    "staleness_check_id": staleness_check_id,
                }
            )

            logger.debug(
                "DBU check %d (%s): recency=%s, staleness=%s → %s",
                idx,
                change.fact,
                recency_result.verdict.value,
                staleness_result.verdict.value,
                "PASS" if passed else "FAIL",
            )

        total_count = len(changes)
        dimension_score = passed_count / total_count

        logger.info(
            "DBU: %d/%d belief changes passed (score=%.4f)",
            passed_count,
            total_count,
            dimension_score,
        )

        return DimensionResult(
            dimension_name=self.name,
            score=dimension_score,
            passed_checks=passed_count,
            total_checks=total_count,
            details=details,
        )
