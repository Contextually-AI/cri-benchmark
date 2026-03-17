"""Temporal Consistency (TC) dimension scorer.

Measures how well the memory system handles the temporal dimension
of knowledge — understanding what is current versus outdated,
maintaining chronological ordering, and recognizing the evolution
of information over time.

Two scorer implementations are provided:

- :class:`TCDimension` — New binary-verdict scorer extending
  :class:`MetricDimension`.  Uses :func:`tc_temporal_validity_check`
  rubric prompts and :class:`BinaryJudge` majority-vote evaluation.

Algorithm (TCDimension)
~~~~~~~~~~~~~~~~~~~~~~~

For each :class:`~cri.models.TemporalFact` in
``ground_truth.temporal_facts``:

1. Query the adapter with ``temporal_fact.query_topic`` to obtain stored
   facts relevant to the temporal fact.
2. Build an evaluation prompt via :func:`tc_temporal_validity_check`.
3. Submit the prompt to the :class:`BinaryJudge`.
4. Determine pass/fail:

   - **should_be_current = True** — the fact is expected to be present
     and treated as current.  ``YES`` verdict → pass.
   - **should_be_current = False** — the fact has expired and should
     **not** be asserted as current.  ``NO`` verdict → pass (the system
     correctly does not assert it).

5. Score = ``passed_checks / total_checks`` (0.0 when no temporal facts).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from cri.models import DimensionResult, GroundTruth, Verdict
from cri.scoring.dimensions.base import MetricDimension
from cri.scoring.rubrics import tc_temporal_validity_check

if TYPE_CHECKING:
    from cri.adapter import MemoryAdapter
    from cri.judge import BinaryJudge

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# New binary-verdict scorer — TCDimension
# ---------------------------------------------------------------------------


class TCDimension(MetricDimension):
    """Temporal Consistency dimension scorer (binary-verdict pipeline).

    Evaluates whether the memory system correctly tracks the temporal
    validity of facts — knowing which facts are current and which have
    expired or been superseded.

    For each :class:`~cri.models.TemporalFact` in the ground truth the
    scorer:

    1. Queries the adapter for facts related to the temporal fact's topic.
    2. Uses the :func:`tc_temporal_validity_check` rubric to build a
       judge prompt.
    3. Collects a binary verdict from the :class:`BinaryJudge`.
    4. Maps the verdict to a pass/fail outcome based on whether the fact
       ``should_be_current``.

    The dimension score is the ratio of correctly-handled temporal facts
    to the total number of temporal facts.
    """

    name: str = "TC"
    description: str = "Measures how well the system handles temporal evolution of knowledge, distinguishing current from outdated information."

    async def score(
        self,
        adapter: MemoryAdapter,
        ground_truth: GroundTruth,
        judge: BinaryJudge,
    ) -> DimensionResult:
        """Evaluate temporal consistency of the memory system.

        Args:
            adapter: The memory system under evaluation.
            ground_truth: Expected outcomes including temporal facts.
            judge: Binary verdict judge for semantic evaluation.

        Returns:
            A :class:`~cri.models.DimensionResult` with the TC score,
            check counts, and per-check detail records.
        """
        temporal_facts = ground_truth.temporal_facts

        if not temporal_facts:
            logger.info("TC: no temporal facts — returning 1.0 (vacuous)")
            return DimensionResult(
                dimension_name=self.name,
                score=1.0,
                passed_checks=0,
                total_checks=0,
                details=[],
            )

        passed = 0
        details: list[dict[str, object]] = []

        for tf in temporal_facts:
            check_id = f"tc_{tf.fact_id}"

            # 1. Retrieve facts related to this topic
            stored_facts = adapter.retrieve(tf.query_topic)
            fact_texts = [f.text for f in stored_facts]

            # 2. Build the judge prompt
            prompt = tc_temporal_validity_check(
                fact_description=tf.description,
                expected_current=tf.should_be_current,
                stored_facts=fact_texts,
            )

            # 3. Get a binary verdict
            result = judge.judge(check_id=check_id, prompt=prompt)

            # 4. Determine pass/fail
            check_passed = result.verdict is Verdict.YES if tf.should_be_current else result.verdict is Verdict.NO

            if check_passed:
                passed += 1

            details.append(
                {
                    "check_id": check_id,
                    "fact_id": tf.fact_id,
                    "description": tf.description,
                    "should_be_current": tf.should_be_current,
                    "verdict": result.verdict.value,
                    "passed": check_passed,
                    "num_stored_facts": len(fact_texts),
                }
            )

            logger.debug(
                "TC check %s: should_be_current=%s verdict=%s passed=%s",
                check_id,
                tf.should_be_current,
                result.verdict.value,
                check_passed,
            )

        total = len(temporal_facts)
        dimension_score = passed / total if total > 0 else 0.0

        logger.info("TC: %d/%d checks passed — score %.4f", passed, total, dimension_score)

        return DimensionResult(
            dimension_name=self.name,
            score=round(dimension_score, 4),
            passed_checks=passed,
            total_checks=total,
            details=details,
        )
