"""Long-Horizon Narrative Coherence (LNC) dimension scorer.

Measures whether the memory system maintains a coherent story of the user
across causally connected events — not just isolated facts.

Algorithm
---------
For each :class:`~cri.models.NarrativeArc` in the ground truth:

1. Query the adapter with the arc's ``query_topic`` to retrieve stored facts.
2. Run three binary checks per arc:
   - **Sequence**: Do the stored facts reflect the correct chronological
     order of events?  (YES = pass)
   - **Causality**: Are causal relationships between events preserved?
     (YES = pass)
   - **Contradiction**: Are there internal contradictions in the narrative?
     (NO = pass — no contradictions is good)
3. ``arc_score = (sequence_pass + causality_pass + contradiction_pass) / 3``

The dimension score is ``mean(arc_scores)``.  If there are no narrative arcs
the score defaults to ``1.0`` (vacuously correct).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from cri.models import DimensionResult, Verdict
from cri.scoring.dimensions.base import MetricDimension
from cri.scoring.rubrics import (
    lnc_causality_check,
    lnc_contradiction_check,
    lnc_sequence_check,
)

if TYPE_CHECKING:
    from cri.adapter import MemoryAdapter
    from cri.judge import BinaryJudge
    from cri.models import GroundTruth

logger = logging.getLogger(__name__)


class LNCDimension(MetricDimension):
    """Binary-verdict scorer for the Long-Horizon Narrative Coherence dimension.

    LNC evaluates whether the memory system maintains a coherent narrative
    of the user's life across causally connected events:

    - Does the system preserve the chronological sequence of events?
    - Are causal relationships between events captured?
    - Is the narrative free of internal contradictions?

    For each :class:`~cri.models.NarrativeArc`, the scorer queries the
    adapter and runs three binary checks via
    :func:`~cri.scoring.rubrics.lnc_sequence_check`,
    :func:`~cri.scoring.rubrics.lnc_causality_check`, and
    :func:`~cri.scoring.rubrics.lnc_contradiction_check`.
    """

    name: str = "LNC"
    description: str = (
        "Measures whether the memory system maintains a coherent narrative "
        "across causally connected events — preserving chronological order, "
        "causal relationships, and internal consistency."
    )

    async def score(
        self,
        adapter: MemoryAdapter,
        ground_truth: GroundTruth,
        judge: BinaryJudge,
    ) -> DimensionResult:
        """Evaluate long-horizon narrative coherence.

        Args:
            adapter: The memory system under evaluation.
            ground_truth: Expected outcomes including narrative arcs.
            judge: Binary verdict judge for semantic evaluation.

        Returns:
            A :class:`~cri.models.DimensionResult` with the LNC score.
        """
        arcs = ground_truth.narrative_arcs

        # Vacuously correct when there are no narrative arcs.
        if not arcs:
            logger.info("LNC: no narrative arcs in ground truth; score=1.0")
            return DimensionResult(
                dimension_name=self.name,
                score=1.0,
                passed_checks=0,
                total_checks=0,
                details=[],
            )

        total_checks = len(arcs) * 3  # 3 checks per arc
        total_passed = 0
        arc_scores: list[float] = []
        details: list[dict[str, object]] = []

        for arc in arcs:
            # 1. Retrieve facts related to the narrative arc.
            stored_facts = adapter.retrieve(arc.query_topic)
            fact_texts = [f.text for f in stored_facts]

            # 2a. Sequence check — YES = events in correct order (pass).
            seq_id = f"lnc-seq-{arc.arc_id}"
            seq_prompt = lnc_sequence_check(
                events_in_order=arc.events_in_order,
                topic=arc.topic,
                stored_facts=fact_texts,
            )
            seq_result = judge.judge(check_id=seq_id, prompt=seq_prompt)
            seq_pass = seq_result.verdict is Verdict.YES

            # 2b. Causality check — YES = causal links preserved (pass).
            caus_id = f"lnc-caus-{arc.arc_id}"
            caus_prompt = lnc_causality_check(
                causal_links=arc.causal_links,
                topic=arc.topic,
                stored_facts=fact_texts,
            )
            caus_result = judge.judge(check_id=caus_id, prompt=caus_prompt)
            caus_pass = caus_result.verdict is Verdict.YES

            # 2c. Contradiction check — NO = no contradictions (pass).
            contra_id = f"lnc-contra-{arc.arc_id}"
            contra_prompt = lnc_contradiction_check(
                topic=arc.topic,
                stored_facts=fact_texts,
            )
            contra_result = judge.judge(check_id=contra_id, prompt=contra_prompt)
            contra_pass = contra_result.verdict is Verdict.NO

            # 3. Compute arc score.
            checks_passed = sum([seq_pass, caus_pass, contra_pass])
            total_passed += checks_passed
            arc_score = checks_passed / 3.0
            arc_scores.append(arc_score)

            # 4. Collect per-arc detail.
            details.append(
                {
                    "arc_id": arc.arc_id,
                    "topic": arc.topic,
                    "stored_facts_count": len(fact_texts),
                    "sequence_verdict": seq_result.verdict.value,
                    "sequence_passed": seq_pass,
                    "causality_verdict": caus_result.verdict.value,
                    "causality_passed": caus_pass,
                    "contradiction_verdict": contra_result.verdict.value,
                    "contradiction_passed": contra_pass,
                    "arc_score": round(arc_score, 4),
                }
            )

            logger.debug(
                "LNC arc %s: seq=%s caus=%s contra=%s arc_score=%.4f",
                arc.arc_id,
                seq_result.verdict.value,
                caus_result.verdict.value,
                contra_result.verdict.value,
                arc_score,
            )

        score = sum(arc_scores) / len(arc_scores)
        logger.info(
            "LNC: %d/%d checks passed across %d arcs (score=%.4f)",
            total_passed,
            total_checks,
            len(arcs),
            score,
        )

        return DimensionResult(
            dimension_name=self.name,
            score=round(score, 4),
            passed_checks=total_passed,
            total_checks=total_checks,
            details=details,
        )
