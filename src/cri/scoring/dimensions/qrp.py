"""Query Relevance Precision (QRP) dimension scorer.

Measures the precision, relevance, and completeness of the memory
system's responses to queries. Evaluates how directly and accurately
the system answers what was asked.

Two scorer implementations are provided:

- :class:`QRPDimension` — New binary-verdict scorer extending
  :class:`MetricDimension`.  Uses :func:`qrp_relevance_check` and
  :func:`qrp_irrelevance_check` rubric prompts with
  :class:`BinaryJudge` majority-vote evaluation.

Algorithm (QRPDimension)
~~~~~~~~~~~~~~~~~~~~~~~~

For each :class:`~cri.models.QueryRelevancePair` in
``ground_truth.query_relevance_pairs``:

1. Query the adapter with ``pair.query`` to obtain the returned facts.
2. **Relevance checks (recall)**: For each expected relevant fact, build
   a prompt via :func:`qrp_relevance_check` and submit it to the judge.
   ``YES`` verdict → the fact was found (pass).
3. **Irrelevance checks (precision)**: For each expected irrelevant fact,
   build a prompt via :func:`qrp_irrelevance_check` and submit it to
   the judge.  ``YES`` verdict means the irrelevant fact was incorrectly
   included (fail); ``NO`` verdict means it was correctly excluded (pass).
4. Compute per-pair metrics:

   - ``recall = relevant_found / total_relevant`` (1.0 if no relevant facts)
   - ``precision = irrelevant_excluded / total_irrelevant`` (1.0 if no
     irrelevant facts)

5. ``pair_score = 0.5 * recall + 0.5 * precision``

Final score = arithmetic mean of all pair scores (0.0 when there are no
query-relevance pairs).
"""

from __future__ import annotations

import logging
import statistics
from typing import TYPE_CHECKING

from cri.models import DimensionResult, GroundTruth, Verdict
from cri.scoring.dimensions.base import MetricDimension
from cri.scoring.rubrics import qrp_irrelevance_check, qrp_relevance_check

if TYPE_CHECKING:
    from cri.adapter import MemoryAdapter
    from cri.judge import BinaryJudge

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# New binary-verdict scorer — QRPDimension
# ---------------------------------------------------------------------------


class QRPDimension(MetricDimension):
    """Query Relevance Precision dimension scorer (binary-verdict pipeline).

    Evaluates how precisely and completely the memory system retrieves
    relevant facts in response to queries, while filtering out irrelevant
    information.

    For each :class:`~cri.models.QueryRelevancePair` in the ground truth
    the scorer:

    1. Queries the adapter to obtain returned facts.
    2. Uses :func:`qrp_relevance_check` to verify each expected relevant
       fact is present (recall).
    3. Uses :func:`qrp_irrelevance_check` to verify each expected
       irrelevant fact is absent (precision).
    4. Computes per-pair score as ``0.5 * recall + 0.5 * precision``.
    5. Averages across all pairs for the final dimension score.
    """

    name: str = "QRP"
    description: str = (
        "Measures the precision, relevance, and completeness of "
        "the system's responses to queries about the user."
    )

    async def score(
        self,
        adapter: MemoryAdapter,
        ground_truth: GroundTruth,
        judge: BinaryJudge,
    ) -> DimensionResult:
        """Evaluate query relevance precision of the memory system.

        Args:
            adapter: The memory system under evaluation.
            ground_truth: Expected outcomes including query-relevance pairs.
            judge: Binary verdict judge for semantic evaluation.

        Returns:
            A :class:`~cri.models.DimensionResult` with the QRP score,
            check counts, and per-check detail records.
        """
        pairs = ground_truth.query_relevance_pairs

        if not pairs:
            logger.info("QRP: no query-relevance pairs — returning 1.0 (vacuous)")
            return DimensionResult(
                dimension_name=self.name,
                score=1.0,
                passed_checks=0,
                total_checks=0,
                details=[],
            )

        total_passed = 0
        total_checks = 0
        pair_scores: list[float] = []
        details: list[dict[str, object]] = []

        for pair in pairs:
            # 1. Query the adapter for facts related to this query
            stored_facts = adapter.query(pair.query)
            fact_texts = [f.text for f in stored_facts]

            # 2. Relevance checks (recall)
            relevance_passed = 0
            relevance_total = len(pair.expected_relevant_facts)

            for idx, expected_fact in enumerate(pair.expected_relevant_facts):
                check_id = f"qrp_rel_{pair.query_id}_{idx}"
                total_checks += 1

                prompt = qrp_relevance_check(
                    query=pair.query,
                    expected_fact=expected_fact,
                    stored_facts=fact_texts,
                )

                result = judge.judge(check_id=check_id, prompt=prompt)

                # YES = fact was found (pass)
                check_passed = result.verdict is Verdict.YES
                if check_passed:
                    relevance_passed += 1
                    total_passed += 1

                details.append(
                    {
                        "check_id": check_id,
                        "query_id": pair.query_id,
                        "query": pair.query,
                        "check_type": "relevance",
                        "expected_fact": expected_fact,
                        "verdict": result.verdict.value,
                        "passed": check_passed,
                        "num_returned_facts": len(fact_texts),
                    }
                )

                logger.debug(
                    "QRP relevance check %s: verdict=%s passed=%s",
                    check_id,
                    result.verdict.value,
                    check_passed,
                )

            # 3. Irrelevance checks (precision)
            irrelevance_passed = 0
            irrelevance_total = len(pair.expected_irrelevant_facts)

            for idx, irrelevant_fact in enumerate(pair.expected_irrelevant_facts):
                check_id = f"qrp_irr_{pair.query_id}_{idx}"
                total_checks += 1

                prompt = qrp_irrelevance_check(
                    query=pair.query,
                    irrelevant_fact=irrelevant_fact,
                    stored_facts=fact_texts,
                )

                result = judge.judge(check_id=check_id, prompt=prompt)

                # YES = irrelevant fact was included (FAIL)
                # NO = irrelevant fact was correctly excluded (PASS)
                check_passed = result.verdict is Verdict.NO
                if check_passed:
                    irrelevance_passed += 1
                    total_passed += 1

                details.append(
                    {
                        "check_id": check_id,
                        "query_id": pair.query_id,
                        "query": pair.query,
                        "check_type": "irrelevance",
                        "expected_fact": irrelevant_fact,
                        "verdict": result.verdict.value,
                        "passed": check_passed,
                        "num_returned_facts": len(fact_texts),
                    }
                )

                logger.debug(
                    "QRP irrelevance check %s: verdict=%s passed=%s",
                    check_id,
                    result.verdict.value,
                    check_passed,
                )

            # 4. Compute per-pair recall and precision
            recall = relevance_passed / relevance_total if relevance_total > 0 else 1.0
            precision = irrelevance_passed / irrelevance_total if irrelevance_total > 0 else 1.0

            # 5. pair_score = 0.5 * recall + 0.5 * precision
            pair_score = 0.5 * recall + 0.5 * precision
            pair_scores.append(pair_score)

            logger.debug(
                "QRP pair %s: recall=%.4f (%d/%d) precision=%.4f (%d/%d) pair_score=%.4f",
                pair.query_id,
                recall,
                relevance_passed,
                relevance_total,
                precision,
                irrelevance_passed,
                irrelevance_total,
                pair_score,
            )

        # 6. Final score = mean of pair_scores
        dimension_score = statistics.mean(pair_scores) if pair_scores else 0.0

        logger.info(
            "QRP: %d/%d checks passed across %d pairs — score %.4f",
            total_passed,
            total_checks,
            len(pairs),
            dimension_score,
        )

        return DimensionResult(
            dimension_name=self.name,
            score=round(dimension_score, 4),
            passed_checks=total_passed,
            total_checks=total_checks,
            details=details,
        )
