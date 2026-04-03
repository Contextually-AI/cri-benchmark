"""Query Relevance Precision (QRP) dimension scorer.

Measures the precision, relevance, and completeness of the memory
system's responses to queries.  All query-relevance pairs and their
inner checks run **concurrently**.
"""

from __future__ import annotations

import asyncio
import logging
import statistics
from typing import TYPE_CHECKING

from cri.models import DimensionResult, GroundTruth, Verdict
from cri.scoring.dimensions.base import MetricDimension
from cri.scoring.rubrics import qrp_irrelevance_check, qrp_relevance_check

if TYPE_CHECKING:
    from cri.adapter import MemoryAdapter
    from cri.judge import BinaryJudge
    from cri.models import QueryRelevancePair

logger = logging.getLogger(__name__)


class QRPDimension(MetricDimension):
    """Query Relevance Precision dimension scorer (concurrent checks)."""

    name: str = "QRP"
    description: str = "Measures the precision, relevance, and completeness of the system's responses to queries about the user."

    async def score(
        self,
        adapter: MemoryAdapter,
        ground_truth: GroundTruth,
        judge: BinaryJudge,
    ) -> DimensionResult:
        """Evaluate query relevance precision of the memory system."""
        pairs = ground_truth.query_relevance_pairs

        if not pairs:
            logger.info("QRP: no query-relevance pairs — returning 1.0 (vacuous)")
            return DimensionResult(
                dimension_name=self.name, score=1.0, passed_checks=0, total_checks=0, details=[]
            )

        async def _score_pair(pair: QueryRelevancePair) -> tuple[float, int, int, list[dict[str, object]]]:
            """Score a single query-relevance pair. Returns (pair_score, passed, total, details)."""
            stored_facts = adapter.retrieve(pair.query)
            fact_texts = [f.text for f in stored_facts]

            if not fact_texts:
                # No facts returned — both recall and precision are 0.
                pair_details: list[dict[str, object]] = []
                for idx, expected_fact in enumerate(pair.expected_relevant_facts):
                    pair_details.append({
                        "check_id": f"qrp_rel_{pair.query_id}_{idx}",
                        "query_id": pair.query_id, "query": pair.query,
                        "check_type": "relevance", "expected_fact": expected_fact,
                        "verdict": "NO", "passed": False, "num_returned_facts": 0,
                    })
                for idx, irrelevant_fact in enumerate(pair.expected_irrelevant_facts):
                    pair_details.append({
                        "check_id": f"qrp_irr_{pair.query_id}_{idx}",
                        "query_id": pair.query_id, "query": pair.query,
                        "check_type": "irrelevance", "expected_fact": irrelevant_fact,
                        "verdict": "N/A", "passed": False, "num_returned_facts": 0,
                    })
                total = len(pair.expected_relevant_facts) + len(pair.expected_irrelevant_facts)
                return 0.0, 0, total, pair_details

            # --- Fire all relevance + irrelevance checks concurrently ---

            async def _rel_check(idx: int, expected_fact: str) -> dict[str, object]:
                check_id = f"qrp_rel_{pair.query_id}_{idx}"
                result = await judge.judge_across_chunks(
                    check_id, fact_texts,
                    lambda chunk, _q=pair.query, _f=expected_fact: qrp_relevance_check(  # type: ignore[misc]
                        query=_q, expected_fact=_f, stored_facts=chunk,
                    ),
                )
                check_passed = result.verdict is Verdict.YES
                logger.debug("QRP relevance check %s: verdict=%s passed=%s", check_id, result.verdict.value, check_passed)
                return {
                    "check_id": check_id, "query_id": pair.query_id, "query": pair.query,
                    "check_type": "relevance", "expected_fact": expected_fact,
                    "verdict": result.verdict.value, "passed": check_passed,
                    "num_returned_facts": len(fact_texts),
                }

            async def _irr_check(idx: int, irrelevant_fact: str) -> dict[str, object]:
                check_id = f"qrp_irr_{pair.query_id}_{idx}"
                result = await judge.judge_across_chunks(
                    check_id, fact_texts,
                    lambda chunk, _q=pair.query, _f=irrelevant_fact: qrp_irrelevance_check(  # type: ignore[misc]
                        query=_q, irrelevant_fact=_f, stored_facts=chunk,
                    ),
                )
                # YES = irrelevant fact included (FAIL); NO = correctly excluded (PASS)
                check_passed = result.verdict is Verdict.NO
                logger.debug("QRP irrelevance check %s: verdict=%s passed=%s", check_id, result.verdict.value, check_passed)
                return {
                    "check_id": check_id, "query_id": pair.query_id, "query": pair.query,
                    "check_type": "irrelevance", "expected_fact": irrelevant_fact,
                    "verdict": result.verdict.value, "passed": check_passed,
                    "num_returned_facts": len(fact_texts),
                }

            all_tasks: list[asyncio.Task[dict[str, object]]] = []
            for idx, ef in enumerate(pair.expected_relevant_facts):
                all_tasks.append(asyncio.create_task(_rel_check(idx, ef)))
            for idx, irf in enumerate(pair.expected_irrelevant_facts):
                all_tasks.append(asyncio.create_task(_irr_check(idx, irf)))

            all_details = await asyncio.gather(*all_tasks)

            # Compute per-pair recall and precision.
            rel_details = [d for d in all_details if d["check_type"] == "relevance"]
            irr_details = [d for d in all_details if d["check_type"] == "irrelevance"]

            rel_passed = sum(1 for d in rel_details if d["passed"])
            rel_total = len(rel_details)
            irr_passed = sum(1 for d in irr_details if d["passed"])
            irr_total = len(irr_details)

            recall = rel_passed / rel_total if rel_total > 0 else 1.0
            precision = irr_passed / irr_total if irr_total > 0 else 1.0
            pair_score = 0.5 * recall + 0.5 * precision

            total_passed = rel_passed + irr_passed
            total_checks = rel_total + irr_total

            logger.debug(
                "QRP pair %s: recall=%.4f (%d/%d) precision=%.4f (%d/%d) pair_score=%.4f",
                pair.query_id, recall, rel_passed, rel_total, precision, irr_passed, irr_total, pair_score,
            )

            return pair_score, total_passed, total_checks, list(all_details)

        # Run all pairs concurrently.
        pair_results = await asyncio.gather(*[_score_pair(p) for p in pairs])

        total_passed = 0
        total_checks = 0
        pair_scores: list[float] = []
        all_details: list[dict[str, object]] = []

        for pair_score, pp, tc, dets in pair_results:
            pair_scores.append(pair_score)
            total_passed += pp
            total_checks += tc
            all_details.extend(dets)

        dimension_score = statistics.mean(pair_scores) if pair_scores else 0.0

        logger.info(
            "QRP: %d/%d checks passed across %d pairs — score %.4f",
            total_passed, total_checks, len(pairs), dimension_score,
        )

        return DimensionResult(
            dimension_name=self.name,
            score=round(dimension_score, 4),
            passed_checks=total_passed,
            total_checks=total_checks,
            details=all_details,
        )
