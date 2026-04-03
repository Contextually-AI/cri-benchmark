"""Persona Accuracy Score (PAS) dimension scorer.

Measures how accurately the memory system recalls specific persona
details after ingesting events about a user. This is the most
fundamental dimension -- a memory system must at minimum be able
to accurately recall what it has been told.

:class:`ProfileAccuracyScore` uses :class:`~cri.judge.BinaryJudge` with
the :func:`~cri.scoring.rubrics.pas_check` rubric to evaluate each
profile dimension independently.  All checks run **concurrently**.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from cri.models import DimensionResult, Verdict
from cri.scoring.dimensions.base import MetricDimension
from cri.scoring.rubrics import pas_check

if TYPE_CHECKING:
    from cri.adapter import MemoryAdapter
    from cri.judge import BinaryJudge
    from cri.models import GroundTruth

logger = logging.getLogger(__name__)


class ProfileAccuracyScore(MetricDimension):
    """Binary-verdict scorer for the Profile Accuracy Score dimension.

    All individual profile checks run concurrently for maximum throughput.
    """

    name: str = "PAS"
    description: str = (
        "Measures factual recall accuracy of stored persona details. "
        "Evaluates whether the memory system correctly captured and can "
        "retrieve specific profile attributes such as demographics, "
        "preferences, and explicitly stated facts."
    )

    async def score(
        self,
        adapter: MemoryAdapter,
        ground_truth: GroundTruth,
        judge: BinaryJudge,
    ) -> DimensionResult:
        """Evaluate persona recall accuracy across all profile dimensions."""

        async def _check_one(
            check_id: str,
            fact_texts: list[str],
            dim_display_name: str,
            dim_name: str,
            expected_value: str,
        ) -> dict[str, object]:
            result = await judge.judge_across_chunks(
                check_id,
                fact_texts,
                lambda chunk, _dim=dim_display_name, _val=expected_value: pas_check(  # type: ignore[misc]
                    dimension=_dim,
                    gold_answer=_val,
                    stored_facts=chunk,
                ),
            )
            passed = result.verdict == Verdict.YES
            logger.debug("PAS check %s: expected=%r verdict=%s", check_id, expected_value, result.verdict.value)
            return {
                "check_id": check_id,
                "dimension_name": dim_name,
                "expected_value": expected_value,
                "verdict": result.verdict.value,
                "passed": passed,
            }

        # Build all tasks up front.
        tasks: list[asyncio.Task[dict[str, object]]] = []

        for dim_name, profile_dim in ground_truth.final_profile.items():
            stored_facts = adapter.retrieve(profile_dim.query_topic)
            fact_texts = [sf.text for sf in stored_facts]

            if isinstance(profile_dim.value, list):
                values: list[str] = profile_dim.value
                for idx, val in enumerate(values):
                    cid = f"pas-{dim_name}-{idx}"
                    tasks.append(asyncio.create_task(_check_one(cid, fact_texts, profile_dim.dimension_name, dim_name, val)))
            else:
                cid = f"pas-{dim_name}"
                tasks.append(asyncio.create_task(_check_one(cid, fact_texts, profile_dim.dimension_name, dim_name, profile_dim.value)))

        if not tasks:
            return DimensionResult(dimension_name=self.name, score=0.0, passed_checks=0, total_checks=0, details=[])

        details_list = await asyncio.gather(*tasks)

        passed_count = sum(1 for d in details_list if d["passed"])
        total_count = len(details_list)
        score_val = passed_count / total_count if total_count > 0 else 0.0

        return DimensionResult(
            dimension_name=self.name,
            score=score_val,
            passed_checks=passed_count,
            total_checks=total_count,
            details=list(details_list),
        )
