"""MEI -- Memory Efficiency Index.

Measures whether the memory system has retained all ground-truth facts,
regardless of how much additional information it stores.  Coverage is
measured by scanning all stored facts in fixed-size chunks **concurrently**.

Formula::

    MEI = covered_gt_facts / total_gt_facts  (pure coverage)
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from cri.models import DimensionResult
from cri.scoring.dimensions.base import MetricDimension
from cri.scoring.rubrics import MAX_FACTS_PER_PROMPT, mei_coverage_chunk_check

if TYPE_CHECKING:
    from cri.adapter import MemoryAdapter
    from cri.judge import BinaryJudge
    from cri.models import GroundTruth

logger = logging.getLogger(__name__)


class MEIDimension(MetricDimension):
    """Memory Efficiency Index scorer (concurrent chunk scanning)."""

    name: str = "MEI"
    description: str = "Memory Efficiency Index — measures whether all ground-truth facts are retained (pure coverage, storage volume not penalised)."

    async def score(
        self,
        adapter: MemoryAdapter,
        ground_truth: GroundTruth,
        judge: BinaryJudge,
    ) -> DimensionResult:
        """Score the adapter's ground-truth coverage."""
        gt_facts = _build_gt_facts(ground_truth)

        if not gt_facts:
            logger.warning("MEI: no ground-truth facts — returning 0.0 (no data).")
            return DimensionResult(
                dimension_name=self.name, score=0.0, passed_checks=0, total_checks=0, details=[]
            )

        all_stored = adapter.get_events()
        stored_texts = [sf.text for sf in all_stored]
        total_stored = len(all_stored)

        if total_stored == 0:
            logger.info("MEI: adapter stored 0 facts — returning 0.0.")
            return DimensionResult(
                dimension_name=self.name, score=0.0, passed_checks=0, total_checks=len(gt_facts), details=[]
            )

        # -- Concurrent chunk coverage scan ------------------------------------
        total_gt = len(gt_facts)
        chunk_size = MAX_FACTS_PER_PROMPT
        num_chunks = (total_stored + chunk_size - 1) // chunk_size

        async def _scan_chunk(chunk_idx: int) -> set[int]:
            start = chunk_idx * chunk_size
            chunk = stored_texts[start : start + chunk_size]
            prompt = mei_coverage_chunk_check(chunk, gt_facts)
            newly_covered = await judge.judge_coverage(f"mei-chunk-{chunk_idx}", prompt)
            # Guard against out-of-range indices.
            return {i for i in newly_covered if 0 <= i < total_gt}

        # Fire all chunks concurrently (semaphore throttles).
        chunk_results = await asyncio.gather(*[_scan_chunk(i) for i in range(num_chunks)])

        covered: set[int] = set()
        for result_set in chunk_results:
            covered |= result_set

        # -- Build per-GT-fact detail records ----------------------------------
        details: list[dict[str, object]] = []
        for idx, (gt_key, gt_value) in enumerate(gt_facts):
            passed = idx in covered
            details.append({
                "check_id": f"mei-coverage-{idx}",
                "gt_key": gt_key,
                "gt_value": gt_value,
                "verdict": "YES" if passed else "NO",
                "passed": passed,
            })

        covered_count = len(covered)
        coverage = covered_count / total_gt

        details.append({
            "summary": True,
            "total_stored_facts": total_stored,
            "total_gt_facts": total_gt,
            "covered_gt_facts": covered_count,
            "coverage": round(coverage, 4),
            "chunks_scanned": num_chunks,
        })

        logger.info(
            "MEI: covered=%d/%d gt facts across %d chunks, stored=%d total, MEI=%.4f",
            covered_count, total_gt, num_chunks, total_stored, coverage,
        )

        return DimensionResult(
            dimension_name=self.name,
            score=round(coverage, 4),
            passed_checks=covered_count,
            total_checks=total_gt,
            details=details,
        )


def _build_gt_facts(ground_truth: GroundTruth) -> list[tuple[str, str]]:
    """Extract (key, value) pairs from the ground truth's final profile."""
    facts: list[tuple[str, str]] = []
    for dim_name, profile_dim in ground_truth.final_profile.items():
        if isinstance(profile_dim.value, list):
            for v in profile_dim.value:
                facts.append((dim_name, v))
        else:
            facts.append((dim_name, profile_dim.value))
    return facts
