"""Persona Accuracy Score (PAS) dimension scorer.

Measures how accurately the memory system recalls specific persona
details after ingesting events about a user. This is the most
fundamental dimension — a memory system must at minimum be able
to accurately recall what it has been told.

Two implementations are provided:

- :class:`ProfileAccuracyScore` — New binary-verdict scorer based on
  :class:`~cri.scoring.dimensions.base.MetricDimension`. Uses
  :class:`~cri.judge.BinaryJudge` with the :func:`~cri.scoring.rubrics.pas_check`
  rubric to evaluate each profile dimension independently.
"""

from __future__ import annotations

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


# ---------------------------------------------------------------------------
# New binary-verdict scorer — ProfileAccuracyScore
# ---------------------------------------------------------------------------


class ProfileAccuracyScore(MetricDimension):
    """Binary-verdict scorer for the Profile Accuracy Score dimension.

    Evaluates factual recall accuracy by checking whether each expected
    profile dimension value is semantically present in the facts stored
    by the memory system.

    **Algorithm**:

    For every :class:`~cri.models.ProfileDimension` in
    ``ground_truth.final_profile``:

    1. Query the adapter for facts related to the dimension's ``query_topic``.
    2. If the dimension value is a list (multi-value dimension), create one
       binary check per list element.  Otherwise create a single check.
    3. For each check, generate an LLM judge prompt via
       :func:`~cri.scoring.rubrics.pas_check` and evaluate it using the
       :class:`~cri.judge.BinaryJudge`.
    4. A ``YES`` verdict means the check passed (the fact was found).

    The dimension score is ``passed_checks / total_checks`` (0.0 when no
    checks exist).
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
        """Evaluate persona recall accuracy across all profile dimensions.

        Args:
            adapter: The memory system under evaluation.
            ground_truth: Expected outcomes containing the final profile.
            judge: A binary verdict judge for semantic evaluation.

        Returns:
            A :class:`~cri.models.DimensionResult` with the PAS score.
        """
        details: list[dict[str, object]] = []
        passed_count = 0
        total_count = 0

        for dim_name, profile_dim in ground_truth.final_profile.items():
            # Query the adapter for facts relevant to this profile dimension
            stored_facts = adapter.query(profile_dim.query_topic)
            fact_texts = [sf.text for sf in stored_facts]

            # Determine values to check
            if isinstance(profile_dim.value, list):
                is_multi = True
                values: list[str] = profile_dim.value
            else:
                is_multi = False
                values = [profile_dim.value]

            for idx, expected_value in enumerate(values):
                # Build check ID
                check_id = f"pas-{dim_name}-{idx}" if is_multi else f"pas-{dim_name}"

                # Generate the judge prompt using the pas_check rubric
                prompt = pas_check(
                    dimension=profile_dim.dimension_name,
                    gold_answer=expected_value,
                    stored_facts=fact_texts,
                )

                # Evaluate with the binary judge (synchronous call)
                result = judge.judge(check_id, prompt)
                passed = result.verdict == Verdict.YES

                if passed:
                    passed_count += 1
                total_count += 1

                details.append(
                    {
                        "check_id": check_id,
                        "dimension_name": dim_name,
                        "expected_value": expected_value,
                        "verdict": result.verdict.value,
                        "passed": passed,
                    }
                )

                logger.debug(
                    "PAS check %s: expected=%r verdict=%s",
                    check_id,
                    expected_value,
                    result.verdict.value,
                )

        score = passed_count / total_count if total_count > 0 else 0.0

        return DimensionResult(
            dimension_name=self.name,
            score=score,
            passed_checks=passed_count,
            total_checks=total_count,
            details=details,
        )
