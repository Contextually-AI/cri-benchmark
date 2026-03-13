"""SFC — Selective Forgetting Capability.

Evaluates whether a memory system can appropriately *forget* ephemeral,
superseded, or session-contextual information while retaining facts that
should persist.

Formula::

    should_forget_score  = correctly_absent / total_should_forget
    should_remember_score = correctly_present / total_should_remember
    SFC = 0.6 × should_forget + 0.4 × should_remember

The dimension requires ``forgettable_facts`` in the ground truth — a list
of :class:`~cri.models.ForgettableFact` items that should have been
discarded by the end of the conversation.  Retention checks use
``final_profile`` from the existing ground truth.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING

from cri.models import (
    DimensionResult,
    ForgettableFact,
    ProfileDimension,
    Verdict,
)
from cri.scoring.dimensions.base import MetricDimension
from cri.scoring.rubrics import sfc_forgetting_check, sfc_retention_check

if TYPE_CHECKING:
    from cri.adapter import MemoryAdapter
    from cri.judge import BinaryJudge
    from cri.models import GroundTruth

logger = logging.getLogger(__name__)

# Sub-dimension weights as defined in the proposal.
_WEIGHT_FORGET = 0.60
_WEIGHT_REMEMBER = 0.40


class SFCDimension(MetricDimension):
    """Selective Forgetting Capability scorer.

    Evaluates two complementary aspects of memory hygiene:

    1. **Should-forget**: facts that were mentioned but should have been
       discarded (ephemeral states, session context, fully superseded).
    2. **Should-remember**: facts from the final profile that must persist.

    The composite score is a weighted average of the two sub-scores.
    """

    name: str = "SFC"
    description: str = (
        "Selective Forgetting Capability — measures whether the system "
        "appropriately forgets ephemeral information while retaining "
        "persistent facts."
    )

    async def score(
        self,
        adapter: MemoryAdapter,
        ground_truth: GroundTruth,
        judge: BinaryJudge,
    ) -> DimensionResult:
        """Score the adapter's selective forgetting behaviour.

        Returns a :class:`~cri.models.DimensionResult` with a score in
        [0.0, 1.0] representing the weighted combination of forgetting
        and retention quality.
        """
        forgettable = ground_truth.forgettable_facts
        profile_items = list(ground_truth.final_profile.items())

        # Early return when there is nothing to evaluate.
        if not forgettable and not profile_items:
            logger.info("SFC: no forgettable facts and no profile — returning 1.0 (vacuous).")
            return DimensionResult(
                dimension_name=self.name,
                score=1.0,
                passed_checks=0,
                total_checks=0,
                details=[],
            )

        # -- Retrieve all stored facts once ----------------------------------
        all_stored = adapter.get_all_facts()
        stored_texts = [sf.text for sf in all_stored]

        details: list[dict[str, object]] = []

        # -- Phase 1: Should-forget checks -----------------------------------
        forget_passed, forget_total = self._evaluate_forgetting(
            forgettable,
            stored_texts,
            judge,
            details,
        )

        # -- Phase 2: Should-remember checks ---------------------------------
        remember_passed, remember_total = self._evaluate_retention(
            profile_items,
            stored_texts,
            judge,
            details,
        )

        # -- Compute composite -----------------------------------------------
        forget_score = forget_passed / forget_total if forget_total > 0 else 1.0
        remember_score = remember_passed / remember_total if remember_total > 0 else 1.0
        composite = _WEIGHT_FORGET * forget_score + _WEIGHT_REMEMBER * remember_score

        total_passed = forget_passed + remember_passed
        total_checks = forget_total + remember_total

        details.append(
            {
                "summary": True,
                "forget_score": round(forget_score, 4),
                "remember_score": round(remember_score, 4),
                "composite": round(composite, 4),
                "weight_forget": _WEIGHT_FORGET,
                "weight_remember": _WEIGHT_REMEMBER,
            }
        )

        logger.info(
            "SFC: forget=%.4f (%d/%d) remember=%.4f (%d/%d) composite=%.4f",
            forget_score,
            forget_passed,
            forget_total,
            remember_score,
            remember_passed,
            remember_total,
            composite,
        )

        return DimensionResult(
            dimension_name=self.name,
            score=round(composite, 4),
            passed_checks=total_passed,
            total_checks=total_checks,
            details=details,
        )

    # ------------------------------------------------------------------
    # Sub-evaluators
    # ------------------------------------------------------------------

    @staticmethod
    def _evaluate_forgetting(
        forgettable_facts: Sequence[ForgettableFact],
        stored_texts: list[str],
        judge: BinaryJudge,
        details: list[dict[str, object]],
    ) -> tuple[int, int]:
        """Check that each forgettable fact has been discarded.

        A *YES* verdict means the fact is still present → **failure**.
        A *NO* verdict means the fact was correctly forgotten → **pass**.
        """
        passed = 0
        total = 0

        for ff in forgettable_facts:
            total += 1
            check_id = f"sfc-forget-{ff.fact_id}"
            prompt = sfc_forgetting_check(ff.text, ff.reason, stored_texts)
            result = judge.judge(check_id, prompt)
            # NO = correctly absent → pass
            check_passed = result.verdict is Verdict.NO

            if check_passed:
                passed += 1

            details.append(
                {
                    "check_id": check_id,
                    "sub_dimension": "should_forget",
                    "fact_id": ff.fact_id,
                    "fact_text": ff.text,
                    "reason": ff.reason,
                    "verdict": result.verdict.value,
                    "passed": check_passed,
                }
            )

            logger.debug(
                "SFC forget check %s: verdict=%s passed=%s",
                check_id,
                result.verdict.value,
                check_passed,
            )

        return passed, total

    @staticmethod
    def _evaluate_retention(
        profile_items: Sequence[tuple[str, ProfileDimension]],
        stored_texts: list[str],
        judge: BinaryJudge,
        details: list[dict[str, object]],
    ) -> tuple[int, int]:
        """Check that each final-profile fact is still present.

        A *YES* verdict means the fact is present → **pass**.
        """
        passed = 0
        total = 0

        for dim_name, profile_dim in profile_items:
            values = (
                profile_dim.value if isinstance(profile_dim.value, list) else [profile_dim.value]
            )

            for v_idx, value in enumerate(values):
                total += 1
                check_id = (
                    f"sfc-retain-{dim_name}-{v_idx}"
                    if len(values) > 1
                    else f"sfc-retain-{dim_name}"
                )
                prompt = sfc_retention_check(dim_name, value, stored_texts)
                result = judge.judge(check_id, prompt)
                check_passed = result.verdict is Verdict.YES

                if check_passed:
                    passed += 1

                details.append(
                    {
                        "check_id": check_id,
                        "sub_dimension": "should_remember",
                        "dimension_name": dim_name,
                        "expected_value": value,
                        "verdict": result.verdict.value,
                        "passed": check_passed,
                    }
                )

                logger.debug(
                    "SFC retain check %s: expected=%r verdict=%s",
                    check_id,
                    value,
                    result.verdict.value,
                )

        return passed, total
