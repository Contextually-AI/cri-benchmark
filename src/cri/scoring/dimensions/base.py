"""Base class for CRI dimension scorers.

All dimension implementations should subclass :class:`MetricDimension` and
define class-level ``name`` and ``description`` attributes, then implement
the ``score`` abstract method.

Example::

    from cri.scoring.dimensions.base import MetricDimension
    from cri.models import DimensionResult, GroundTruth

    class PersonaAccuracy(MetricDimension):
        name = "PAS"
        description = "Measures factual recall accuracy of stored persona details."

        async def score(self, adapter, ground_truth, judge):
            # ... dimension-specific evaluation logic ...
            return DimensionResult(
                dimension_name=self.name,
                score=0.85,
                passed_checks=17,
                total_checks=20,
            )
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from cri.models import DimensionResult, GroundTruth

if TYPE_CHECKING:
    from cri.adapter import MemoryAdapter
    from cri.judge import BinaryJudge


# ---------------------------------------------------------------------------
# New base class — MetricDimension
# ---------------------------------------------------------------------------


class MetricDimension(ABC):
    """Abstract base class for CRI evaluation dimensions.

    Each concrete subclass represents one of the CRI evaluation dimensions
    (e.g., PAS, DBU, TC) and encapsulates the scoring logic specific to
    that dimension.

    Subclasses **must** define two class-level attributes:

    - ``name`` — A short identifier string (e.g., ``"PAS"``, ``"DBU"``).
    - ``description`` — A human-readable sentence explaining what the
      dimension measures.

    Subclasses **must** implement the :meth:`score` abstract method, which
    evaluates a memory system (via its adapter) against ground truth using
    a binary judge.

    The ``score`` method follows a standard pattern:

    1. Derive dimension-specific checks from the ``ground_truth``.
    2. For each check, query the ``adapter`` and evaluate the response
       with the ``judge``.
    3. Aggregate binary verdicts into a :class:`~cri.models.DimensionResult`.
    """

    name: str
    """Short identifier for this dimension (e.g., ``'PAS'``, ``'DBU'``)."""

    description: str
    """Human-readable description of what this dimension measures."""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Validate that concrete subclasses define required class attributes.

        Validation is skipped for intermediate abstract subclasses (those
        that still have unimplemented abstract methods). Since
        ``__init_subclass__`` runs *before* ``ABCMeta.__new__`` populates
        ``__abstractmethods__``, we detect abstractness by inspecting whether
        any methods in the MRO still carry ``__isabstractmethod__ = True``.
        """
        super().__init_subclass__(**kwargs)
        # Check if any method on the class is still abstract.
        # We cannot rely on __abstractmethods__ because ABCMeta sets it
        # after __init_subclass__ returns.
        has_abstract = any(getattr(getattr(cls, name, None), "__isabstractmethod__", False) for name in dir(cls))
        if has_abstract:
            return
        for attr in ("name", "description"):
            if not isinstance(getattr(cls, attr, None), str) or not getattr(cls, attr):
                raise TypeError(f"Concrete MetricDimension subclass {cls.__name__!r} must define a non-empty class-level str attribute {attr!r}")

    @abstractmethod
    async def score(
        self,
        adapter: MemoryAdapter,
        ground_truth: GroundTruth,
        judge: BinaryJudge,
    ) -> DimensionResult:
        """Evaluate the memory system on this dimension.

        Implementations should:

        1. Derive evaluation checks from ``ground_truth`` that are relevant
           to this dimension (e.g., profile dimensions for PAS, belief
           changes for DBU).
        2. Query the ``adapter`` for each check — the adapter has already
           ingested all events before scoring begins.
        3. Use the ``judge`` to produce a binary verdict (YES / NO) for each
           check by comparing the adapter's response against the expected
           answer.
        4. Aggregate the binary verdicts into a
           :class:`~cri.models.DimensionResult` with:

           - ``score`` — ratio of passed checks to total checks (0.0–1.0)
           - ``passed_checks`` / ``total_checks`` — raw counts
           - ``details`` — per-check records for transparency

        Args:
            adapter: The memory system under evaluation. It exposes query
                methods that return the system's current knowledge.
            ground_truth: The expected outcomes for the benchmark dataset,
                containing the final profile, belief changes, conflict
                scenarios, temporal facts, and query-relevance pairs.
            judge: A binary verdict judge that evaluates individual
                responses against expected answers, returning YES or NO.

        Returns:
            A :class:`~cri.models.DimensionResult` containing the
            dimension-level score, check counts, and per-check details.
        """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
