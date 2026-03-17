"""No-Memory baseline adapter for the CRI Benchmark.

This adapter represents the **absolute lower bound** of memory system
performance.  It discards every ingested message, answers every query with an
empty result set, and reports an empty fact store.

Why it exists
~~~~~~~~~~~~~

Every meaningful memory system should score **strictly higher** than this
adapter.  If a system under test cannot beat ``NoMemoryAdapter``, it
provides no value over having no memory at all.  The adapter therefore
establishes the "floor" of the CRI score range and is essential for
calibrating how much value a real memory implementation adds.

Protocol compliance
~~~~~~~~~~~~~~~~~~~

``NoMemoryAdapter`` satisfies the :class:`~cri.adapter.MemoryAdapter`
protocol through structural subtyping (duck typing).  It exposes all
three required methods — ``ingest``, ``retrieve``, and ``get_events`` —
with compatible signatures.  No inheritance from ``MemoryAdapter`` is
needed::

    >>> from cri.adapter import MemoryAdapter
    >>> assert isinstance(NoMemoryAdapter(), MemoryAdapter)

External dependencies
~~~~~~~~~~~~~~~~~~~~~

**None.**  This adapter is implemented entirely with the Python standard
library and the CRI model layer.  It does not call any LLM, database, or
network service.
"""

from __future__ import annotations

from cri.models import Message, StoredFact


class NoMemoryAdapter:
    """Adapter with zero memory — discards all input and returns nothing.

    This is the **lower-bound baseline** for the CRI Benchmark.  It serves
    as a reference point: any memory system that cannot outperform this
    adapter is effectively useless.

    Behavior
    --------
    * :meth:`ingest` — Complete no-op.  All messages are silently discarded.
    * :meth:`retrieve` — Always returns an empty list regardless of the query.
    * :meth:`get_events` — Always returns an empty list.

    Example
    -------
    >>> adapter = NoMemoryAdapter()
    >>> adapter.ingest([Message(message_id=1, role="user",
    ...     content="I live in Berlin", timestamp="2025-01-01T10:00:00")])
    >>> adapter.retrieve("location")
    []
    >>> adapter.get_events()
    []
    """

    @property
    def name(self) -> str:
        """Human-readable name for this adapter."""
        return "no-memory"

    # ------------------------------------------------------------------
    # MemoryAdapter protocol methods
    # ------------------------------------------------------------------

    def ingest(self, messages: list[Message]) -> None:
        """Discard all messages — no information is ever stored.

        Parameters
        ----------
        messages : list[Message]
            Conversation messages to process.  Ignored entirely.
        """
        # Intentional no-op: this adapter stores nothing.

    def retrieve(self, query: str) -> list[StoredFact]:
        """Return an empty list — the adapter has no knowledge.

        Parameters
        ----------
        query : str
            The query string.  Ignored entirely.

        Returns
        -------
        list[StoredFact]
            Always an empty list.
        """
        return []

    def get_events(self) -> list[StoredFact]:
        """Return an empty list — no facts are stored.

        Returns
        -------
        list[StoredFact]
            Always an empty list.
        """
        return []

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset adapter state.  No-op since no state exists."""

    def __repr__(self) -> str:
        return "NoMemoryAdapter()"
