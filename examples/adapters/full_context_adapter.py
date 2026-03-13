"""Full-Context baseline adapter for the CRI Benchmark.

This adapter stores every **user** message verbatim as a
:class:`~cri.models.StoredFact` and returns the entire fact store for
every query — without any filtering, ranking, or deduplication.

Why it exists
~~~~~~~~~~~~~

``FullContextAdapter`` represents the **upper bound for recall**.  Because
it retains every user message and returns all of them on every query, it
will never *miss* a relevant fact.  However, it will also return every
*irrelevant* fact, meaning its **precision** will be poor.

Together with :class:`NoMemoryAdapter` (the lower bound), this adapter
defines the performance envelope that meaningful memory systems should
operate within:

* A good system should approach ``FullContextAdapter``'s **recall** while
  maintaining much better **precision**.
* A good system should massively outperform ``NoMemoryAdapter`` across
  all dimensions.

Protocol compliance
~~~~~~~~~~~~~~~~~~~

``FullContextAdapter`` satisfies the :class:`~cri.adapter.MemoryAdapter`
protocol through structural subtyping.  No inheritance is needed::

    >>> from cri.adapter import MemoryAdapter
    >>> assert isinstance(FullContextAdapter(), MemoryAdapter)

External dependencies
~~~~~~~~~~~~~~~~~~~~~

**None.**  This adapter is implemented entirely with the Python standard
library and the CRI model layer.  It does not call any LLM, database, or
network service.
"""

from __future__ import annotations

from cri.models import Message, StoredFact


class FullContextAdapter:
    """Adapter that stores every user message and returns everything on query.

    This is the **upper-bound recall baseline** for the CRI Benchmark.
    It demonstrates the maximum possible recall (every fact is always
    returned) at the cost of zero precision filtering.

    Only messages with ``role == "user"`` are stored.  Assistant messages
    are assumed to carry no user-factual content and are therefore
    ignored.

    Behavior
    --------
    * :meth:`ingest` — Stores each user message as a ``StoredFact``
      with ``text`` set to the message's ``content``.  Assistant messages
      are skipped.
    * :meth:`query` — Returns **all** stored facts regardless of the
      topic string.  No filtering is performed.
    * :meth:`get_all_facts` — Returns all stored facts (identical to
      ``query`` in this adapter).

    Example
    -------
    >>> adapter = FullContextAdapter()
    >>> adapter.ingest([
    ...     Message(message_id=1, role="user",
    ...             content="I live in Berlin",
    ...             timestamp="2025-01-01T10:00:00"),
    ...     Message(message_id=2, role="assistant",
    ...             content="That's great!",
    ...             timestamp="2025-01-01T10:00:01"),
    ...     Message(message_id=3, role="user",
    ...             content="I work as an engineer",
    ...             timestamp="2025-01-01T10:00:02"),
    ... ])
    >>> len(adapter.get_all_facts())
    2
    >>> adapter.query("anything")[0].text
    'I live in Berlin'
    """

    def __init__(self) -> None:
        self._facts: list[StoredFact] = []

    @property
    def name(self) -> str:
        """Human-readable name for this adapter."""
        return "full-context"

    # ------------------------------------------------------------------
    # MemoryAdapter protocol methods
    # ------------------------------------------------------------------

    def ingest(self, messages: list[Message]) -> None:
        """Store each user message as a :class:`StoredFact`.

        Assistant messages are intentionally ignored — they are assumed
        to contain system-generated text rather than user-factual content.

        Parameters
        ----------
        messages : list[Message]
            Ordered list of conversation messages.  Only messages where
            ``role == "user"`` are retained.
        """
        for msg in messages:
            if msg.role == "user":
                self._facts.append(
                    StoredFact(
                        text=msg.content,
                        metadata={
                            "source_message_id": msg.message_id,
                            "timestamp": msg.timestamp,
                        },
                    )
                )

    def query(self, topic: str) -> list[StoredFact]:
        """Return **all** stored facts — no filtering is applied.

        This maximizes recall at the expense of precision: every relevant
        fact will be present, but so will every irrelevant one.

        Parameters
        ----------
        topic : str
            The query topic.  Ignored — all facts are returned regardless.

        Returns
        -------
        list[StoredFact]
            A copy of the complete internal fact list.
        """
        return list(self._facts)

    def get_all_facts(self) -> list[StoredFact]:
        """Return all stored facts.

        Returns
        -------
        list[StoredFact]
            A copy of the complete internal fact list.
        """
        return list(self._facts)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all stored facts, returning the adapter to its initial state."""
        self._facts.clear()

    def __repr__(self) -> str:
        return f"FullContextAdapter(facts={len(self._facts)})"
