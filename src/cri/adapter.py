"""Adapter interface for integrating memory systems with the CRI Benchmark.

The CRI Benchmark — Contextual Resonance Index — evaluates how well AI memory
systems maintain, update, and retrieve contextual knowledge over time.  To
benchmark **any** memory system, it only needs to satisfy the
:class:`MemoryAdapter` protocol defined in this module.

Alignment with the Universal Personalization Protocol (UPP)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``MemoryAdapter`` protocol is designed to align with the
`Universal Personalization Protocol (UPP) <https://pypi.org/project/upp-python/>`_.
Method names and semantics mirror UPP's client operations:

.. list-table::
   :header-rows: 1

   * - CRI Method
     - UPP Operation
     - Purpose
   * - ``ingest(messages)``
     - ``upp/ingest``
     - Feed conversation data into the memory system
   * - ``retrieve(query)``
     - ``upp/retrieve``
     - Retrieve facts relevant to a query
   * - ``get_events()``
     - ``upp/get_events``
     - Dump all stored facts for auditing

CRI adapts UPP's per-message, entity-scoped async interface into a
batch-oriented, synchronous protocol suitable for benchmark evaluation.
The return type :class:`~cri.models.StoredFact` is CRI's simplified view of
a UPP :class:`~upp.StoredEvent`.

Why a Protocol?
~~~~~~~~~~~~~~~

``MemoryAdapter`` is defined as a :class:`typing.Protocol` rather than an
abstract base class.  This gives CRI *structural subtyping* — sometimes called
"static duck typing" — which means:

* **No inheritance required.**  Your class does not need to subclass or even
  import ``MemoryAdapter``.  As long as it exposes the right methods with
  compatible signatures, it *is* a valid adapter.
* **Zero coupling.**  Memory system authors can implement the interface in
  their own package without taking a dependency on CRI at all.
* **Runtime checking.**  Because the protocol is decorated with
  ``@runtime_checkable``, the benchmark runner can verify adapter compliance
  at runtime with a simple ``isinstance(adapter, MemoryAdapter)`` check before
  starting an evaluation run.

The three methods
~~~~~~~~~~~~~~~~~

The protocol defines exactly three methods that together cover the full
benchmark lifecycle:

1. :meth:`ingest` — Feed a chronological list of conversation messages into
   the memory system.  The system is expected to process each message and
   extract / store any relevant facts.  Aligned with ``upp/ingest``.
2. :meth:`retrieve` — Retrieve stored facts that are semantically relevant to
   a given query string.  This is used by the benchmark's evaluation dimensions
   (PAS, DBU, MEI, TC, CRQ, QRP) to probe the memory system's understanding.
   Aligned with ``upp/retrieve``.
3. :meth:`get_events` — Return *every* stored fact, regardless of query
   relevance.  This powers memory-hygiene auditing — checking for phantom
   facts, duplication, or forgotten updates.  Aligned with ``upp/get_events``.

Integration overview
~~~~~~~~~~~~~~~~~~~~

A typical integration looks like this:

.. code-block:: python

    from cri.models import Message, StoredFact

    class MyMemoryAdapter:
        \"\"\"Adapter for the Acme Memory Engine.\"\"\"

        def __init__(self, engine: AcmeEngine) -> None:
            self._engine = engine

        def ingest(self, messages: list[Message]) -> None:
            for msg in messages:
                self._engine.process(msg.content, role=msg.role,
                                     ts=msg.timestamp)

        def retrieve(self, query: str) -> list[StoredFact]:
            results = self._engine.search(query)
            return [
                StoredFact(text=r.text, metadata={"score": r.score})
                for r in results
            ]

        def get_events(self) -> list[StoredFact]:
            return [
                StoredFact(text=f.text, metadata=f.meta)
                for f in self._engine.dump_all()
            ]

Because ``MemoryAdapter`` uses structural subtyping, ``MyMemoryAdapter``
satisfies the protocol automatically — no base class needed.  The CRI runner
will validate compliance at startup with::

    assert isinstance(adapter, MemoryAdapter), "Adapter does not satisfy MemoryAdapter protocol"

See the *Integration Guide* in the project documentation for a full
walkthrough.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from cri.models import Message, StoredFact

__all__ = [
    "MemoryAdapter",
]


@runtime_checkable
class MemoryAdapter(Protocol):
    """Protocol that any memory system must satisfy to be evaluated by CRI.

    ``MemoryAdapter`` defines the minimal surface area a memory system must
    expose so the CRI benchmark runner can:

    * feed it conversation data (:meth:`ingest`),
    * probe its knowledge through targeted queries (:meth:`retrieve`), and
    * audit the full contents of its memory store (:meth:`get_events`).

    Method names are aligned with the Universal Personalization Protocol (UPP)
    to provide a consistent interface across the ecosystem.

    Implementing this protocol
    --------------------------

    Because ``MemoryAdapter`` is a :class:`~typing.Protocol`, your class does
    **not** need to inherit from it.  Simply implement the three methods with
    compatible signatures and the protocol is satisfied.  Both static type
    checkers (``mypy``, ``pyright``) and the runtime ``isinstance()`` check
    will recognise your class as a valid adapter.

    If you *prefer* explicit inheritance for documentation clarity, you may
    still subclass ``MemoryAdapter`` — it works either way.

    Methods at a glance
    -------------------

    +-------------------+-------------------------------------------+
    | Method            | Purpose                                   |
    +===================+===========================================+
    | ``ingest``        | Process conversation messages and store    |
    |                   | extracted facts.  (``upp/ingest``)        |
    +-------------------+-------------------------------------------+
    | ``retrieve``      | Retrieve facts relevant to a query.       |
    |                   | (``upp/retrieve``)                        |
    +-------------------+-------------------------------------------+
    | ``get_events``    | Dump the entire fact store for auditing.  |
    |                   | (``upp/get_events``)                      |
    +-------------------+-------------------------------------------+

    Runtime checking
    ----------------

    The ``@runtime_checkable`` decorator enables runtime validation::

        >>> isinstance(my_adapter, MemoryAdapter)
        True

    The benchmark runner performs this check before starting a run to fail
    fast with a clear error message if the adapter is non-compliant.

    Example
    -------

    A minimal (no-op) adapter for testing::

        from cri.models import Message, StoredFact

        class NullAdapter:
            def ingest(self, messages: list[Message]) -> None:
                pass  # discard everything

            def retrieve(self, query: str) -> list[StoredFact]:
                return []  # know nothing

            def get_events(self) -> list[StoredFact]:
                return []  # store nothing

        assert isinstance(NullAdapter(), MemoryAdapter)
    """

    def ingest(self, messages: list[Message]) -> None:
        """Feed a sequence of conversation messages into the memory system.

        Aligned with ``upp/ingest``.  The benchmark runner calls this method
        once (or in batches) to provide the full conversation history that the
        memory system must process.  Messages are supplied in **chronological
        order** — sorted by ``message_id`` / ``timestamp`` — and each message
        includes:

        * ``role`` — whether the message comes from the *user* or the
          *assistant*.
        * ``content`` — the textual body of the message.
        * ``timestamp`` — an ISO-8601 string indicating when the message
          occurred within the simulated timeline.
        * ``session_id`` / ``day`` — optional grouping metadata.

        The memory system is expected to:

        1. Parse the messages for factual content about the user or relevant
           entities.
        2. Store extracted facts in its internal knowledge representation.
        3. Handle knowledge updates — if a newer message contradicts or
           supersedes an earlier one, the system should update accordingly.

        Parameters
        ----------
        messages : list[Message]
            An ordered list of :class:`~cri.models.Message` objects
            representing a conversation history.  The list is never empty
            when called by the benchmark runner.

        Returns
        -------
        None
            This method operates via side-effects on the memory system's
            internal state.

        Notes
        -----
        * Messages may span multiple simulated days and sessions.
        * The same message list is never passed twice in a single benchmark
          run; however, the runner may call ``ingest`` multiple times with
          different batches during certain evaluation modes.
        * Latency of this method is measured and reported in the benchmark's
          :class:`~cri.models.PerformanceProfile`.
        """
        ...

    def retrieve(self, query: str) -> list[StoredFact]:
        """Retrieve stored facts that are relevant to the given query.

        Aligned with ``upp/retrieve``.  After ingestion, the benchmark runner
        calls ``retrieve`` with various query strings to evaluate how well the
        memory system retrieves pertinent knowledge.  Queries are aligned with
        the benchmark's evaluation dimensions:

        * **PAS** (Persona Accuracy Score) — queries about profile attributes
          like occupation, hobbies, or location.
        * **DBU** (Dynamic Belief Updating) — queries about facts that changed
          during the conversation.
        * **MEI** (Memory Efficiency Index) — evaluates storage coverage
          and efficiency against ground truth.
        * **TC** (Temporal Coherence) — queries about time-sensitive facts.
        * **CRQ** (Conflict Resolution Quality) — queries targeting areas
          where contradictory information was introduced.
        * **QRP** (Query Response Precision) — general retrieval precision
          probes.

        The system should return **only** facts it considers relevant to the
        query.  Returning irrelevant facts penalises precision scores;
        missing relevant facts penalises recall.

        Parameters
        ----------
        query : str
            A natural-language query string describing what information is
            being requested (e.g., ``"current occupation"``,
            ``"dietary preferences"``).

        Returns
        -------
        list[StoredFact]
            A list of :class:`~cri.models.StoredFact` objects that the memory
            system considers relevant to *query*.  May be empty if the system
            has no relevant knowledge.  Each ``StoredFact`` contains:

            * ``text`` — a textual representation of the fact.
            * ``metadata`` — an optional dict with system-specific metadata
              (confidence scores, timestamps, provenance, etc.).

        Notes
        -----
        * Query latency is measured and included in the performance profile.
        * The benchmark does **not** prescribe how relevance is determined —
          that is entirely up to the memory system's implementation.
        """
        ...

    def get_events(self) -> list[StoredFact]:
        """Return every fact currently stored in the memory system.

        Aligned with ``upp/get_events``.  This method provides a **complete
        dump** of the memory system's internal fact store, regardless of any
        query or relevance filter.  The benchmark runner uses it for **memory
        hygiene auditing**:

        * **Noise filtering** — checking whether the system stored irrelevant
          conversational noise (e.g., greetings, filler) as facts.
        * **Duplication detection** — identifying redundant or near-duplicate
          stored facts.
        * **Completeness verification** — ensuring that important signal
          messages resulted in stored facts.
        * **Memory growth analysis** — tracking how the number of stored facts
          evolves as more messages are ingested.
        * **Phantom fact detection** — identifying facts that were never
          present in the conversation.

        Parameters
        ----------
        (none)

        Returns
        -------
        list[StoredFact]
            A complete list of all :class:`~cri.models.StoredFact` objects
            currently held by the memory system.  The order is not prescribed
            but chronological or insertion-order is recommended for
            reproducibility.  Returns an empty list if no facts have been
            stored.

        Notes
        -----
        * This method may be called multiple times during a run — for example,
          once after ingestion and again after queries — to observe whether
          querying has side-effects on stored state.
        * Implementations should ensure this method returns a *snapshot* and
          does not modify internal state.
        """
        ...
