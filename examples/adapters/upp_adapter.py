"""UPP (Universal Personalization Protocol) adapter for CRI Benchmark.

Bridges the CRI :class:`~cri.adapter.MemoryAdapter` protocol to a UPP-compatible
memory system via the UPP Python SDK.  The adapter accepts a :class:`UPPClient`
instance (dependency injection) and translates the three required CRI methods
into UPP operations:

.. list-table::
   :header-rows: 1

   * - CRI Method
     - UPP Operation
   * - ``ingest(messages)``
     - ``upp/ingest`` (one call per user message)
   * - ``query(topic)``
     - ``upp/retrieve`` (filtered to ``valid`` events)
   * - ``get_all_facts()``
     - ``upp/get_events`` (all statuses for auditing)

The adapter satisfies :class:`~cri.adapter.MemoryAdapter` via structural
subtyping — no inheritance required.

Requirements
~~~~~~~~~~~~

::

    pip install cri-benchmark[upp]

Example usage
~~~~~~~~~~~~~

.. code-block:: python

    from upp.client import UPPClient
    from examples.adapters.upp_adapter import UPPAdapter

    client = UPPClient(ingest=..., retriever=..., ontology=...)
    adapter = UPPAdapter(client=client, entity_key="benchmark_user_001")

    adapter.ingest(messages)
    facts = adapter.query("current occupation")
    all_facts = adapter.get_all_facts()
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from cri.models import Message, StoredFact

__all__ = ["UPPAdapter"]

logger = logging.getLogger(__name__)


def _stored_event_to_fact(
    event: Any,
    *,
    include_audit_fields: bool = False,
) -> StoredFact:
    """Convert a UPP StoredEvent to a CRI StoredFact.

    Args:
        event: A UPP ``StoredEvent`` instance.
        include_audit_fields: If True, include ``entity_key`` and
            ``superseded_by`` in metadata (used by ``get_all_facts``).

    Returns:
        A :class:`~cri.models.StoredFact` with UPP metadata preserved.
    """
    metadata: dict[str, Any] = {
        "labels": event.labels,
        "confidence": event.confidence,
        "source_type": str(event.source_type),
        "status": str(event.status),
        "event_id": event.id,
        "created_at": event.created_at.isoformat(),
        "valid_from": event.valid_from,
        "valid_until": event.valid_until,
    }
    if include_audit_fields:
        metadata["entity_key"] = event.entity_key
        metadata["superseded_by"] = event.superseded_by
    return StoredFact(text=event.value, metadata=metadata)


class UPPAdapter:
    """CRI Benchmark adapter bridging to the UPP protocol.

    Translates CRI's synchronous ``MemoryAdapter`` protocol into async
    UPP client calls using a dedicated event loop.

    Parameters
    ----------
    client : UPPClient
        A configured UPP client instance (from ``upp.client.UPPClient``).
    entity_key : str
        The UPP entity key for all operations.  Each benchmark run should
        use a unique key to avoid cross-run contamination.
    include_assistant_messages : bool
        If ``True``, ingest assistant messages in addition to user messages.
        Defaults to ``False`` (matching the convention of other CRI adapters).
    """

    def __init__(
        self,
        client: Any,
        entity_key: str = "cri_benchmark_user",
        include_assistant_messages: bool = False,
    ) -> None:
        self._client = client
        self._entity_key = entity_key
        self._include_assistant = include_assistant_messages
        self._loop = asyncio.new_event_loop()

    # ------------------------------------------------------------------
    # MemoryAdapter protocol methods
    # ------------------------------------------------------------------

    def ingest(self, messages: list[Message]) -> None:
        """Feed conversation messages into the UPP memory system.

        Sends each qualifying message's content via ``upp/ingest``.  Only
        user messages are sent by default; assistant messages are skipped
        unless ``include_assistant_messages`` was set to ``True``.

        Parameters
        ----------
        messages : list[Message]
            Chronologically ordered conversation messages.
        """
        for msg in messages:
            if msg.role != "user" and not self._include_assistant:
                continue
            if not msg.content or not msg.content.strip():
                continue
            self._loop.run_until_complete(
                self._client.ingest(
                    entity_key=self._entity_key,
                    text=msg.content,
                )
            )

    def query(self, topic: str) -> list[StoredFact]:
        """Retrieve facts relevant to a topic from the UPP system.

        Calls ``upp/retrieve`` and returns only events with ``valid``
        status.  Superseded and staged events are excluded to maximize
        precision on CRI's DBU and CRQ dimensions.

        Parameters
        ----------
        topic : str
            A natural-language topic string.

        Returns
        -------
        list[StoredFact]
            Facts with UPP metadata (labels, confidence, status, etc.)
            in the ``metadata`` dict.
        """
        events = self._loop.run_until_complete(
            self._client.retrieve(
                entity_key=self._entity_key,
                query=topic,
            )
        )
        return [_stored_event_to_fact(event) for event in events if str(event.status) == "valid"]

    def get_all_facts(self) -> list[StoredFact]:
        """Return every fact stored in the UPP system for auditing.

        Calls ``upp/get_events`` and returns all events regardless of
        status (valid, staged, superseded).  Audit-specific fields
        (``entity_key``, ``superseded_by``) are included in metadata.

        Returns
        -------
        list[StoredFact]
            Complete fact store with full UPP metadata.
        """
        events = self._loop.run_until_complete(
            self._client.get_events(
                entity_key=self._entity_key,
            )
        )
        return [_stored_event_to_fact(event, include_audit_fields=True) for event in events]

    # ------------------------------------------------------------------
    # Utility methods (not part of the MemoryAdapter protocol)
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Delete all UPP events for the benchmark entity.

        Calls ``upp/delete_events`` with no event IDs to clear all stored
        events, providing a clean slate between benchmark runs.
        """
        self._loop.run_until_complete(
            self._client.delete_events(
                entity_key=self._entity_key,
                event_ids=None,
            )
        )

    def close(self) -> None:
        """Shut down the internal event loop."""
        if self._loop and not self._loop.is_closed():
            self._loop.close()

    def __repr__(self) -> str:
        return f"UPPAdapter(entity_key={self._entity_key!r})"
