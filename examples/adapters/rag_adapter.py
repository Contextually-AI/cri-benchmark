"""Example RAG (Retrieval-Augmented Generation) adapter for CRI Benchmark.

This adapter demonstrates how a **simple vector-store-based RAG system**
performs on the CRI benchmark.  It uses `ChromaDB <https://www.trychroma.com/>`_
as an in-memory vector store with its default embedding function to chunk,
embed, store, and retrieve user messages.

Unlike more sophisticated ontology-based memory systems, this adapter
treats memory as a flat bag of text chunks — it does **not** reason about
entities, relationships, temporal validity, or belief updates.  It therefore
serves as a **lower-middle baseline**: better than having no memory at all
(see :mod:`no_memory_adapter`), but significantly weaker than systems that
maintain structured knowledge representations.

The adapter satisfies the :class:`~cri.adapter.MemoryAdapter` protocol,
which requires only three synchronous methods:

* :meth:`ingest` — stores user messages as individual chunks in ChromaDB.
* :meth:`query` — embeds a topic string and retrieves the top-*k* most
  similar chunks.
* :meth:`get_all_facts` — returns every stored chunk for memory auditing.

No LLM is used for response generation — this adapter is **pure retrieval**.
It returns raw retrieved chunks as :class:`~cri.models.StoredFact` objects,
letting the benchmark evaluation pipeline handle assessment.

Requirements
~~~~~~~~~~~~

ChromaDB is an optional dependency::

    pip install cri-benchmark[rag]

Example usage
~~~~~~~~~~~~~

.. code-block:: python

    from examples.adapters.rag_adapter import RAGAdapter
    from cri.adapter import MemoryAdapter

    adapter = RAGAdapter(n_results=10)
    assert isinstance(adapter, MemoryAdapter)

    adapter.ingest(messages)
    facts = adapter.query("current occupation")
    all_facts = adapter.get_all_facts()
"""

from __future__ import annotations

from typing import Any

from cri.models import Message, StoredFact

__all__ = ["RAGAdapter"]


class RAGAdapter:
    """A simple RAG adapter using ChromaDB as an in-memory vector store.

    This adapter chunks user messages into individual documents, embeds them
    using ChromaDB's default embedding function, and stores them in a
    collection configured for cosine similarity.  Queries embed the topic
    string and retrieve the top-*k* most similar chunks.

    The adapter satisfies the :class:`~cri.adapter.MemoryAdapter` protocol
    via structural subtyping — no inheritance required.

    Parameters
    ----------
    collection_name : str
        Name of the ChromaDB collection to create.  Defaults to
        ``"cri_rag_chunks"``.
    n_results : int
        Number of top-*k* results to return from :meth:`query`.  Defaults
        to ``5``.

    Raises
    ------
    ImportError
        If ``chromadb`` is not installed.  Install it with::

            pip install cri-benchmark[rag]

    Example
    -------
    >>> adapter = RAGAdapter(n_results=3)
    >>> adapter.ingest(messages)
    >>> results = adapter.query("dietary preferences")
    >>> len(results) <= 3
    True
    """

    def __init__(
        self,
        collection_name: str = "cri_rag_chunks",
        n_results: int = 5,
    ) -> None:
        try:
            import chromadb
        except ImportError:
            raise ImportError(
                "chromadb is required for RAGAdapter. "
                "Install it with: pip install cri-benchmark[rag]"
            ) from None

        self._n_results = n_results
        self._client: Any = chromadb.Client()
        self._collection: Any = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    # ------------------------------------------------------------------
    # MemoryAdapter protocol methods
    # ------------------------------------------------------------------

    def ingest(self, messages: list[Message]) -> None:
        """Feed conversation messages into the vector store.

        Only **user** messages are stored — assistant messages are discarded
        since they do not contain first-person factual claims about the user.
        Each user message becomes a single chunk (document) in ChromaDB,
        identified by its ``message_id``.

        Parameters
        ----------
        messages : list[Message]
            Chronologically ordered conversation messages.  Messages with
            ``role != "user"`` are silently skipped.

        Notes
        -----
        * Calling ``ingest`` multiple times is safe — ChromaDB's ``upsert``
          ensures that duplicate ``message_id`` values overwrite rather than
          duplicate.
        * Message metadata (``timestamp``, ``day``, ``session_id``) is
          preserved in the ChromaDB document metadata for potential
          downstream analysis.
        """
        if not messages:
            return

        documents: list[str] = []
        metadatas: list[dict[str, Any]] = []
        ids: list[str] = []

        for msg in messages:
            if msg.role != "user":
                continue
            # Skip empty content
            if not msg.content or not msg.content.strip():
                continue

            doc_id = f"msg_{msg.message_id}"
            documents.append(msg.content)
            metadatas.append(
                {
                    "message_id": msg.message_id,
                    "timestamp": msg.timestamp,
                    "day": msg.day if msg.day is not None else -1,
                    "session_id": msg.session_id or "",
                    "role": msg.role,
                }
            )
            ids.append(doc_id)

        if documents:
            self._collection.upsert(
                documents=documents,
                metadatas=metadatas,
                ids=ids,
            )

    def query(self, topic: str) -> list[StoredFact]:
        """Retrieve stored chunks most similar to the given topic.

        The topic string is embedded using ChromaDB's default embedding
        function and compared against all stored chunks via cosine
        similarity.  The top-*k* results (controlled by ``n_results``)
        are returned as :class:`~cri.models.StoredFact` objects.

        Parameters
        ----------
        topic : str
            A natural-language topic string describing the information
            being requested (e.g., ``"current occupation"``).

        Returns
        -------
        list[StoredFact]
            Up to ``n_results`` stored facts, ordered by descending
            similarity.  Each fact's ``metadata`` dict includes:

            * ``distance`` — the cosine distance (lower = more similar).
            * ``message_id`` — the originating message's ID.
            * ``timestamp`` — the originating message's timestamp.
            * ``day`` — the simulation day.
            * ``session_id`` — the session the message belonged to.

            Returns an empty list if the store is empty.
        """
        count = self._collection.count()
        if count == 0:
            return []

        # Retrieve at most n_results or however many documents exist
        k = min(self._n_results, count)
        results = self._collection.query(
            query_texts=[topic],
            n_results=k,
        )

        facts: list[StoredFact] = []
        if results and results["documents"] and results["documents"][0]:
            documents = results["documents"][0]
            distances = (
                results["distances"][0] if results.get("distances") else [None] * len(documents)
            )
            metadatas = (
                results["metadatas"][0] if results.get("metadatas") else [{}] * len(documents)
            )

            for doc, dist, meta in zip(documents, distances, metadatas, strict=False):
                fact_metadata: dict[str, Any] = {}
                if meta:
                    fact_metadata.update(meta)
                if dist is not None:
                    fact_metadata["distance"] = dist
                facts.append(StoredFact(text=doc, metadata=fact_metadata))

        return facts

    def get_all_facts(self) -> list[StoredFact]:
        """Return every chunk currently stored in the vector store.

        This method provides a complete dump of the ChromaDB collection,
        used by the benchmark for memory-hygiene auditing (noise detection,
        duplication analysis, completeness verification).

        Returns
        -------
        list[StoredFact]
            All stored facts in insertion order.  Each fact's ``metadata``
            dict includes the original message metadata (``message_id``,
            ``timestamp``, ``day``, ``session_id``).  Returns an empty
            list if no facts have been stored.
        """
        count = self._collection.count()
        if count == 0:
            return []

        # ChromaDB's get() returns all documents when no IDs/filters given
        all_data = self._collection.get()

        facts: list[StoredFact] = []
        if all_data and all_data["documents"]:
            documents = all_data["documents"]
            metadatas = (
                all_data["metadatas"] if all_data.get("metadatas") else [{}] * len(documents)
            )

            for doc, meta in zip(documents, metadatas, strict=False):
                fact_metadata: dict[str, Any] = {}
                if meta:
                    fact_metadata.update(meta)
                facts.append(StoredFact(text=doc, metadata=fact_metadata))

        return facts
