"""Tests for the CRI Benchmark baseline adapters.

Verifies that :class:`NoMemoryAdapter` and :class:`FullContextAdapter`
correctly implement the :class:`~cri.adapter.MemoryAdapter` protocol and
behave as documented lower-bound and upper-bound baselines.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Ensure the examples/adapters directory is importable.
# ---------------------------------------------------------------------------
_ADAPTERS_DIR = str(Path(__file__).resolve().parent.parent / "examples" / "adapters")
if _ADAPTERS_DIR not in sys.path:
    sys.path.insert(0, _ADAPTERS_DIR)

from full_context_adapter import FullContextAdapter  # noqa: E402
from no_memory_adapter import NoMemoryAdapter  # noqa: E402

from cri.adapter import MemoryAdapter  # noqa: E402
from cri.models import Message, StoredFact  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_messages() -> list[Message]:
    """Create a small set of mixed user/assistant messages for testing."""
    return [
        Message(
            message_id=1,
            role="user",
            content="I live in Berlin",
            timestamp="2025-01-01T10:00:00",
            session_id="s1",
            day=1,
        ),
        Message(
            message_id=2,
            role="assistant",
            content="That's great! Berlin is a wonderful city.",
            timestamp="2025-01-01T10:00:05",
            session_id="s1",
            day=1,
        ),
        Message(
            message_id=3,
            role="user",
            content="I work as a software engineer",
            timestamp="2025-01-01T10:01:00",
            session_id="s1",
            day=1,
        ),
        Message(
            message_id=4,
            role="assistant",
            content="Software engineering is a popular profession.",
            timestamp="2025-01-01T10:01:05",
            session_id="s1",
            day=1,
        ),
        Message(
            message_id=5,
            role="user",
            content="My favorite food is sushi",
            timestamp="2025-01-02T09:00:00",
            session_id="s2",
            day=2,
        ),
    ]


@pytest.fixture
def messages() -> list[Message]:
    return _make_messages()


@pytest.fixture
def no_memory_adapter() -> NoMemoryAdapter:
    return NoMemoryAdapter()


@pytest.fixture
def full_context_adapter() -> FullContextAdapter:
    return FullContextAdapter()


# ===================================================================
# NoMemoryAdapter tests
# ===================================================================


class TestNoMemoryAdapter:
    """Tests for the no-memory lower-bound baseline."""

    def test_protocol_compliance(self, no_memory_adapter: NoMemoryAdapter) -> None:
        """NoMemoryAdapter must satisfy the MemoryAdapter protocol."""
        assert isinstance(no_memory_adapter, MemoryAdapter)

    def test_name_property(self, no_memory_adapter: NoMemoryAdapter) -> None:
        assert no_memory_adapter.name == "no-memory"

    def test_ingest_is_noop(self, no_memory_adapter: NoMemoryAdapter, messages: list[Message]) -> None:
        """Ingest should succeed silently and store nothing."""
        no_memory_adapter.ingest(messages)
        assert no_memory_adapter.get_events() == []

    def test_ingest_empty_list(self, no_memory_adapter: NoMemoryAdapter) -> None:
        """Ingesting an empty list should not raise."""
        no_memory_adapter.ingest([])
        assert no_memory_adapter.get_events() == []

    def test_retrieve_returns_empty(self, no_memory_adapter: NoMemoryAdapter, messages: list[Message]) -> None:
        """Retrieve should always return an empty list."""
        no_memory_adapter.ingest(messages)
        assert no_memory_adapter.retrieve("location") == []
        assert no_memory_adapter.retrieve("occupation") == []
        assert no_memory_adapter.retrieve("") == []

    def test_get_events_returns_empty(self, no_memory_adapter: NoMemoryAdapter) -> None:
        """get_events should always return an empty list."""
        assert no_memory_adapter.get_events() == []

    def test_retrieve_before_ingest(self, no_memory_adapter: NoMemoryAdapter) -> None:
        """Retrieve before any ingestion should return empty list."""
        assert no_memory_adapter.retrieve("anything") == []

    def test_reset_is_noop(self, no_memory_adapter: NoMemoryAdapter) -> None:
        """Reset should succeed without error."""
        no_memory_adapter.reset()
        assert no_memory_adapter.get_events() == []

    def test_repr(self, no_memory_adapter: NoMemoryAdapter) -> None:
        assert repr(no_memory_adapter) == "NoMemoryAdapter()"

    def test_multiple_ingestions(self, no_memory_adapter: NoMemoryAdapter, messages: list[Message]) -> None:
        """Multiple ingestions should still result in no stored facts."""
        no_memory_adapter.ingest(messages)
        no_memory_adapter.ingest(messages)
        no_memory_adapter.ingest(messages)
        assert no_memory_adapter.get_events() == []
        assert no_memory_adapter.retrieve("location") == []


# ===================================================================
# FullContextAdapter tests
# ===================================================================


class TestFullContextAdapter:
    """Tests for the full-context upper-bound baseline."""

    def test_protocol_compliance(self, full_context_adapter: FullContextAdapter) -> None:
        """FullContextAdapter must satisfy the MemoryAdapter protocol."""
        assert isinstance(full_context_adapter, MemoryAdapter)

    def test_name_property(self, full_context_adapter: FullContextAdapter) -> None:
        assert full_context_adapter.name == "full-context"

    def test_ingest_stores_user_messages(self, full_context_adapter: FullContextAdapter, messages: list[Message]) -> None:
        """Only user messages should be stored as facts."""
        full_context_adapter.ingest(messages)
        facts = full_context_adapter.get_events()
        # 3 user messages out of 5 total
        assert len(facts) == 3

    def test_ingest_ignores_assistant_messages(self, full_context_adapter: FullContextAdapter, messages: list[Message]) -> None:
        """Assistant messages must not appear in stored facts."""
        full_context_adapter.ingest(messages)
        facts = full_context_adapter.get_events()
        fact_texts = [f.text for f in facts]
        # Verify no assistant messages leaked through
        assert "That's great! Berlin is a wonderful city." not in fact_texts
        assert "Software engineering is a popular profession." not in fact_texts

    def test_stored_fact_content(self, full_context_adapter: FullContextAdapter, messages: list[Message]) -> None:
        """Stored facts should have the user message content as text."""
        full_context_adapter.ingest(messages)
        facts = full_context_adapter.get_events()
        fact_texts = [f.text for f in facts]
        assert "I live in Berlin" in fact_texts
        assert "I work as a software engineer" in fact_texts
        assert "My favorite food is sushi" in fact_texts

    def test_stored_fact_metadata(self, full_context_adapter: FullContextAdapter, messages: list[Message]) -> None:
        """Stored facts should include source metadata."""
        full_context_adapter.ingest(messages)
        facts = full_context_adapter.get_events()
        for fact in facts:
            assert "source_message_id" in fact.metadata
            assert "timestamp" in fact.metadata

    def test_retrieve_returns_all_facts(self, full_context_adapter: FullContextAdapter, messages: list[Message]) -> None:
        """Retrieve should return ALL facts regardless of topic."""
        full_context_adapter.ingest(messages)
        # Any topic should return all 3 user-message facts
        result_location = full_context_adapter.retrieve("location")
        result_food = full_context_adapter.retrieve("food")
        result_random = full_context_adapter.retrieve("completely unrelated topic")
        assert len(result_location) == 3
        assert len(result_food) == 3
        assert len(result_random) == 3

    def test_retrieve_returns_copies(self, full_context_adapter: FullContextAdapter, messages: list[Message]) -> None:
        """Each retrieve call should return a new list (not the internal one)."""
        full_context_adapter.ingest(messages)
        result1 = full_context_adapter.retrieve("topic")
        result2 = full_context_adapter.retrieve("topic")
        # Same content but different list objects
        assert result1 == result2
        assert result1 is not result2

    def test_get_events_returns_copies(self, full_context_adapter: FullContextAdapter, messages: list[Message]) -> None:
        """get_events should return a new list each time."""
        full_context_adapter.ingest(messages)
        facts1 = full_context_adapter.get_events()
        facts2 = full_context_adapter.get_events()
        assert facts1 == facts2
        assert facts1 is not facts2

    def test_ingest_empty_list(self, full_context_adapter: FullContextAdapter) -> None:
        """Ingesting an empty list should not raise and store nothing."""
        full_context_adapter.ingest([])
        assert full_context_adapter.get_events() == []

    def test_retrieve_before_ingest(self, full_context_adapter: FullContextAdapter) -> None:
        """Retrieve before ingestion should return empty list."""
        assert full_context_adapter.retrieve("anything") == []

    def test_get_events_before_ingest(self, full_context_adapter: FullContextAdapter) -> None:
        """get_events before ingestion should return empty list."""
        assert full_context_adapter.get_events() == []

    def test_reset_clears_state(self, full_context_adapter: FullContextAdapter, messages: list[Message]) -> None:
        """Reset should clear all stored facts."""
        full_context_adapter.ingest(messages)
        assert len(full_context_adapter.get_events()) == 3
        full_context_adapter.reset()
        assert full_context_adapter.get_events() == []
        assert full_context_adapter.retrieve("location") == []

    def test_multiple_ingestions_accumulate(self, full_context_adapter: FullContextAdapter, messages: list[Message]) -> None:
        """Multiple ingest calls should accumulate facts."""
        full_context_adapter.ingest(messages)
        assert len(full_context_adapter.get_events()) == 3
        full_context_adapter.ingest(messages)
        assert len(full_context_adapter.get_events()) == 6

    def test_reset_then_reingest(self, full_context_adapter: FullContextAdapter, messages: list[Message]) -> None:
        """After reset, re-ingesting should work normally."""
        full_context_adapter.ingest(messages)
        full_context_adapter.reset()
        full_context_adapter.ingest(messages[:2])  # Only first 2 messages
        facts = full_context_adapter.get_events()
        # First 2 messages: 1 user + 1 assistant → 1 fact
        assert len(facts) == 1
        assert facts[0].text == "I live in Berlin"

    def test_repr(self, full_context_adapter: FullContextAdapter) -> None:
        assert "FullContextAdapter" in repr(full_context_adapter)

    def test_repr_shows_fact_count(self, full_context_adapter: FullContextAdapter, messages: list[Message]) -> None:
        full_context_adapter.ingest(messages)
        assert "3" in repr(full_context_adapter)

    def test_stored_facts_are_stored_fact_instances(self, full_context_adapter: FullContextAdapter, messages: list[Message]) -> None:
        """All returned facts must be StoredFact instances."""
        full_context_adapter.ingest(messages)
        for fact in full_context_adapter.get_events():
            assert isinstance(fact, StoredFact)
        for fact in full_context_adapter.retrieve("test"):
            assert isinstance(fact, StoredFact)


# ===================================================================
# Cross-adapter comparison tests
# ===================================================================


class TestBaselineComparison:
    """Tests that verify the relationship between the two baselines."""

    def test_full_context_stores_more_than_no_memory(self, messages: list[Message]) -> None:
        """FullContextAdapter should always store more facts than NoMemoryAdapter."""
        no_mem = NoMemoryAdapter()
        full_ctx = FullContextAdapter()

        no_mem.ingest(messages)
        full_ctx.ingest(messages)

        assert len(full_ctx.get_events()) > len(no_mem.get_events())

    def test_both_handle_empty_gracefully(self) -> None:
        """Both adapters should handle empty message lists without error."""
        no_mem = NoMemoryAdapter()
        full_ctx = FullContextAdapter()

        no_mem.ingest([])
        full_ctx.ingest([])

        assert no_mem.get_events() == []
        assert full_ctx.get_events() == []

    def test_retrieve_result_types_consistent(self, messages: list[Message]) -> None:
        """Both adapters should return list[StoredFact] from retrieve."""
        no_mem = NoMemoryAdapter()
        full_ctx = FullContextAdapter()

        no_mem.ingest(messages)
        full_ctx.ingest(messages)

        no_mem_result = no_mem.retrieve("location")
        full_ctx_result = full_ctx.retrieve("location")

        assert isinstance(no_mem_result, list)
        assert isinstance(full_ctx_result, list)
        for fact in full_ctx_result:
            assert isinstance(fact, StoredFact)
