"""Tests for the CRI Benchmark UPP adapter.

Verifies that :class:`UPPAdapter` correctly implements the
:class:`~cri.adapter.MemoryAdapter` protocol and properly bridges
CRI operations to UPP client calls.

All tests use a mocked UPP client — no running UPP server is required.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

# ---------------------------------------------------------------------------
# Skip entire module if upp is not installed
# ---------------------------------------------------------------------------
upp_models = pytest.importorskip("upp.models.events", reason="upp-python not installed")
upp_enums = pytest.importorskip("upp.models.enums", reason="upp-python not installed")

from upp.models.enums import EventStatus, SourceType  # noqa: E402
from upp.models.events import StoredEvent  # noqa: E402

# ---------------------------------------------------------------------------
# Ensure the examples/adapters directory is importable.
# ---------------------------------------------------------------------------
_ADAPTERS_DIR = str(Path(__file__).resolve().parent.parent / "examples" / "adapters")
if _ADAPTERS_DIR not in sys.path:
    sys.path.insert(0, _ADAPTERS_DIR)

from upp_adapter import UPPAdapter  # noqa: E402

from cri.adapter import MemoryAdapter  # noqa: E402
from cri.models import Message, StoredFact  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_client() -> MagicMock:
    """Create a mock UPP client with async methods."""
    client = MagicMock()
    client.ingest = AsyncMock(return_value=[])
    client.retrieve = AsyncMock(return_value=[])
    client.get_events = AsyncMock(return_value=[])
    client.delete_events = AsyncMock(return_value=0)
    return client


def _make_stored_event(
    value: str,
    *,
    status: EventStatus = EventStatus.VALID,
    labels: list[str] | None = None,
    event_id: str = "evt-001",
    confidence: float = 0.9,
    superseded_by: str | None = None,
) -> StoredEvent:
    """Create a UPP StoredEvent for testing."""
    return StoredEvent(
        value=value,
        labels=labels or ["who_occupation"],
        confidence=confidence,
        source_type=SourceType.USER_STATED,
        id=event_id,
        entity_key="test_user",
        status=status,
        created_at=datetime(2025, 1, 15, 10, 30, 0),
        superseded_by=superseded_by,
    )


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
    ]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_client() -> MagicMock:
    return _make_mock_client()


@pytest.fixture
def adapter(mock_client: MagicMock) -> UPPAdapter:
    a = UPPAdapter(client=mock_client, entity_key="test_user")
    yield a
    a.close()


@pytest.fixture
def messages() -> list[Message]:
    return _make_messages()


# ===================================================================
# Protocol compliance
# ===================================================================


class TestProtocolCompliance:
    """Verify the adapter satisfies the MemoryAdapter protocol."""

    def test_isinstance_check(self, adapter: UPPAdapter) -> None:
        assert isinstance(adapter, MemoryAdapter)

    def test_has_ingest_method(self, adapter: UPPAdapter) -> None:
        assert callable(getattr(adapter, "ingest", None))

    def test_has_query_method(self, adapter: UPPAdapter) -> None:
        assert callable(getattr(adapter, "query", None))

    def test_has_get_all_facts_method(self, adapter: UPPAdapter) -> None:
        assert callable(getattr(adapter, "get_all_facts", None))


# ===================================================================
# Ingest tests
# ===================================================================


class TestIngest:
    """Tests for the ingest method."""

    def test_ingest_sends_user_messages(
        self, adapter: UPPAdapter, mock_client: MagicMock, messages: list[Message]
    ) -> None:
        """Only user messages should be forwarded to the UPP client."""
        adapter.ingest(messages)
        # 2 user messages out of 3 total
        assert mock_client.ingest.call_count == 2
        calls = mock_client.ingest.call_args_list
        assert calls[0].kwargs["text"] == "I live in Berlin"
        assert calls[1].kwargs["text"] == "I work as a software engineer"

    def test_ingest_uses_correct_entity_key(
        self, adapter: UPPAdapter, mock_client: MagicMock, messages: list[Message]
    ) -> None:
        adapter.ingest(messages)
        for call in mock_client.ingest.call_args_list:
            assert call.kwargs["entity_key"] == "test_user"

    def test_ingest_filters_assistant_messages(
        self, adapter: UPPAdapter, mock_client: MagicMock, messages: list[Message]
    ) -> None:
        """Assistant messages should be skipped by default."""
        adapter.ingest(messages)
        for call in mock_client.ingest.call_args_list:
            assert "Berlin is a wonderful city" not in call.kwargs["text"]

    def test_ingest_includes_assistant_when_configured(
        self, mock_client: MagicMock, messages: list[Message]
    ) -> None:
        """When include_assistant_messages=True, all messages are sent."""
        adapter = UPPAdapter(
            client=mock_client,
            entity_key="test_user",
            include_assistant_messages=True,
        )
        adapter.ingest(messages)
        adapter.close()
        assert mock_client.ingest.call_count == 3

    def test_ingest_empty_list(self, adapter: UPPAdapter, mock_client: MagicMock) -> None:
        """Ingesting an empty list should not call the UPP client."""
        adapter.ingest([])
        mock_client.ingest.assert_not_called()

    def test_ingest_skips_empty_content(self, adapter: UPPAdapter, mock_client: MagicMock) -> None:
        """Messages with empty or whitespace-only content should be skipped."""
        messages = [
            Message(
                message_id=1,
                role="user",
                content="",
                timestamp="2025-01-01T10:00:00",
            ),
            Message(
                message_id=2,
                role="user",
                content="   ",
                timestamp="2025-01-01T10:00:05",
            ),
            Message(
                message_id=3,
                role="user",
                content="Real content",
                timestamp="2025-01-01T10:01:00",
            ),
        ]
        adapter.ingest(messages)
        assert mock_client.ingest.call_count == 1
        assert mock_client.ingest.call_args.kwargs["text"] == "Real content"


# ===================================================================
# Query tests
# ===================================================================


class TestQuery:
    """Tests for the query method."""

    def test_query_maps_events_to_facts(self, adapter: UPPAdapter, mock_client: MagicMock) -> None:
        """StoredEvents should be converted to StoredFacts."""
        mock_client.retrieve.return_value = [
            _make_stored_event("User is a software engineer"),
        ]
        facts = adapter.query("occupation")
        assert len(facts) == 1
        assert facts[0].text == "User is a software engineer"
        assert isinstance(facts[0], StoredFact)

    def test_query_metadata_populated(self, adapter: UPPAdapter, mock_client: MagicMock) -> None:
        """Fact metadata should include UPP event fields."""
        mock_client.retrieve.return_value = [
            _make_stored_event(
                "User lives in Berlin",
                labels=["where_home"],
                confidence=0.95,
                event_id="evt-456",
            ),
        ]
        facts = adapter.query("location")
        meta = facts[0].metadata
        assert meta["labels"] == ["where_home"]
        assert meta["confidence"] == 0.95
        assert meta["event_id"] == "evt-456"
        assert meta["status"] == "valid"
        assert "created_at" in meta

    def test_query_filters_superseded_events(
        self, adapter: UPPAdapter, mock_client: MagicMock
    ) -> None:
        """Only valid events should be returned; superseded ones are excluded."""
        mock_client.retrieve.return_value = [
            _make_stored_event("Lives in Berlin", event_id="evt-1"),
            _make_stored_event(
                "Lives in Munich",
                status=EventStatus.SUPERSEDED,
                event_id="evt-2",
                superseded_by="evt-1",
            ),
        ]
        facts = adapter.query("location")
        assert len(facts) == 1
        assert facts[0].text == "Lives in Berlin"

    def test_query_filters_staged_events(self, adapter: UPPAdapter, mock_client: MagicMock) -> None:
        """Staged events should also be excluded from query results."""
        mock_client.retrieve.return_value = [
            _make_stored_event("Confirmed fact", event_id="evt-1"),
            _make_stored_event(
                "Low confidence fact",
                status=EventStatus.STAGED,
                event_id="evt-2",
            ),
        ]
        facts = adapter.query("topic")
        assert len(facts) == 1
        assert facts[0].text == "Confirmed fact"

    def test_query_returns_empty_when_no_events(
        self, adapter: UPPAdapter, mock_client: MagicMock
    ) -> None:
        mock_client.retrieve.return_value = []
        facts = adapter.query("anything")
        assert facts == []

    def test_query_uses_correct_entity_key(
        self, adapter: UPPAdapter, mock_client: MagicMock
    ) -> None:
        adapter.query("occupation")
        mock_client.retrieve.assert_called_once_with(entity_key="test_user", query="occupation")


# ===================================================================
# get_all_facts tests
# ===================================================================


class TestGetAllFacts:
    """Tests for the get_all_facts method."""

    def test_returns_all_events_including_superseded(
        self, adapter: UPPAdapter, mock_client: MagicMock
    ) -> None:
        """get_all_facts must return all events regardless of status."""
        mock_client.get_events.return_value = [
            _make_stored_event("Lives in Berlin", event_id="evt-1"),
            _make_stored_event(
                "Lives in Munich",
                status=EventStatus.SUPERSEDED,
                event_id="evt-2",
                superseded_by="evt-1",
            ),
            _make_stored_event(
                "Low confidence fact",
                status=EventStatus.STAGED,
                event_id="evt-3",
            ),
        ]
        facts = adapter.get_all_facts()
        assert len(facts) == 3

    def test_audit_metadata_included(self, adapter: UPPAdapter, mock_client: MagicMock) -> None:
        """Audit-specific fields should be in metadata."""
        mock_client.get_events.return_value = [
            _make_stored_event(
                "Old location",
                status=EventStatus.SUPERSEDED,
                event_id="evt-old",
                superseded_by="evt-new",
            ),
        ]
        facts = adapter.get_all_facts()
        meta = facts[0].metadata
        assert meta["entity_key"] == "test_user"
        assert meta["superseded_by"] == "evt-new"
        assert meta["status"] == "superseded"

    def test_returns_empty_when_no_events(
        self, adapter: UPPAdapter, mock_client: MagicMock
    ) -> None:
        mock_client.get_events.return_value = []
        assert adapter.get_all_facts() == []

    def test_uses_correct_entity_key(self, adapter: UPPAdapter, mock_client: MagicMock) -> None:
        adapter.get_all_facts()
        mock_client.get_events.assert_called_once_with(entity_key="test_user")

    def test_all_facts_are_stored_fact_instances(
        self, adapter: UPPAdapter, mock_client: MagicMock
    ) -> None:
        mock_client.get_events.return_value = [
            _make_stored_event("Fact 1", event_id="evt-1"),
            _make_stored_event("Fact 2", event_id="evt-2"),
        ]
        for fact in adapter.get_all_facts():
            assert isinstance(fact, StoredFact)


# ===================================================================
# Reset tests
# ===================================================================


class TestReset:
    """Tests for the reset method."""

    def test_reset_calls_delete_events(self, adapter: UPPAdapter, mock_client: MagicMock) -> None:
        adapter.reset()
        mock_client.delete_events.assert_called_once_with(entity_key="test_user", event_ids=None)


# ===================================================================
# Misc tests
# ===================================================================


class TestMisc:
    """Miscellaneous adapter tests."""

    def test_repr(self, adapter: UPPAdapter) -> None:
        assert repr(adapter) == "UPPAdapter(entity_key='test_user')"

    def test_custom_entity_key(self, mock_client: MagicMock) -> None:
        adapter = UPPAdapter(client=mock_client, entity_key="custom_key_123")
        adapter.query("test")
        adapter.close()
        mock_client.retrieve.assert_called_once_with(entity_key="custom_key_123", query="test")

    def test_close_is_safe_to_call_twice(self, mock_client: MagicMock) -> None:
        adapter = UPPAdapter(client=mock_client)
        adapter.close()
        adapter.close()  # Should not raise
