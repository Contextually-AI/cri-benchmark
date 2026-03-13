"""Comprehensive tests for the CRI MemoryAdapter protocol and baseline adapters.

Test coverage:
- Protocol import and runtime-checkable verification
- Protocol compliance for compliant and non-compliant classes
- NoMemoryAdapter: protocol compliance, no-op behavior, edge cases
- FullContextAdapter: protocol compliance, user-only storage, query behavior
- Cross-adapter baseline comparison
- Edge cases: empty inputs, special characters, large batches
- Method signature verification
- Return type validation
"""

from __future__ import annotations

import sys
from pathlib import Path

from cri.adapter import MemoryAdapter
from cri.models import Message, StoredFact

# ---------------------------------------------------------------------------
# Ensure the examples/adapters directory is importable.
# ---------------------------------------------------------------------------
_ADAPTERS_DIR = str(Path(__file__).resolve().parent.parent / "examples" / "adapters")
if _ADAPTERS_DIR not in sys.path:
    sys.path.insert(0, _ADAPTERS_DIR)

from full_context_adapter import FullContextAdapter  # noqa: E402
from no_memory_adapter import NoMemoryAdapter  # noqa: E402

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


class CompliantAdapter:
    """A minimal adapter that satisfies the MemoryAdapter protocol via duck typing."""

    def __init__(self) -> None:
        self._facts: list[StoredFact] = []

    def ingest(self, messages: list[Message]) -> None:
        for msg in messages:
            if msg.role == "user":
                self._facts.append(StoredFact(text=msg.content, metadata={"day": msg.day}))

    def query(self, topic: str) -> list[StoredFact]:
        return [f for f in self._facts if topic.lower() in f.text.lower()]

    def get_all_facts(self) -> list[StoredFact]:
        return list(self._facts)


class MissingIngest:
    """Adapter missing the ingest method."""

    def query(self, topic: str) -> list[StoredFact]:
        return []

    def get_all_facts(self) -> list[StoredFact]:
        return []


class MissingQuery:
    """Adapter missing the query method."""

    def ingest(self, messages: list[Message]) -> None:
        pass

    def get_all_facts(self) -> list[StoredFact]:
        return []


class MissingGetAllFacts:
    """Adapter missing the get_all_facts method."""

    def ingest(self, messages: list[Message]) -> None:
        pass

    def query(self, topic: str) -> list[StoredFact]:
        return []


class EmptyClass:
    """A class with none of the required methods."""

    pass


def _make_messages() -> list[Message]:
    """Create a small set of test messages (3 user, 2 assistant)."""
    return [
        Message(
            message_id=1,
            role="user",
            content="I work as a software engineer in Berlin.",
            timestamp="2025-01-15T09:00:00Z",
            session_id="s1",
            day=1,
        ),
        Message(
            message_id=2,
            role="assistant",
            content="That's great! How long have you been there?",
            timestamp="2025-01-15T09:01:00Z",
            session_id="s1",
            day=1,
        ),
        Message(
            message_id=3,
            role="user",
            content="About three years. I love hiking on weekends.",
            timestamp="2025-01-15T09:02:00Z",
            session_id="s1",
            day=1,
        ),
        Message(
            message_id=4,
            role="assistant",
            content="Hiking sounds wonderful!",
            timestamp="2025-01-15T09:02:30Z",
            session_id="s1",
            day=1,
        ),
        Message(
            message_id=5,
            role="user",
            content="My favorite food is sushi.",
            timestamp="2025-01-16T09:00:00Z",
            session_id="s2",
            day=2,
        ),
    ]


def _make_large_message_batch(n: int) -> list[Message]:
    """Create a batch of n user messages."""
    return [
        Message(
            message_id=i,
            role="user",
            content=f"Fact number {i} about the user.",
            timestamp=f"2025-01-{(i % 28) + 1:02d}T10:00:00Z",
            session_id=f"s-{i}",
            day=(i % 30) + 1,
        )
        for i in range(1, n + 1)
    ]


# ---------------------------------------------------------------------------
# Import tests
# ---------------------------------------------------------------------------


class TestImports:
    """Verify that all protocol types can be imported cleanly."""

    def test_memory_adapter_importable(self) -> None:
        from cri.adapter import MemoryAdapter as MemoryAdapterCls

        assert MemoryAdapterCls is not None

    def test_message_importable(self) -> None:
        from cri.models import Message as MessageCls

        assert MessageCls is not None

    def test_stored_fact_importable(self) -> None:
        from cri.models import StoredFact as StoredFactCls

        assert StoredFactCls is not None

    def test_all_exports(self) -> None:
        from cri import adapter

        assert hasattr(adapter, "__all__")
        assert "MemoryAdapter" in adapter.__all__

    def test_no_memory_adapter_importable(self) -> None:
        assert NoMemoryAdapter is not None

    def test_full_context_adapter_importable(self) -> None:
        assert FullContextAdapter is not None


# ---------------------------------------------------------------------------
# Protocol meta tests
# ---------------------------------------------------------------------------


class TestProtocolMeta:
    """Verify the MemoryAdapter is a runtime-checkable Protocol with correct methods."""

    def test_is_runtime_checkable(self) -> None:
        assert getattr(MemoryAdapter, "_is_runtime_protocol", False)

    def test_protocol_has_ingest(self) -> None:
        assert hasattr(MemoryAdapter, "ingest")

    def test_protocol_has_query(self) -> None:
        assert hasattr(MemoryAdapter, "query")

    def test_protocol_has_get_all_facts(self) -> None:
        assert hasattr(MemoryAdapter, "get_all_facts")


# ---------------------------------------------------------------------------
# Protocol compliance tests
# ---------------------------------------------------------------------------


class TestProtocolCompliance:
    """Verify isinstance() checks for compliant and non-compliant classes."""

    def test_compliant_adapter_satisfies_protocol(self) -> None:
        assert isinstance(CompliantAdapter(), MemoryAdapter)

    def test_no_memory_adapter_satisfies_protocol(self) -> None:
        assert isinstance(NoMemoryAdapter(), MemoryAdapter)

    def test_full_context_adapter_satisfies_protocol(self) -> None:
        assert isinstance(FullContextAdapter(), MemoryAdapter)

    def test_missing_ingest_fails_protocol(self) -> None:
        assert not isinstance(MissingIngest(), MemoryAdapter)

    def test_missing_query_fails_protocol(self) -> None:
        assert not isinstance(MissingQuery(), MemoryAdapter)

    def test_missing_get_all_facts_fails_protocol(self) -> None:
        assert not isinstance(MissingGetAllFacts(), MemoryAdapter)

    def test_empty_class_fails_protocol(self) -> None:
        assert not isinstance(EmptyClass(), MemoryAdapter)

    def test_string_is_not_adapter(self) -> None:
        assert not isinstance("not an adapter", MemoryAdapter)

    def test_none_is_not_adapter(self) -> None:
        assert not isinstance(None, MemoryAdapter)

    def test_int_is_not_adapter(self) -> None:
        assert not isinstance(42, MemoryAdapter)

    def test_dict_is_not_adapter(self) -> None:
        assert not isinstance({}, MemoryAdapter)


# ---------------------------------------------------------------------------
# Method signature verification
# ---------------------------------------------------------------------------


class TestMethodSignatures:
    """Verify that adapter methods accept the correct argument types."""

    def test_ingest_accepts_message_list(self) -> None:
        adapter = CompliantAdapter()
        msgs = _make_messages()
        # Should not raise
        adapter.ingest(msgs)

    def test_ingest_returns_none(self) -> None:
        adapter = CompliantAdapter()
        result = adapter.ingest(_make_messages())
        assert result is None

    def test_query_accepts_string(self) -> None:
        adapter = CompliantAdapter()
        result = adapter.query("test topic")
        assert isinstance(result, list)

    def test_query_returns_list_of_stored_facts(self) -> None:
        adapter = CompliantAdapter()
        adapter.ingest(_make_messages())
        results = adapter.query("software engineer")
        assert isinstance(results, list)
        for item in results:
            assert isinstance(item, StoredFact)

    def test_get_all_facts_takes_no_args(self) -> None:
        adapter = CompliantAdapter()
        result = adapter.get_all_facts()
        assert isinstance(result, list)

    def test_get_all_facts_returns_list_of_stored_facts(self) -> None:
        adapter = CompliantAdapter()
        adapter.ingest(_make_messages())
        facts = adapter.get_all_facts()
        assert isinstance(facts, list)
        for fact in facts:
            assert isinstance(fact, StoredFact)


# ---------------------------------------------------------------------------
# Functional tests — CompliantAdapter
# ---------------------------------------------------------------------------


class TestCompliantAdapter:
    """Test a concrete MemoryAdapter implementation end-to-end."""

    def test_ingest_stores_user_messages(self) -> None:
        adapter = CompliantAdapter()
        adapter.ingest(_make_messages())
        # 3 user messages out of 5 total
        assert len(adapter.get_all_facts()) == 3

    def test_query_returns_relevant_facts(self) -> None:
        adapter = CompliantAdapter()
        adapter.ingest(_make_messages())
        results = adapter.query("software engineer")
        assert len(results) == 1
        assert "software engineer" in results[0].text.lower()

    def test_query_returns_empty_for_unknown_topic(self) -> None:
        adapter = CompliantAdapter()
        adapter.ingest(_make_messages())
        assert adapter.query("quantum physics") == []

    def test_get_all_facts_returns_all(self) -> None:
        adapter = CompliantAdapter()
        adapter.ingest(_make_messages())
        facts = adapter.get_all_facts()
        assert len(facts) == 3
        texts = [f.text for f in facts]
        assert any("software engineer" in t.lower() for t in texts)
        assert any("hiking" in t.lower() for t in texts)
        assert any("sushi" in t.lower() for t in texts)

    def test_get_all_facts_empty_before_ingest(self) -> None:
        adapter = CompliantAdapter()
        assert adapter.get_all_facts() == []

    def test_stored_fact_metadata(self) -> None:
        adapter = CompliantAdapter()
        adapter.ingest(_make_messages())
        facts = adapter.get_all_facts()
        for fact in facts:
            assert "day" in fact.metadata

    def test_ingest_with_empty_list(self) -> None:
        adapter = CompliantAdapter()
        adapter.ingest([])
        assert adapter.get_all_facts() == []

    def test_multiple_ingestions(self) -> None:
        adapter = CompliantAdapter()
        msgs = _make_messages()
        adapter.ingest(msgs[:2])
        adapter.ingest(msgs[2:])
        facts = adapter.get_all_facts()
        # msg 1 (user) + msg 3 (user) + msg 5 (user) = 3 facts
        assert len(facts) == 3


# ===================================================================
# NoMemoryAdapter tests
# ===================================================================


class TestNoMemoryAdapter:
    """Tests for the no-memory lower-bound baseline."""

    def test_protocol_compliance(self) -> None:
        assert isinstance(NoMemoryAdapter(), MemoryAdapter)

    def test_name_property(self) -> None:
        assert NoMemoryAdapter().name == "no-memory"

    def test_ingest_is_noop(self) -> None:
        adapter = NoMemoryAdapter()
        adapter.ingest(_make_messages())
        assert adapter.get_all_facts() == []

    def test_ingest_empty_list(self) -> None:
        adapter = NoMemoryAdapter()
        adapter.ingest([])
        assert adapter.get_all_facts() == []

    def test_query_returns_empty(self) -> None:
        adapter = NoMemoryAdapter()
        adapter.ingest(_make_messages())
        assert adapter.query("location") == []
        assert adapter.query("occupation") == []
        assert adapter.query("") == []

    def test_query_before_ingest(self) -> None:
        adapter = NoMemoryAdapter()
        assert adapter.query("anything") == []

    def test_get_all_facts_returns_empty(self) -> None:
        assert NoMemoryAdapter().get_all_facts() == []

    def test_reset_is_noop(self) -> None:
        adapter = NoMemoryAdapter()
        adapter.reset()  # should not raise
        assert adapter.get_all_facts() == []

    def test_repr(self) -> None:
        assert repr(NoMemoryAdapter()) == "NoMemoryAdapter()"

    def test_multiple_ingestions_still_empty(self) -> None:
        adapter = NoMemoryAdapter()
        msgs = _make_messages()
        adapter.ingest(msgs)
        adapter.ingest(msgs)
        adapter.ingest(msgs)
        assert adapter.get_all_facts() == []
        assert adapter.query("location") == []

    def test_large_batch_ingest(self) -> None:
        """Ingesting a large number of messages should not raise and store nothing."""
        adapter = NoMemoryAdapter()
        adapter.ingest(_make_large_message_batch(500))
        assert adapter.get_all_facts() == []

    def test_query_return_type_is_list(self) -> None:
        adapter = NoMemoryAdapter()
        result = adapter.query("anything")
        assert isinstance(result, list)

    def test_get_all_facts_return_type_is_list(self) -> None:
        adapter = NoMemoryAdapter()
        result = adapter.get_all_facts()
        assert isinstance(result, list)

    def test_ingest_returns_none(self) -> None:
        adapter = NoMemoryAdapter()
        result = adapter.ingest(_make_messages())
        assert result is None


# ===================================================================
# FullContextAdapter tests
# ===================================================================


class TestFullContextAdapter:
    """Tests for the full-context upper-bound baseline."""

    def test_protocol_compliance(self) -> None:
        assert isinstance(FullContextAdapter(), MemoryAdapter)

    def test_name_property(self) -> None:
        assert FullContextAdapter().name == "full-context"

    def test_ingest_stores_user_messages(self) -> None:
        adapter = FullContextAdapter()
        adapter.ingest(_make_messages())
        facts = adapter.get_all_facts()
        # 3 user messages out of 5
        assert len(facts) == 3

    def test_ingest_ignores_assistant_messages(self) -> None:
        adapter = FullContextAdapter()
        adapter.ingest(_make_messages())
        fact_texts = [f.text for f in adapter.get_all_facts()]
        assert "That's great! How long have you been there?" not in fact_texts
        assert "Hiking sounds wonderful!" not in fact_texts

    def test_stored_fact_content(self) -> None:
        adapter = FullContextAdapter()
        adapter.ingest(_make_messages())
        fact_texts = [f.text for f in adapter.get_all_facts()]
        assert "I work as a software engineer in Berlin." in fact_texts
        assert "About three years. I love hiking on weekends." in fact_texts
        assert "My favorite food is sushi." in fact_texts

    def test_stored_fact_metadata_has_source_message_id(self) -> None:
        adapter = FullContextAdapter()
        adapter.ingest(_make_messages())
        for fact in adapter.get_all_facts():
            assert "source_message_id" in fact.metadata
            assert "timestamp" in fact.metadata

    def test_query_returns_all_facts_regardless_of_topic(self) -> None:
        adapter = FullContextAdapter()
        adapter.ingest(_make_messages())
        r1 = adapter.query("location")
        r2 = adapter.query("food")
        r3 = adapter.query("completely unrelated topic")
        assert len(r1) == 3
        assert len(r2) == 3
        assert len(r3) == 3

    def test_query_returns_copies(self) -> None:
        adapter = FullContextAdapter()
        adapter.ingest(_make_messages())
        result1 = adapter.query("topic")
        result2 = adapter.query("topic")
        assert result1 == result2
        assert result1 is not result2

    def test_get_all_facts_returns_copies(self) -> None:
        adapter = FullContextAdapter()
        adapter.ingest(_make_messages())
        facts1 = adapter.get_all_facts()
        facts2 = adapter.get_all_facts()
        assert facts1 == facts2
        assert facts1 is not facts2

    def test_ingest_empty_list(self) -> None:
        adapter = FullContextAdapter()
        adapter.ingest([])
        assert adapter.get_all_facts() == []

    def test_query_before_ingest(self) -> None:
        assert FullContextAdapter().query("anything") == []

    def test_get_all_facts_before_ingest(self) -> None:
        assert FullContextAdapter().get_all_facts() == []

    def test_reset_clears_state(self) -> None:
        adapter = FullContextAdapter()
        adapter.ingest(_make_messages())
        assert len(adapter.get_all_facts()) == 3
        adapter.reset()
        assert adapter.get_all_facts() == []
        assert adapter.query("location") == []

    def test_multiple_ingestions_accumulate(self) -> None:
        adapter = FullContextAdapter()
        msgs = _make_messages()
        adapter.ingest(msgs)
        assert len(adapter.get_all_facts()) == 3
        adapter.ingest(msgs)
        assert len(adapter.get_all_facts()) == 6

    def test_reset_then_reingest(self) -> None:
        adapter = FullContextAdapter()
        adapter.ingest(_make_messages())
        adapter.reset()
        adapter.ingest(_make_messages()[:2])
        facts = adapter.get_all_facts()
        # First 2 messages: 1 user + 1 assistant → 1 fact
        assert len(facts) == 1
        assert facts[0].text == "I work as a software engineer in Berlin."

    def test_repr(self) -> None:
        assert "FullContextAdapter" in repr(FullContextAdapter())

    def test_repr_shows_fact_count(self) -> None:
        adapter = FullContextAdapter()
        adapter.ingest(_make_messages())
        assert "3" in repr(adapter)

    def test_stored_facts_are_stored_fact_instances(self) -> None:
        adapter = FullContextAdapter()
        adapter.ingest(_make_messages())
        for fact in adapter.get_all_facts():
            assert isinstance(fact, StoredFact)
        for fact in adapter.query("test"):
            assert isinstance(fact, StoredFact)

    def test_large_batch_ingest(self) -> None:
        """Ingesting 500 user messages should store all 500."""
        adapter = FullContextAdapter()
        adapter.ingest(_make_large_message_batch(500))
        assert len(adapter.get_all_facts()) == 500

    def test_ingest_returns_none(self) -> None:
        adapter = FullContextAdapter()
        result = adapter.ingest(_make_messages())
        assert result is None

    def test_special_characters_in_content(self) -> None:
        """Messages with unicode and special chars should be stored correctly."""
        msg = Message(
            message_id=1,
            role="user",
            content='I love 🎉 "party" & <fun> émojis!',
            timestamp="2025-01-01T10:00:00Z",
        )
        adapter = FullContextAdapter()
        adapter.ingest([msg])
        facts = adapter.get_all_facts()
        assert len(facts) == 1
        assert "🎉" in facts[0].text

    def test_only_user_role_messages_stored(self) -> None:
        """Even with multiple roles, only user messages are retained."""
        msgs = [
            Message(
                message_id=1,
                role="assistant",
                content="Hello!",
                timestamp="2025-01-01T10:00:00Z",
            ),
            Message(
                message_id=2,
                role="assistant",
                content="How are you?",
                timestamp="2025-01-01T10:00:01Z",
            ),
            Message(
                message_id=3,
                role="user",
                content="I'm a teacher.",
                timestamp="2025-01-01T10:00:02Z",
            ),
            Message(
                message_id=4,
                role="assistant",
                content="Nice!",
                timestamp="2025-01-01T10:00:03Z",
            ),
        ]
        adapter = FullContextAdapter()
        adapter.ingest(msgs)
        assert len(adapter.get_all_facts()) == 1
        assert adapter.get_all_facts()[0].text == "I'm a teacher."

    def test_preserves_message_order(self) -> None:
        """Facts should be stored in the order messages were ingested."""
        msgs = [
            Message(
                message_id=1,
                role="user",
                content="First fact",
                timestamp="2025-01-01T10:00:00Z",
            ),
            Message(
                message_id=2,
                role="user",
                content="Second fact",
                timestamp="2025-01-01T10:01:00Z",
            ),
            Message(
                message_id=3,
                role="user",
                content="Third fact",
                timestamp="2025-01-01T10:02:00Z",
            ),
        ]
        adapter = FullContextAdapter()
        adapter.ingest(msgs)
        texts = [f.text for f in adapter.get_all_facts()]
        assert texts == ["First fact", "Second fact", "Third fact"]


# ===================================================================
# Cross-adapter comparison tests
# ===================================================================


class TestBaselineComparison:
    """Tests that verify the relationship between the two baselines."""

    def test_full_context_stores_more_than_no_memory(self) -> None:
        no_mem = NoMemoryAdapter()
        full_ctx = FullContextAdapter()
        msgs = _make_messages()
        no_mem.ingest(msgs)
        full_ctx.ingest(msgs)
        assert len(full_ctx.get_all_facts()) > len(no_mem.get_all_facts())

    def test_both_handle_empty_gracefully(self) -> None:
        no_mem = NoMemoryAdapter()
        full_ctx = FullContextAdapter()
        no_mem.ingest([])
        full_ctx.ingest([])
        assert no_mem.get_all_facts() == []
        assert full_ctx.get_all_facts() == []

    def test_query_result_types_consistent(self) -> None:
        no_mem = NoMemoryAdapter()
        full_ctx = FullContextAdapter()
        msgs = _make_messages()
        no_mem.ingest(msgs)
        full_ctx.ingest(msgs)
        assert isinstance(no_mem.query("location"), list)
        assert isinstance(full_ctx.query("location"), list)
        for fact in full_ctx.query("location"):
            assert isinstance(fact, StoredFact)

    def test_both_satisfy_protocol(self) -> None:
        assert isinstance(NoMemoryAdapter(), MemoryAdapter)
        assert isinstance(FullContextAdapter(), MemoryAdapter)

    def test_no_memory_always_zero_facts(self) -> None:
        """NoMemory should always have 0 facts regardless of input size."""
        adapter = NoMemoryAdapter()
        for n in [0, 1, 10, 100]:
            adapter.ingest(_make_large_message_batch(n))
            assert len(adapter.get_all_facts()) == 0

    def test_full_context_fact_count_equals_user_messages(self) -> None:
        """FullContext should store exactly as many facts as user messages."""
        adapter = FullContextAdapter()
        msgs = _make_messages()
        user_count = sum(1 for m in msgs if m.role == "user")
        adapter.ingest(msgs)
        assert len(adapter.get_all_facts()) == user_count
