"""Tests for the CRI performance profiling module.

Covers:
- InstrumentedAdapter protocol compliance and transparent delegation
- Timing measurement accuracy (positive latencies)
- Memory growth curve construction
- PerformanceProfiler.wrap_adapter / get_profile lifecycle
- Edge cases: no operations, single operation, many operations
- PerformanceTracker (original low-level tracker)
"""

from __future__ import annotations

import time

import pytest

from cri.adapter import MemoryAdapter
from cri.models import Message, PerformanceProfile, StoredFact
from cri.performance import (
    InstrumentedAdapter,
    LatencyRecord,
    PerformanceProfiler,
    PerformanceTracker,
)

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


class SimpleTestAdapter:
    """Minimal adapter for testing — stores user messages as facts."""

    def __init__(self) -> None:
        self._facts: list[StoredFact] = []

    def ingest(self, messages: list[Message]) -> None:
        for msg in messages:
            if msg.role == "user":
                self._facts.append(StoredFact(text=msg.content))

    def retrieve(self, topic: str) -> list[StoredFact]:
        return [f for f in self._facts if topic.lower() in f.text.lower()]

    def get_events(self) -> list[StoredFact]:
        return list(self._facts)


class SlowTestAdapter:
    """Adapter with artificial delays for timing verification."""

    def __init__(self, delay_ms: float = 5.0) -> None:
        self._facts: list[StoredFact] = []
        self._delay_s = delay_ms / 1000.0

    def ingest(self, messages: list[Message]) -> None:
        time.sleep(self._delay_s)
        for msg in messages:
            if msg.role == "user":
                self._facts.append(StoredFact(text=msg.content))

    def retrieve(self, topic: str) -> list[StoredFact]:
        time.sleep(self._delay_s)
        return [f for f in self._facts if topic.lower() in f.text.lower()]

    def get_events(self) -> list[StoredFact]:
        return list(self._facts)


class BrokenAdapter:
    """An adapter missing the retrieve method — should fail validation."""

    def ingest(self, messages: list[Message]) -> None:
        pass

    def get_events(self) -> list[StoredFact]:
        return []


def _make_messages(n: int = 3) -> list[Message]:
    """Create a small set of test messages."""
    msgs = [
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
    ]
    return msgs[:n]


# ---------------------------------------------------------------------------
# InstrumentedAdapter — Protocol compliance
# ---------------------------------------------------------------------------


class TestInstrumentedAdapterProtocol:
    """Verify that InstrumentedAdapter satisfies the MemoryAdapter protocol."""

    def test_isinstance_check(self) -> None:
        adapter = InstrumentedAdapter(SimpleTestAdapter())
        assert isinstance(adapter, MemoryAdapter)

    def test_has_ingest(self) -> None:
        adapter = InstrumentedAdapter(SimpleTestAdapter())
        assert callable(getattr(adapter, "ingest", None))

    def test_has_retrieve(self) -> None:
        adapter = InstrumentedAdapter(SimpleTestAdapter())
        assert callable(getattr(adapter, "retrieve", None))

    def test_has_get_events(self) -> None:
        adapter = InstrumentedAdapter(SimpleTestAdapter())
        assert callable(getattr(adapter, "get_events", None))


# ---------------------------------------------------------------------------
# InstrumentedAdapter — Transparent delegation
# ---------------------------------------------------------------------------


class TestInstrumentedAdapterDelegation:
    """Verify that InstrumentedAdapter transparently delegates to the wrapped adapter."""

    def test_ingest_delegates(self) -> None:
        inner = SimpleTestAdapter()
        adapter = InstrumentedAdapter(inner)
        msgs = _make_messages()
        adapter.ingest(msgs)
        # inner should have stored the user messages
        assert len(inner.get_events()) == 2

    def test_retrieve_delegates_and_returns(self) -> None:
        inner = SimpleTestAdapter()
        adapter = InstrumentedAdapter(inner)
        adapter.ingest(_make_messages())
        results = adapter.retrieve("software engineer")
        assert len(results) == 1
        assert "software engineer" in results[0].text.lower()

    def test_retrieve_returns_empty_for_unknown(self) -> None:
        inner = SimpleTestAdapter()
        adapter = InstrumentedAdapter(inner)
        adapter.ingest(_make_messages())
        results = adapter.retrieve("quantum physics")
        assert results == []

    def test_get_events_delegates(self) -> None:
        inner = SimpleTestAdapter()
        adapter = InstrumentedAdapter(inner)
        adapter.ingest(_make_messages())
        facts = adapter.get_events()
        assert len(facts) == 2

    def test_get_events_empty_before_ingest(self) -> None:
        adapter = InstrumentedAdapter(SimpleTestAdapter())
        assert adapter.get_events() == []

    def test_ingest_empty_list(self) -> None:
        adapter = InstrumentedAdapter(SimpleTestAdapter())
        adapter.ingest([])  # should not raise
        assert adapter.get_events() == []

    def test_multiple_ingestions(self) -> None:
        adapter = InstrumentedAdapter(SimpleTestAdapter())
        msgs = _make_messages()
        adapter.ingest(msgs[:2])
        adapter.ingest(msgs[2:])
        facts = adapter.get_events()
        assert len(facts) == 2  # 2 user messages

    def test_stored_fact_types(self) -> None:
        adapter = InstrumentedAdapter(SimpleTestAdapter())
        adapter.ingest(_make_messages())
        facts = adapter.get_events()
        for f in facts:
            assert isinstance(f, StoredFact)


# ---------------------------------------------------------------------------
# InstrumentedAdapter — Timing records
# ---------------------------------------------------------------------------


class TestInstrumentedAdapterTiming:
    """Verify that timing data is collected correctly."""

    def test_ingest_records_latency(self) -> None:
        adapter = InstrumentedAdapter(SimpleTestAdapter())
        adapter.ingest(_make_messages())
        assert len(adapter.ingest_records) == 1
        assert adapter.ingest_records[0].latency_ms >= 0.0
        assert adapter.ingest_records[0].message_count == 3

    def test_retrieve_records_latency(self) -> None:
        adapter = InstrumentedAdapter(SimpleTestAdapter())
        adapter.ingest(_make_messages())
        adapter.retrieve("software")
        assert len(adapter.retrieve_records) == 1
        assert adapter.retrieve_records[0].latency_ms >= 0.0
        assert adapter.retrieve_records[0].query == "software"

    def test_get_events_records_latency(self) -> None:
        adapter = InstrumentedAdapter(SimpleTestAdapter())
        adapter.ingest(_make_messages())
        adapter.get_events()
        assert len(adapter.get_events_records) == 1
        assert adapter.get_events_records[0].latency_ms >= 0.0
        assert adapter.get_events_records[0].fact_count == 2

    def test_multiple_retrieves_recorded(self) -> None:
        adapter = InstrumentedAdapter(SimpleTestAdapter())
        adapter.ingest(_make_messages())
        adapter.retrieve("software")
        adapter.retrieve("hiking")
        adapter.retrieve("unknown")
        assert len(adapter.retrieve_records) == 3

    def test_slow_adapter_measures_delay(self) -> None:
        adapter = InstrumentedAdapter(SlowTestAdapter(delay_ms=10.0))
        adapter.ingest(_make_messages())
        # Ingest should take at least ~10ms
        assert adapter.ingest_records[0].latency_ms >= 5.0

    def test_slow_retrieve_measures_delay(self) -> None:
        adapter = InstrumentedAdapter(SlowTestAdapter(delay_ms=50.0))
        adapter.ingest(_make_messages())
        adapter.retrieve("software")
        assert adapter.retrieve_records[0].latency_ms >= 10.0

    def test_total_messages_ingested_tracked(self) -> None:
        adapter = InstrumentedAdapter(SimpleTestAdapter())
        adapter.ingest(_make_messages(2))
        adapter.ingest(_make_messages(3))
        assert adapter.total_messages_ingested == 5


# ---------------------------------------------------------------------------
# InstrumentedAdapter — Memory growth curve
# ---------------------------------------------------------------------------


class TestMemoryGrowthCurve:
    """Verify the memory growth curve is built correctly."""

    def test_growth_curve_after_single_ingest(self) -> None:
        adapter = InstrumentedAdapter(SimpleTestAdapter())
        adapter.ingest(_make_messages())
        assert len(adapter.memory_growth_curve) == 1
        msg_count, fact_count = adapter.memory_growth_curve[0]
        assert msg_count == 3  # 3 messages ingested
        assert fact_count == 2  # 2 user messages stored as facts

    def test_growth_curve_after_multiple_ingests(self) -> None:
        adapter = InstrumentedAdapter(SimpleTestAdapter())
        # First batch: 2 messages (1 user, 1 assistant)
        adapter.ingest(_make_messages(2))
        # Second batch: 1 message (user)
        adapter.ingest(_make_messages(3)[2:])

        assert len(adapter.memory_growth_curve) == 2
        # After first ingest: 2 messages total, 1 user fact
        assert adapter.memory_growth_curve[0] == (2, 1)
        # After second ingest: 3 messages total, 2 user facts
        assert adapter.memory_growth_curve[1] == (3, 2)

    def test_growth_curve_empty_ingest(self) -> None:
        adapter = InstrumentedAdapter(SimpleTestAdapter())
        adapter.ingest([])
        assert len(adapter.memory_growth_curve) == 1
        assert adapter.memory_growth_curve[0] == (0, 0)

    def test_growth_curve_tuples(self) -> None:
        adapter = InstrumentedAdapter(SimpleTestAdapter())
        adapter.ingest(_make_messages())
        for point in adapter.memory_growth_curve:
            assert isinstance(point, tuple)
            assert len(point) == 2
            assert isinstance(point[0], int)
            assert isinstance(point[1], int)


# ---------------------------------------------------------------------------
# PerformanceProfiler — wrap_adapter
# ---------------------------------------------------------------------------


class TestPerformanceProfilerWrap:
    """Test PerformanceProfiler.wrap_adapter()."""

    def test_wrap_returns_instrumented_adapter(self) -> None:
        profiler = PerformanceProfiler()
        result = profiler.wrap_adapter(SimpleTestAdapter())
        assert isinstance(result, InstrumentedAdapter)

    def test_wrapped_satisfies_protocol(self) -> None:
        profiler = PerformanceProfiler()
        result = profiler.wrap_adapter(SimpleTestAdapter())
        assert isinstance(result, MemoryAdapter)

    def test_wrap_rejects_broken_adapter(self) -> None:
        profiler = PerformanceProfiler()
        with pytest.raises(TypeError, match="retrieve"):
            profiler.wrap_adapter(BrokenAdapter())

    def test_wrap_rejects_non_adapter(self) -> None:
        profiler = PerformanceProfiler()
        with pytest.raises(TypeError):
            profiler.wrap_adapter("not an adapter")

    def test_wrap_rejects_none(self) -> None:
        profiler = PerformanceProfiler()
        with pytest.raises(TypeError):
            profiler.wrap_adapter(None)


# ---------------------------------------------------------------------------
# PerformanceProfiler — get_profile
# ---------------------------------------------------------------------------


class TestPerformanceProfilerGetProfile:
    """Test PerformanceProfiler.get_profile()."""

    def test_get_profile_returns_performance_profile(self) -> None:
        profiler = PerformanceProfiler()
        adapter = profiler.wrap_adapter(SimpleTestAdapter())
        adapter.ingest(_make_messages())
        adapter.retrieve("software")
        profile = profiler.get_profile()
        assert isinstance(profile, PerformanceProfile)

    def test_get_profile_without_wrap_raises(self) -> None:
        profiler = PerformanceProfiler()
        with pytest.raises(RuntimeError, match="No adapter has been wrapped"):
            profiler.get_profile()

    def test_profile_ingest_latency(self) -> None:
        profiler = PerformanceProfiler()
        adapter = profiler.wrap_adapter(SimpleTestAdapter())
        adapter.ingest(_make_messages())
        profile = profiler.get_profile()
        # Average per-message latency should be non-negative
        assert profile.ingest_latency_ms >= 0.0

    def test_profile_query_latency_stats(self) -> None:
        profiler = PerformanceProfiler()
        adapter = profiler.wrap_adapter(SimpleTestAdapter())
        adapter.ingest(_make_messages())
        for topic in ["software", "hiking", "berlin", "cooking", "travel"]:
            adapter.retrieve(topic)
        profile = profiler.get_profile()
        assert profile.query_latency_avg_ms >= 0.0
        assert profile.query_latency_p95_ms >= 0.0
        assert profile.query_latency_p99_ms >= 0.0
        # p95 >= avg (not always true for small samples, but generally)
        # Just verify they are non-negative numbers

    def test_profile_total_facts_stored(self) -> None:
        profiler = PerformanceProfiler()
        adapter = profiler.wrap_adapter(SimpleTestAdapter())
        adapter.ingest(_make_messages())
        profile = profiler.get_profile()
        assert profile.total_facts_stored == 2

    def test_profile_memory_growth_curve(self) -> None:
        profiler = PerformanceProfiler()
        adapter = profiler.wrap_adapter(SimpleTestAdapter())
        adapter.ingest(_make_messages(2))
        adapter.ingest(_make_messages(3)[2:])
        profile = profiler.get_profile()
        assert len(profile.memory_growth_curve) == 2
        assert profile.memory_growth_curve[0] == (2, 1)
        assert profile.memory_growth_curve[1] == (3, 2)

    def test_profile_judge_defaults(self) -> None:
        profiler = PerformanceProfiler()
        adapter = profiler.wrap_adapter(SimpleTestAdapter())
        adapter.ingest(_make_messages())
        profile = profiler.get_profile()
        assert profile.judge_api_calls == 0
        assert profile.judge_total_cost_estimate is None

    def test_profile_judge_custom_values(self) -> None:
        profiler = PerformanceProfiler()
        adapter = profiler.wrap_adapter(SimpleTestAdapter())
        adapter.ingest(_make_messages())
        profiler.judge_api_calls = 42
        profiler.judge_total_cost_estimate = 1.23
        profile = profiler.get_profile()
        assert profile.judge_api_calls == 42
        assert profile.judge_total_cost_estimate == 1.23


# ---------------------------------------------------------------------------
# PerformanceProfiler — Edge cases
# ---------------------------------------------------------------------------


class TestPerformanceProfilerEdgeCases:
    """Test edge cases in profile generation."""

    def test_no_operations(self) -> None:
        """Profile with no ingests or queries should return zero latencies."""
        profiler = PerformanceProfiler()
        profiler.wrap_adapter(SimpleTestAdapter())
        profile = profiler.get_profile()
        assert profile.ingest_latency_ms == 0.0
        assert profile.query_latency_avg_ms == 0.0
        assert profile.query_latency_p95_ms == 0.0
        assert profile.query_latency_p99_ms == 0.0
        assert profile.total_facts_stored == 0

    def test_ingest_only_no_queries(self) -> None:
        """Profile with ingests but no queries."""
        profiler = PerformanceProfiler()
        adapter = profiler.wrap_adapter(SimpleTestAdapter())
        adapter.ingest(_make_messages())
        profile = profiler.get_profile()
        assert profile.ingest_latency_ms >= 0.0
        assert profile.query_latency_avg_ms == 0.0
        assert profile.total_facts_stored == 2

    def test_queries_only_no_ingest(self) -> None:
        """Profile with queries but no ingests."""
        profiler = PerformanceProfiler()
        adapter = profiler.wrap_adapter(SimpleTestAdapter())
        adapter.retrieve("anything")
        profile = profiler.get_profile()
        assert profile.ingest_latency_ms == 0.0
        assert profile.query_latency_avg_ms >= 0.0
        assert profile.total_facts_stored == 0

    def test_single_query(self) -> None:
        """Profile with exactly one query — p95/p99 should equal the single value."""
        profiler = PerformanceProfiler()
        adapter = profiler.wrap_adapter(SimpleTestAdapter())
        adapter.ingest(_make_messages())
        adapter.retrieve("software")
        profile = profiler.get_profile()
        # With a single query, avg == p95 == p99
        assert profile.query_latency_avg_ms == profile.query_latency_p95_ms
        assert profile.query_latency_avg_ms == profile.query_latency_p99_ms

    def test_many_queries(self) -> None:
        """Profile with many queries should compute reasonable percentiles."""
        profiler = PerformanceProfiler()
        adapter = profiler.wrap_adapter(SimpleTestAdapter())
        adapter.ingest(_make_messages())
        # Run 100 queries
        for i in range(100):
            adapter.retrieve(f"topic_{i}")
        profile = profiler.get_profile()
        assert profile.query_latency_avg_ms >= 0.0
        assert profile.query_latency_p95_ms >= 0.0
        assert profile.query_latency_p99_ms >= 0.0

    def test_empty_ingest(self) -> None:
        """Profile after ingesting an empty message list."""
        profiler = PerformanceProfiler()
        adapter = profiler.wrap_adapter(SimpleTestAdapter())
        adapter.ingest([])
        profile = profiler.get_profile()
        # 0 messages → per-message latency should be 0.0
        assert profile.ingest_latency_ms == 0.0
        assert profile.total_facts_stored == 0

    def test_per_message_ingest_latency_calculation(self) -> None:
        """Verify ingest latency is computed per-message, not per-call."""
        profiler = PerformanceProfiler()
        adapter = profiler.wrap_adapter(SlowTestAdapter(delay_ms=50.0))
        # 3 messages, ~50ms total delay
        adapter.ingest(_make_messages(3))
        profile = profiler.get_profile()
        # Per-message = ~50ms / 3 ≈ 16.7ms; should be at least 5.0ms
        assert profile.ingest_latency_ms >= 5.0


# ---------------------------------------------------------------------------
# PerformanceTracker (original low-level tracker) — existing tests preserved
# ---------------------------------------------------------------------------


class TestPerformanceTracker:
    """Tests for the low-level PerformanceTracker class."""

    def test_start_stop_records_latency(self) -> None:
        tracker = PerformanceTracker()
        tracker.start("test_op")
        time.sleep(0.005)
        latency = tracker.stop("test_op")
        assert latency >= 1.0  # at least 1ms

    def test_stop_without_start_raises(self) -> None:
        tracker = PerformanceTracker()
        with pytest.raises(ValueError, match="was not started"):
            tracker.stop("missing_op")

    def test_record_direct(self) -> None:
        tracker = PerformanceTracker()
        tracker.record("op", 42.5)
        records = tracker.get_records("op")
        assert len(records) == 1
        assert records[0].latency_ms == 42.5

    def test_get_records_filtered(self) -> None:
        tracker = PerformanceTracker()
        tracker.record("a", 1.0)
        tracker.record("b", 2.0)
        tracker.record("a", 3.0)
        assert len(tracker.get_records("a")) == 2
        assert len(tracker.get_records("b")) == 1

    def test_get_records_all(self) -> None:
        tracker = PerformanceTracker()
        tracker.record("a", 1.0)
        tracker.record("b", 2.0)
        assert len(tracker.get_records()) == 2

    def test_summary_statistics(self) -> None:
        tracker = PerformanceTracker()
        tracker.record("op", 10.0)
        tracker.record("op", 20.0)
        tracker.record("op", 30.0)
        summary = tracker.summary()
        assert "op" in summary
        assert summary["op"]["count"] == 3.0
        assert summary["op"]["mean_ms"] == 20.0
        assert summary["op"]["min_ms"] == 10.0
        assert summary["op"]["max_ms"] == 30.0

    def test_reset(self) -> None:
        tracker = PerformanceTracker()
        tracker.record("op", 10.0)
        tracker.start("running")
        tracker.reset()
        assert tracker.get_records() == []

    def test_latency_record_metadata(self) -> None:
        record = LatencyRecord(operation="test", latency_ms=5.0, metadata={"key": "value"})
        assert record.operation == "test"
        assert record.latency_ms == 5.0
        assert record.metadata == {"key": "value"}


# ---------------------------------------------------------------------------
# Integration — full pipeline
# ---------------------------------------------------------------------------


class TestIntegration:
    """End-to-end integration tests for the profiling pipeline."""

    def test_full_pipeline(self) -> None:
        """Run a complete profiling pipeline from wrap to profile."""
        profiler = PerformanceProfiler()
        adapter = profiler.wrap_adapter(SimpleTestAdapter())

        # Simulate a benchmark run
        batch1 = _make_messages(2)
        batch2 = _make_messages(3)[2:]

        adapter.ingest(batch1)
        adapter.ingest(batch2)

        adapter.retrieve("software engineer")
        adapter.retrieve("hiking")
        adapter.retrieve("unknown topic")

        all_facts = adapter.get_events()
        assert len(all_facts) == 2

        profiler.judge_api_calls = 10
        profiler.judge_total_cost_estimate = 0.05

        profile = profiler.get_profile()

        # Verify all fields are populated
        assert profile.ingest_latency_ms >= 0.0
        assert profile.query_latency_avg_ms >= 0.0
        assert profile.query_latency_p95_ms >= 0.0
        assert profile.query_latency_p99_ms >= 0.0
        assert profile.total_facts_stored == 2
        assert len(profile.memory_growth_curve) == 2
        assert profile.judge_api_calls == 10
        assert profile.judge_total_cost_estimate == 0.05

    def test_instrumented_adapter_does_not_affect_scoring(self) -> None:
        """Verify that using InstrumentedAdapter produces identical results
        to the raw adapter."""
        raw = SimpleTestAdapter()
        instrumented = InstrumentedAdapter(SimpleTestAdapter())

        msgs = _make_messages()
        raw.ingest(msgs)
        instrumented.ingest(msgs)

        # Query results should be identical
        raw_results = raw.retrieve("software")
        inst_results = instrumented.retrieve("software")
        assert len(raw_results) == len(inst_results)
        assert raw_results[0].text == inst_results[0].text

        # Full fact dump should match
        raw_all = raw.get_events()
        inst_all = instrumented.get_events()
        assert len(raw_all) == len(inst_all)
