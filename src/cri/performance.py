"""Performance tracking and profiling for CRI Benchmark runs.

This module provides two layers of performance instrumentation:

1. **Low-level tracking** — :class:`PerformanceTracker` records individual
   latency measurements with manual start/stop semantics.

2. **Adapter instrumentation** — :class:`InstrumentedAdapter` is a transparent
   decorator that wraps any :class:`~cri.adapter.MemoryAdapter`-compatible
   object, automatically measuring ``ingest``, ``query``, and
   ``get_all_facts`` call timings via :func:`time.monotonic`.

3. **Profiling** — :class:`PerformanceProfiler` orchestrates the wrapping and
   produces a :class:`~cri.models.PerformanceProfile` with latency statistics
   and memory growth curves.

Typical usage::

    from cri.performance import PerformanceProfiler

    profiler = PerformanceProfiler()
    instrumented = profiler.wrap_adapter(my_adapter)

    # Use `instrumented` in place of `my_adapter` everywhere — it satisfies
    # the MemoryAdapter protocol and is transparent to the scoring engine.
    instrumented.ingest(messages)
    results = instrumented.query("occupation")

    profile = profiler.get_profile()
    print(profile.query_latency_avg_ms)
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field
from typing import Any

from cri.models import Message, PerformanceProfile, StoredFact

# ---------------------------------------------------------------------------
# Low-level latency tracking (retained from original implementation)
# ---------------------------------------------------------------------------


@dataclass
class LatencyRecord:
    """A single latency measurement."""

    operation: str
    latency_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)


class PerformanceTracker:
    """Tracks performance metrics during a benchmark run.

    Records latency for event ingestion and query operations,
    computing summary statistics.

    Example::

        tracker = PerformanceTracker()
        tracker.start("ingest")
        # ... do work ...
        tracker.stop("ingest")

        stats = tracker.summary()
    """

    def __init__(self) -> None:
        self._records: list[LatencyRecord] = []
        self._timers: dict[str, float] = {}

    def start(self, operation: str) -> None:
        """Start timing an operation.

        Args:
            operation: Name of the operation (e.g., 'ingest', 'query').
        """
        self._timers[operation] = time.perf_counter()

    def stop(self, operation: str, metadata: dict[str, Any] | None = None) -> float:
        """Stop timing an operation and record the latency.

        Args:
            operation: Name of the operation to stop.
            metadata: Optional metadata for this record.

        Returns:
            The latency in milliseconds.

        Raises:
            ValueError: If the operation was not started.
        """
        start_time = self._timers.pop(operation, None)
        if start_time is None:
            raise ValueError(f"Operation '{operation}' was not started")

        latency_ms = (time.perf_counter() - start_time) * 1000
        self._records.append(
            LatencyRecord(
                operation=operation,
                latency_ms=latency_ms,
                metadata=metadata or {},
            )
        )
        return latency_ms

    def record(
        self,
        operation: str,
        latency_ms: float,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Directly record a latency measurement.

        Args:
            operation: Name of the operation.
            latency_ms: The latency in milliseconds.
            metadata: Optional metadata for this record.
        """
        self._records.append(
            LatencyRecord(
                operation=operation,
                latency_ms=latency_ms,
                metadata=metadata or {},
            )
        )

    def get_records(self, operation: str | None = None) -> list[LatencyRecord]:
        """Get all records, optionally filtered by operation.

        Args:
            operation: If provided, only return records for this operation.

        Returns:
            A list of latency records.
        """
        if operation is None:
            return list(self._records)
        return [r for r in self._records if r.operation == operation]

    def summary(self) -> dict[str, dict[str, float]]:
        """Compute summary statistics for all operations.

        Returns:
            A dictionary mapping operation names to their statistics
            (count, mean, median, min, max, std, p95, p99).
        """
        operations: dict[str, list[float]] = {}
        for record in self._records:
            operations.setdefault(record.operation, []).append(record.latency_ms)

        result: dict[str, dict[str, float]] = {}
        for op, latencies in operations.items():
            sorted_latencies = sorted(latencies)
            n = len(sorted_latencies)
            result[op] = {
                "count": float(n),
                "mean_ms": statistics.mean(sorted_latencies),
                "median_ms": statistics.median(sorted_latencies),
                "min_ms": sorted_latencies[0],
                "max_ms": sorted_latencies[-1],
                "std_ms": statistics.stdev(sorted_latencies) if n > 1 else 0.0,
                "p95_ms": sorted_latencies[int(n * 0.95)] if n > 1 else sorted_latencies[0],
                "p99_ms": sorted_latencies[int(n * 0.99)] if n > 1 else sorted_latencies[0],
            }

        return result

    def reset(self) -> None:
        """Clear all records and timers."""
        self._records.clear()
        self._timers.clear()


# ---------------------------------------------------------------------------
# Instrumented adapter — transparent timing decorator
# ---------------------------------------------------------------------------


@dataclass
class _IngestRecord:
    """Internal record for a single ingest() call."""

    latency_ms: float
    message_count: int


@dataclass
class _QueryRecord:
    """Internal record for a single query() call."""

    latency_ms: float
    topic: str


@dataclass
class _GetAllFactsRecord:
    """Internal record for a single get_all_facts() call."""

    latency_ms: float
    fact_count: int


class InstrumentedAdapter:
    """Transparent timing decorator for any MemoryAdapter-compatible object.

    ``InstrumentedAdapter`` wraps an existing adapter and intercepts every
    call to :meth:`ingest`, :meth:`query`, and :meth:`get_all_facts`,
    recording wall-clock latency via :func:`time.monotonic`.  The wrapper
    itself satisfies the :class:`~cri.adapter.MemoryAdapter` protocol, so
    it can be used as a drop-in replacement anywhere the scoring engine or
    benchmark runner expects an adapter.

    After each :meth:`ingest` call the wrapper also records a memory growth
    data point by calling ``get_all_facts()`` on the underlying adapter to
    count how many facts are stored.  This is used to build the growth curve
    in the :class:`~cri.models.PerformanceProfile`.

    Parameters
    ----------
    wrapped : object
        Any object that satisfies the ``MemoryAdapter`` protocol (i.e. has
        ``ingest``, ``query``, and ``get_all_facts`` methods with the right
        signatures).

    Attributes
    ----------
    wrapped : object
        The underlying adapter being instrumented.
    ingest_records : list[_IngestRecord]
        Timing records for all ``ingest()`` calls.
    query_records : list[_QueryRecord]
        Timing records for all ``query()`` calls.
    get_all_facts_records : list[_GetAllFactsRecord]
        Timing records for all ``get_all_facts()`` calls.
    total_messages_ingested : int
        Cumulative count of messages passed to ``ingest()``.
    memory_growth_curve : list[tuple[int, int]]
        ``(cumulative_messages, facts_stored)`` data points captured after
        each ``ingest()`` call.
    """

    def __init__(self, wrapped: Any) -> None:
        self.wrapped = wrapped
        self.ingest_records: list[_IngestRecord] = []
        self.query_records: list[_QueryRecord] = []
        self.get_all_facts_records: list[_GetAllFactsRecord] = []
        self.total_messages_ingested: int = 0
        self.memory_growth_curve: list[tuple[int, int]] = []

    # -- MemoryAdapter protocol methods ------------------------------------

    def ingest(self, messages: list[Message]) -> None:
        """Delegate to the wrapped adapter and record timing.

        After the underlying ``ingest()`` completes, a growth-curve data
        point is captured by calling ``get_all_facts()`` on the wrapped
        adapter (this secondary call is **not** included in the ingest
        latency measurement).
        """
        msg_count = len(messages)

        start = time.monotonic()
        self.wrapped.ingest(messages)
        elapsed_ms = (time.monotonic() - start) * 1000.0

        self.ingest_records.append(_IngestRecord(latency_ms=elapsed_ms, message_count=msg_count))
        self.total_messages_ingested += msg_count

        # Capture growth-curve data point
        facts = self.wrapped.get_all_facts()
        self.memory_growth_curve.append((self.total_messages_ingested, len(facts)))

    def query(self, topic: str) -> list[StoredFact]:
        """Delegate to the wrapped adapter and record timing."""
        start = time.monotonic()
        result: list[StoredFact] = self.wrapped.query(topic)
        elapsed_ms = (time.monotonic() - start) * 1000.0

        self.query_records.append(_QueryRecord(latency_ms=elapsed_ms, topic=topic))
        return result

    def get_all_facts(self) -> list[StoredFact]:
        """Delegate to the wrapped adapter and record timing."""
        start = time.monotonic()
        result: list[StoredFact] = self.wrapped.get_all_facts()
        elapsed_ms = (time.monotonic() - start) * 1000.0

        self.get_all_facts_records.append(
            _GetAllFactsRecord(latency_ms=elapsed_ms, fact_count=len(result))
        )
        return result


# ---------------------------------------------------------------------------
# Performance profiler — high-level orchestration
# ---------------------------------------------------------------------------


def _percentile(sorted_values: list[float], pct: float) -> float:
    """Compute a percentile from a pre-sorted list of values.

    Uses the nearest-rank method.  Returns 0.0 for an empty list.
    """
    if not sorted_values:
        return 0.0
    n = len(sorted_values)
    if n == 1:
        return sorted_values[0]
    idx = int(n * pct)
    # Clamp to valid range
    idx = min(idx, n - 1)
    return sorted_values[idx]


class PerformanceProfiler:
    """High-level profiler that instruments an adapter and produces a profile.

    The typical workflow is:

    1. Call :meth:`wrap_adapter` to obtain an :class:`InstrumentedAdapter`.
    2. Use the instrumented adapter throughout the benchmark run.
    3. Call :meth:`get_profile` to collect a :class:`~cri.models.PerformanceProfile`.

    The profiler also exposes ``judge_api_calls`` and
    ``judge_total_cost_estimate`` attributes that external code (e.g. the
    benchmark runner) can set before calling :meth:`get_profile`.

    Example::

        profiler = PerformanceProfiler()
        adapter = profiler.wrap_adapter(raw_adapter)

        adapter.ingest(messages_batch_1)
        adapter.ingest(messages_batch_2)
        adapter.query("occupation")
        adapter.query("hobbies")

        profiler.judge_api_calls = 42
        profile = profiler.get_profile()
    """

    def __init__(self) -> None:
        self._instrumented: InstrumentedAdapter | None = None
        self.judge_api_calls: int = 0
        self.judge_total_cost_estimate: float | None = None

    def wrap_adapter(self, adapter: Any) -> InstrumentedAdapter:
        """Wrap an adapter with instrumentation and return the wrapper.

        Parameters
        ----------
        adapter : object
            Any object satisfying the ``MemoryAdapter`` protocol.

        Returns
        -------
        InstrumentedAdapter
            A transparent wrapper that records timing for every call.

        Raises
        ------
        TypeError
            If *adapter* does not have the required ``ingest``, ``query``,
            and ``get_all_facts`` methods.
        """
        # Validate that the adapter has the required methods
        for method_name in ("ingest", "query", "get_all_facts"):
            if not callable(getattr(adapter, method_name, None)):
                raise TypeError(
                    f"Adapter does not have a callable '{method_name}' method. "
                    f"It must satisfy the MemoryAdapter protocol."
                )

        self._instrumented = InstrumentedAdapter(adapter)
        return self._instrumented

    def get_profile(self) -> PerformanceProfile:
        """Build a :class:`~cri.models.PerformanceProfile` from collected data.

        Returns
        -------
        PerformanceProfile
            Latency statistics, memory growth curve, and judge metadata.

        Raises
        ------
        RuntimeError
            If :meth:`wrap_adapter` has not been called yet.
        """
        if self._instrumented is None:
            raise RuntimeError("No adapter has been wrapped yet. Call wrap_adapter() first.")

        inst = self._instrumented

        # --- Ingest latency (per-message average) -------------------------
        total_ingest_time_ms = sum(r.latency_ms for r in inst.ingest_records)
        total_messages = sum(r.message_count for r in inst.ingest_records)
        ingest_latency_ms = total_ingest_time_ms / total_messages if total_messages > 0 else 0.0

        # --- Query latency stats ------------------------------------------
        query_latencies = sorted(r.latency_ms for r in inst.query_records)
        if query_latencies:
            query_avg = statistics.mean(query_latencies)
            query_p95 = _percentile(query_latencies, 0.95)
            query_p99 = _percentile(query_latencies, 0.99)
        else:
            query_avg = 0.0
            query_p95 = 0.0
            query_p99 = 0.0

        # --- Total facts stored -------------------------------------------
        if inst.memory_growth_curve:
            total_facts = inst.memory_growth_curve[-1][1]
        else:
            # Fall back to calling get_all_facts on the wrapped adapter
            total_facts = len(inst.wrapped.get_all_facts())

        return PerformanceProfile(
            ingest_latency_ms=ingest_latency_ms,
            query_latency_avg_ms=query_avg,
            query_latency_p95_ms=query_p95,
            query_latency_p99_ms=query_p99,
            total_facts_stored=total_facts,
            memory_growth_curve=list(inst.memory_growth_curve),
            judge_api_calls=self.judge_api_calls,
            judge_total_cost_estimate=self.judge_total_cost_estimate,
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "LatencyRecord",
    "PerformanceTracker",
    "InstrumentedAdapter",
    "PerformanceProfiler",
]
