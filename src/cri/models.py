"""Core data models for the CRI Benchmark — Contextual Resonance Index.

This module defines all Pydantic v2 models used throughout the CRI benchmark
pipeline. Models are organized into logical groups:

- **Conversation models**: Message and related types for representing
  conversation data that flows through memory systems.
- **Ground truth models**: ProfileDimension, BeliefChange, ConflictScenario,
  TemporalFact, and related types that define expected benchmark outcomes.
- **Dataset models**: DatasetMetadata, ConversationDataset, and GroundTruth
  for packaging benchmark scenarios.
- **Evaluation models**: Verdict, JudgmentResult, DimensionResult, CRIResult
  for capturing benchmark evaluation outcomes.
- **Result models**: PerformanceProfile, BenchmarkResult for full run results.
- **Configuration models**: ScoringConfig, GeneratorConfig for customizing
  benchmark behavior.

"""

from __future__ import annotations

from enum import Enum, StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

# ---------------------------------------------------------------------------
# Evaluation dimension enum (used throughout the codebase)
# ---------------------------------------------------------------------------


class Dimension(StrEnum):
    """Evaluation dimensions of the CRI Benchmark.

    Each dimension measures a distinct property of long-term memory behavior:

    - **PAS** — Persona Accuracy Score: correctness of the stored user profile.
    - **DBU** — Dynamic Belief Updating: ability to update beliefs when new
      information supersedes old information.
    - **TC** — Temporal Coherence: maintaining correct temporal ordering and
      validity of facts over time.
    - **CRQ** — Conflict Resolution Quality: ability to resolve contradictory
      information correctly.
    - **QRP** — Query Response Precision: precision and relevance of responses
      to targeted queries.
    - **MEI** — Memory Efficiency Index: balance between storage coverage
      and storage efficiency.
    - **SFC** — Selective Forgetting Capability: ability to appropriately
      forget ephemeral or superseded information.
    - **LNC** — Long-Horizon Narrative Coherence: maintaining a coherent
      narrative across causally connected events.
    - **ARS** — Adversarial Robustness Score: resistance to malicious
      information injection attempts.
    """

    PAS = "pas"
    DBU = "dbu"
    TC = "tc"
    CRQ = "crq"
    QRP = "qrp"
    MEI = "mei"
    SFC = "sfc"
    LNC = "lnc"
    ARS = "ars"


# ---------------------------------------------------------------------------
# Conversation models
# ---------------------------------------------------------------------------


class Message(BaseModel):
    """A single message in a simulated conversation.

    Messages form the primary input stream that memory systems must process.
    Each message belongs to a conversation turn, has a role (user or assistant),
    and carries temporal metadata such as the simulation day.
    """

    message_id: int = Field(description="Unique sequential identifier for this message")
    role: Literal["user", "assistant"] = Field(description="Who sent the message — 'user' or 'assistant'")
    content: str = Field(description="The textual content of the message")
    timestamp: str = Field(description="ISO-8601 timestamp string indicating when the message occurred")
    session_id: str | None = Field(
        default=None,
        description="Optional session identifier grouping related messages",
    )
    day: int | None = Field(
        default=None,
        description="Simulation day number (1-indexed) when this message occurred",
    )


class StoredFact(BaseModel):
    """A fact stored by the memory system under evaluation.

    Represents a single piece of knowledge that a memory system has extracted
    and stored. Used when comparing what the system stored against ground truth.
    """

    text: str = Field(description="The textual content of the stored fact")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary metadata attached to the fact by the memory system",
    )


# ---------------------------------------------------------------------------
# Ground truth component models
# ---------------------------------------------------------------------------


class ProfileDimension(BaseModel):
    """A single dimension of a user's profile in the ground truth.

    Represents one attribute or characteristic of the persona that the memory
    system should have captured. For example, 'favorite_color' with value 'blue'.
    """

    dimension_name: str = Field(description="Name of the profile dimension (e.g., 'occupation')")
    value: str | list[str] = Field(description="Expected value(s) — a single string or list of valid values")
    query_topic: str = Field(description="Topic string used when querying the memory system about this dimension")
    category: str | None = Field(
        default=None,
        description="Optional category grouping (e.g., 'demographics', 'preferences')",
    )


class BeliefChange(BaseModel):
    """Records a belief update that occurred during the conversation.

    Tracks when a fact changed from one value to another, which messages
    triggered the change, and the topic under which it falls.
    """

    fact: str = Field(description="Description of the fact that changed")
    old_value: str = Field(description="The previous value before the change")
    new_value: str = Field(description="The updated value after the change")
    query_topic: str = Field(description="Topic string for querying this belief")
    changed_around_msg: int = Field(description="Approximate message_id around which the change was introduced")
    key_messages: list[int] = Field(
        default_factory=list,
        description="List of message_ids that are most relevant to this change",
    )


class ConflictScenario(BaseModel):
    """Describes a deliberately introduced conflict in the conversation.

    The benchmark embeds conflicting statements to test whether the memory
    system can identify and resolve contradictions correctly.
    """

    conflict_id: str = Field(description="Unique identifier for this conflict scenario")
    topic: str = Field(description="The topic area where the conflict occurs")
    conflicting_statements: list[str] = Field(description="The contradictory statements introduced in the conversation")
    correct_resolution: str = Field(description="The expected correct resolution of the conflict")
    resolution_type: Literal["recency", "source_authority", "explicit_correction"] = Field(description="Strategy that should be used to resolve the conflict")
    introduced_at_messages: list[int] = Field(description="Message IDs where the conflicting statements appear")


class TemporalFact(BaseModel):
    """A fact with temporal validity constraints.

    Used to evaluate whether the memory system correctly tracks facts that
    have limited time validity — e.g., a job held from 2020 to 2023.
    """

    fact_id: str = Field(description="Unique identifier for this temporal fact")
    description: str = Field(description="Human-readable description of the fact")
    value: str = Field(description="The value of the fact")
    valid_from: str | None = Field(
        default=None,
        description="ISO-8601 timestamp or date string for start of validity",
    )
    valid_until: str | None = Field(
        default=None,
        description="ISO-8601 timestamp or date string for end of validity (None = still valid)",
    )
    query_topic: str = Field(description="Topic string for querying this fact")
    should_be_current: bool = Field(description="Whether this fact should be treated as currently valid")


class QueryRelevancePair(BaseModel):
    """A query paired with expected relevant and irrelevant facts.

    Used to evaluate the precision of the memory system's retrieval —
    whether it returns the right facts and filters out noise.
    """

    query_id: str = Field(description="Unique identifier for this query-relevance pair")
    query: str = Field(description="The query text to pose to the memory system")
    expected_relevant_facts: list[str] = Field(description="Facts that should be retrieved for this query")
    expected_irrelevant_facts: list[str] = Field(description="Facts that should NOT be retrieved for this query")


class ForgettableFact(BaseModel):
    """A fact that the memory system should forget or discard.

    Used by the SFC (Selective Forgetting Capability) dimension to
    evaluate whether a system can appropriately forget ephemeral,
    superseded, or session-contextual information.
    """

    fact_id: str = Field(description="Unique identifier for this forgettable fact")
    text: str = Field(description="The fact text that should be forgotten")
    reason: str = Field(description=("Why this fact should be forgotten — e.g. 'ephemeral_state', 'session_context', 'fully_superseded', 'redundant'"))
    mentioned_at_message: int = Field(description="Message ID where this fact was originally mentioned")
    should_be_absent_after: int = Field(description=("Message ID after which this fact should no longer be stored"))


class NarrativeArc(BaseModel):
    """A narrative arc representing causally connected events over time.

    Used by the LNC (Long-Horizon Narrative Coherence) dimension to evaluate
    whether the memory system maintains a coherent story of the user across
    causally linked events — not just isolated facts.
    """

    arc_id: str = Field(description="Unique identifier for this narrative arc")
    topic: str = Field(description="Human-readable description of the arc topic")
    events_in_order: list[str] = Field(description="Chronologically ordered sequence of events in this arc")
    causal_links: list[str] = Field(description="Causal relationships between events (e.g., 'new job → relocation')")
    query_topic: str = Field(description="Topic string used when querying the memory system about this arc")
    key_messages: list[int] = Field(
        default_factory=list,
        description="Message IDs most relevant to this narrative arc",
    )


class AdversarialMessage(BaseModel):
    """An adversarial attack injected into the conversation.

    Used by the ARS (Adversarial Robustness Score) dimension to evaluate
    whether the memory system resists malicious attempts to corrupt stored
    knowledge — gaslighting, prompt injection, identity confusion, etc.
    """

    attack_id: str = Field(description="Unique identifier for this attack")
    attack_type: str = Field(description=("Category of attack — e.g. 'gaslighting', 'prompt_injection', 'identity_confusion', 'temporal_manipulation'"))
    target_fact: str = Field(description="The fact being attacked (e.g., 'occupation')")
    correct_value: str = Field(description="The correct value that should persist after the attack")
    attack_value: str = Field(description="The malicious value the attack tries to inject")
    query_topic: str = Field(description="Topic string used when querying the memory system after the attack")


class NoiseExample(BaseModel):
    """An example of conversational noise in the dataset.

    Noise messages are those that do NOT contain meaningful factual content
    about the user. The memory system should ideally filter these out.
    """

    text: str = Field(description="The noise message text")
    reason: str = Field(description="Why this message is classified as noise")


class SignalExample(BaseModel):
    """An example of a meaningful signal in the dataset.

    Signal messages contain factual information about the user that should
    be captured by the memory system.
    """

    text: str = Field(description="The signal message text")
    target_fact: str = Field(description="The fact this message conveys or updates")


# ---------------------------------------------------------------------------
# Ground truth aggregate
# ---------------------------------------------------------------------------


class GroundTruth(BaseModel):
    """Complete ground truth for a benchmark dataset.

    Aggregates all expected outcomes: the final profile the system should
    build, the belief changes it should track, conflict scenarios it should
    resolve, temporal facts it should handle, and query-relevance pairs
    for evaluating retrieval precision.
    """

    final_profile: dict[str, ProfileDimension] = Field(description="Expected final profile keyed by dimension name")
    changes: list[BeliefChange] = Field(description="Ordered list of belief changes throughout the conversation")
    noise_examples: list[NoiseExample] = Field(description="Examples of noise messages in the dataset")
    signal_examples: list[SignalExample] = Field(description="Examples of signal messages in the dataset")
    conflicts: list[ConflictScenario] = Field(description="Conflict scenarios embedded in the conversation")
    temporal_facts: list[TemporalFact] = Field(description="Facts with temporal validity constraints")
    query_relevance_pairs: list[QueryRelevancePair] = Field(description="Query-relevance pairs for retrieval precision evaluation")
    forgettable_facts: list[ForgettableFact] = Field(
        default_factory=list,
        description=("Facts that should be forgotten/discarded by the end of the conversation (used by SFC dimension)"),
    )
    narrative_arcs: list[NarrativeArc] = Field(
        default_factory=list,
        description=("Narrative arcs of causally connected events for evaluating long-horizon coherence (used by LNC dimension)"),
    )
    adversarial_messages: list[AdversarialMessage] = Field(
        default_factory=list,
        description=("Adversarial attack messages for evaluating robustness against malicious information injection (used by ARS dimension)"),
    )


# ---------------------------------------------------------------------------
# Dataset models
# ---------------------------------------------------------------------------


class DatasetMetadata(BaseModel):
    """Metadata describing a benchmark conversation dataset.

    Contains provenance information: which persona was used, how many messages
    were generated, the simulation timespan, the version, and the random seed
    for reproducibility.
    """

    dataset_id: str = Field(description="Unique identifier for this dataset")
    persona_id: str = Field(description="Identifier of the persona used to generate the dataset")
    message_count: int = Field(description="Total number of messages in the dataset")
    simulated_days: int = Field(description="Number of simulated days the conversation spans")
    version: str = Field(description="Dataset format version string")
    seed: int | None = Field(
        default=None,
        description="Random seed used for reproducible generation",
    )


class ConversationDataset(BaseModel):
    """A complete benchmark conversation dataset.

    Bundles the metadata, the conversation messages, and the ground truth
    into a single serializable unit. This is the primary input artifact
    for a benchmark run.
    """

    metadata: DatasetMetadata = Field(description="Dataset provenance and configuration metadata")
    messages: list[Message] = Field(description="Ordered list of conversation messages")
    ground_truth: GroundTruth = Field(description="Expected outcomes for evaluation")


# ---------------------------------------------------------------------------
# Evaluation models
# ---------------------------------------------------------------------------


class Verdict(Enum):
    """Binary verdict from an LLM judge evaluation.

    Used by the judge to indicate whether a memory system's response
    meets a specific evaluation check.
    """

    YES = "YES"
    NO = "NO"


class JudgmentResult(BaseModel):
    """Result of a single LLM judge evaluation check.

    Each check poses a yes/no question about the memory system's behavior.
    Multiple votes are collected for robustness and a final verdict is derived.
    """

    check_id: str = Field(description="Unique identifier for the evaluation check")
    verdict: Verdict = Field(description="Final aggregated verdict (YES or NO)")
    votes: list[Verdict] = Field(description="Individual votes from multiple judge calls")
    unanimous: bool = Field(description="Whether all votes agreed")
    prompt: str = Field(description="The prompt sent to the LLM judge")
    raw_responses: list[str] = Field(description="Raw text responses from the judge LLM")


class DimensionResult(BaseModel):
    """Aggregated result for a single CRI evaluation dimension.

    Combines the pass/fail outcomes of multiple checks into a dimension-level
    score. The score is the ratio of passed checks to total checks.
    """

    dimension_name: str = Field(description="Name of the evaluation dimension (e.g., 'PAS')")
    score: float = Field(description="Dimension score as a ratio (0.0 to 1.0)")
    passed_checks: int = Field(description="Number of checks that passed")
    total_checks: int = Field(description="Total number of checks evaluated")
    details: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Per-check detail records for transparency",
    )


class CRIResult(BaseModel):
    """Complete CRI Benchmark result with composite and per-dimension scores.

    The composite CRI score is a weighted average of the individual dimension
    scores. Each dimension score is included both as a top-level convenience
    field and in the detailed breakdown.
    """

    system_name: str = Field(description="Name of the memory system evaluated")
    cri: float = Field(description="Composite CRI score (weighted average)")
    pas: float = Field(description="Persona Accuracy Score")
    dbu: float = Field(description="Dynamic Belief Updating score")
    tc: float = Field(description="Temporal Coherence score")
    crq: float = Field(description="Conflict Resolution Quality score")
    qrp: float = Field(description="Query Response Precision score")
    mei: float = Field(
        default=0.0,
        description="Memory Efficiency Index score",
    )
    sfc: float = Field(
        default=0.0,
        description="Selective Forgetting Capability score",
    )
    lnc: float = Field(
        default=0.0,
        description="Long-Horizon Narrative Coherence score",
    )
    ars: float = Field(
        default=0.0,
        description="Adversarial Robustness Score",
    )
    dimension_weights: dict[str, float] = Field(description="Weights used for each dimension in composite calculation")
    details: dict[str, DimensionResult] = Field(description="Detailed per-dimension results keyed by dimension name")


# ---------------------------------------------------------------------------
# Performance and result models
# ---------------------------------------------------------------------------


class PerformanceProfile(BaseModel):
    """Performance metrics collected during a benchmark run.

    Captures latency, storage, and cost information to complement the
    accuracy-focused CRI scores.
    """

    ingest_latency_ms: float = Field(description="Average latency in ms to ingest a single message")
    query_latency_avg_ms: float = Field(description="Average query response latency in ms")
    query_latency_p95_ms: float = Field(description="95th percentile query response latency in ms")
    query_latency_p99_ms: float = Field(description="99th percentile query response latency in ms")
    total_facts_stored: int = Field(description="Total number of facts stored by the memory system")
    memory_growth_curve: list[tuple[int, int]] = Field(description="List of (message_count, facts_stored) data points")
    judge_api_calls: int = Field(description="Total number of API calls made to the LLM judge")
    judge_total_cost_estimate: float | None = Field(
        default=None,
        description="Estimated total cost of judge API calls in USD",
    )


class BenchmarkResult(BaseModel):
    """Complete result of a benchmark run.

    Bundles the CRI evaluation result, performance profile, and judge log
    into a single artifact that fully describes one evaluation run.
    """

    run_id: str = Field(description="Unique identifier for this benchmark run")
    adapter_name: str = Field(description="Name of the memory system adapter evaluated")
    dataset_id: str = Field(description="Identifier of the dataset used")
    started_at: str = Field(description="ISO-8601 timestamp when the run started")
    completed_at: str = Field(description="ISO-8601 timestamp when the run completed")
    cri_result: CRIResult = Field(description="CRI evaluation scores")
    performance_profile: PerformanceProfile = Field(description="Performance metrics")
    judge_log: list[JudgmentResult] = Field(description="Full log of all judge evaluations")


# ---------------------------------------------------------------------------
# Scoring profiles
# ---------------------------------------------------------------------------


class ScoringProfile(StrEnum):
    """Predefined scoring profiles for the CRI Benchmark.

    Each profile defines which dimensions to evaluate and their weights:

    - **core** — The 6 core dimensions (PAS, DBU, MEI, TC, CRQ, QRP).
    - **extended** — Core + SFC, LNC, ARS (9 dimensions).
    - **full** — Extended + SSI scale sensitivity test (SSI reported separately).
    """

    CORE = "core"
    EXTENDED = "extended"
    FULL = "full"


# Canonical weights for each profile.  Extended and full share the same
# CRI composite weights — the difference is that full also runs SSI.
_PROFILE_WEIGHTS: dict[ScoringProfile, dict[str, float]] = {
    ScoringProfile.CORE: {
        "PAS": 0.25,
        "DBU": 0.20,
        "MEI": 0.20,
        "TC": 0.15,
        "CRQ": 0.10,
        "QRP": 0.10,
    },
    ScoringProfile.EXTENDED: {
        "PAS": 0.20,
        "DBU": 0.20,
        "MEI": 0.15,
        "TC": 0.10,
        "CRQ": 0.10,
        "QRP": 0.10,
        "SFC": 0.05,
        "LNC": 0.05,
        "ARS": 0.05,
    },
    ScoringProfile.FULL: {
        "PAS": 0.20,
        "DBU": 0.20,
        "MEI": 0.15,
        "TC": 0.10,
        "CRQ": 0.10,
        "QRP": 0.10,
        "SFC": 0.05,
        "LNC": 0.05,
        "ARS": 0.05,
    },
}

_PROFILE_DIMENSIONS: dict[ScoringProfile, list[str]] = {
    ScoringProfile.CORE: ["PAS", "DBU", "MEI", "TC", "CRQ", "QRP"],
    ScoringProfile.EXTENDED: ["PAS", "DBU", "MEI", "TC", "CRQ", "QRP", "SFC", "LNC", "ARS"],
    ScoringProfile.FULL: ["PAS", "DBU", "MEI", "TC", "CRQ", "QRP", "SFC", "LNC", "ARS"],
}

# Whether a profile implicitly enables the SSI scale-sensitivity test.
_PROFILE_SCALE_TEST: dict[ScoringProfile, bool] = {
    ScoringProfile.CORE: False,
    ScoringProfile.EXTENDED: False,
    ScoringProfile.FULL: True,
}

# Reference weights used when building custom dimension selections.
# We use the extended profile weights as the canonical weight source.
_REFERENCE_WEIGHTS: dict[str, float] = dict(_PROFILE_WEIGHTS[ScoringProfile.EXTENDED])


# ---------------------------------------------------------------------------
# Configuration models
# ---------------------------------------------------------------------------


class ScoringConfig(BaseModel):
    """Configuration for the CRI scoring engine.

    Controls which dimensions are evaluated and their weights in the
    composite CRI score. Weights must sum to 1.0 for a valid composite.

    Construct via classmethods for the easiest usage::

        config = ScoringConfig.from_profile(ScoringProfile.EXTENDED)
        config = ScoringConfig.from_dimensions(["PAS", "DBU", "MEI"])
    """

    dimension_weights: dict[str, float] = Field(
        default_factory=lambda: dict(_PROFILE_WEIGHTS[ScoringProfile.CORE]),
        description="Weight for each dimension in the composite CRI score",
    )
    enabled_dimensions: list[str] = Field(
        default_factory=lambda: list(_PROFILE_DIMENSIONS[ScoringProfile.CORE]),
        description="List of dimension names to evaluate",
    )
    profile: ScoringProfile = Field(
        default=ScoringProfile.CORE,
        description="The scoring profile used to build this configuration",
    )
    scale_test: bool = Field(
        default=False,
        description="Whether the SSI scale-sensitivity test should be run",
    )

    @classmethod
    def from_profile(cls, profile: ScoringProfile) -> ScoringConfig:
        """Create a config from a predefined scoring profile.

        Args:
            profile: The scoring profile to use.

        Returns:
            A fully configured :class:`ScoringConfig`.
        """
        return cls(
            dimension_weights=dict(_PROFILE_WEIGHTS[profile]),
            enabled_dimensions=list(_PROFILE_DIMENSIONS[profile]),
            profile=profile,
            scale_test=_PROFILE_SCALE_TEST[profile],
        )

    @classmethod
    def from_dimensions(cls, dimensions: list[str]) -> ScoringConfig:
        """Create a config from an explicit list of dimension names.

        Weights are derived from the canonical reference weights and
        re-normalized to sum to 1.0.

        Args:
            dimensions: List of dimension codes (e.g. ``["PAS", "MEI"]``).

        Returns:
            A fully configured :class:`ScoringConfig`.

        Raises:
            ValueError: If any dimension name is not recognized or no
                valid dimensions are provided.
        """
        valid_names = {d.value.upper() for d in Dimension}
        unknown = [d for d in dimensions if d.upper() not in valid_names]
        if unknown:
            raise ValueError(f"Unknown dimension(s): {', '.join(unknown)}")

        normalized = [d.upper() for d in dimensions]
        if not normalized:
            raise ValueError("At least one dimension must be specified.")

        # Gather raw weights (use reference weight, or equal share if missing).
        raw: dict[str, float] = {}
        for dim in normalized:
            raw[dim] = _REFERENCE_WEIGHTS.get(dim, 1.0 / len(normalized))

        total = sum(raw.values())
        weights = {d: w / total for d, w in raw.items()}

        return cls(
            dimension_weights=weights,
            enabled_dimensions=normalized,
            profile=ScoringProfile.CORE,  # custom selection, no named profile
            scale_test=False,
        )

    @model_validator(mode="after")
    def _ensure_weights_cover_enabled(self) -> ScoringConfig:
        """Ensure every enabled dimension has a weight entry."""
        for dim in self.enabled_dimensions:
            if dim not in self.dimension_weights:
                self.dimension_weights[dim] = 0.0
        return self


class GeneratorConfig(BaseModel):
    """Configuration for the benchmark dataset generator.

    Controls the LLM model used for synthetic conversation generation,
    the simulation timespan, and message density.
    """

    llm_model: str = Field(
        default="claude-3-5-sonnet-20241022",
        description="LLM model identifier for conversation generation",
    )
    seed: int | None = Field(
        default=None,
        description="Random seed for reproducible dataset generation",
    )
    simulated_days: int = Field(
        default=90,
        description="Number of simulated days for the conversation",
    )
    messages_per_day_range: tuple[int, int] = Field(
        default=(5, 15),
        description="Min and max messages per simulated day",
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # Enums
    "Dimension",
    "ScoringProfile",
    "Verdict",
    # Conversation models
    "Message",
    "StoredFact",
    # Ground truth components
    "ProfileDimension",
    "BeliefChange",
    "ConflictScenario",
    "TemporalFact",
    "QueryRelevancePair",
    "NoiseExample",
    "SignalExample",
    "ForgettableFact",
    "NarrativeArc",
    "AdversarialMessage",
    # Ground truth aggregate
    "GroundTruth",
    # Dataset models
    "DatasetMetadata",
    "ConversationDataset",
    # Evaluation models
    "JudgmentResult",
    "DimensionResult",
    "CRIResult",
    # Result models
    "PerformanceProfile",
    "BenchmarkResult",
    # Configuration models
    "ScoringConfig",
    "GeneratorConfig",
]
