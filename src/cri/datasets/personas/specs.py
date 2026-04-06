"""Persona specification model for CRI datasets.

The :class:`PersonaSpec` model defines the schema for persona specifications
used to generate and evaluate benchmark datasets. Persona data is loaded from
JSON files in the dataset directories rather than defined as Python constants.

Use :func:`cri.datasets.loader.get_persona` or
:func:`cri.datasets.loader.list_persona_specs` to load personas.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from cri.models import (
    BeliefChange,
    ConflictScenario,
    NoiseExample,
    ProfileDimension,
    QueryRelevancePair,
    SignalExample,
    TemporalFact,
)


class PersonaSpec(BaseModel):
    """Full specification of a persona used to generate benchmark datasets.

    This model carries all ground-truth components needed to synthesise a
    complete :class:`~cri.models.ConversationDataset` — including belief
    changes, conflict scenarios, temporal facts, and query-relevance pairs.
    """

    persona_id: str = Field(description="Unique identifier for this persona (e.g. 'persona-1-base')")
    name: str = Field(description="Human-readable persona name")
    description: str = Field(description="Brief narrative description of the persona's background")
    complexity_level: str = Field(description="Benchmark complexity tier (e.g. 'base')")

    # Ground-truth components
    profile_dimensions: dict[str, ProfileDimension] = Field(description="Expected final profile keyed by dimension name")
    belief_changes: list[BeliefChange] = Field(
        default_factory=list,
        description="Ordered list of belief changes that occur during the conversation",
    )
    noise_examples: list[NoiseExample] = Field(
        default_factory=list,
        description="Template noise messages to weave into the conversation",
    )
    signal_examples: list[SignalExample] = Field(
        default_factory=list,
        description="Template signal messages that reveal persona facts",
    )
    conflicts: list[ConflictScenario] = Field(
        default_factory=list,
        description="Conflict scenarios to embed in the conversation",
    )
    temporal_facts: list[TemporalFact] = Field(
        default_factory=list,
        description="Facts with temporal validity constraints",
    )
    query_relevance_pairs: list[QueryRelevancePair] = Field(
        default_factory=list,
        description="Query-relevance pairs for QRP evaluation",
    )
    # Generation parameters
    simulated_days: int = Field(
        default=90,
        description="Number of simulated days the conversation should span",
    )
    target_message_count: int = Field(
        default=200,
        description="Approximate number of messages to generate",
    )


__all__ = [
    "PersonaSpec",
]
