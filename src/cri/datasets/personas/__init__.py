"""Pre-defined persona specifications for canonical datasets."""

from cri.datasets.personas.specs import (
    ALL_PERSONAS,
    PERSONA_ADVANCED,
    PERSONA_BASIC,
    PERSONA_INTERMEDIATE,
    RichPersonaSpec,
    get_persona_advanced,
    get_persona_basic,
    get_persona_intermediate,
)

__all__ = [
    "RichPersonaSpec",
    "PERSONA_BASIC",
    "PERSONA_INTERMEDIATE",
    "PERSONA_ADVANCED",
    "ALL_PERSONAS",
    "get_persona_basic",
    "get_persona_intermediate",
    "get_persona_advanced",
]
