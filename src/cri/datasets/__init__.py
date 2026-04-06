"""CRI Benchmark dataset loading and generation.

Provides tools for loading datasets, validating dataset structure,
generating synthetic benchmark datasets, and loading persona specs.
"""

from cri.datasets.loader import (
    DATASETS_DIR,
    DatasetInfo,
    get_persona,
    list_datasets,
    list_persona_specs,
    load_dataset,
    load_ground_truth,
    load_messages,
    load_persona_spec,
    validate_dataset,
)

__all__ = [
    "DATASETS_DIR",
    "DatasetInfo",
    "get_persona",
    "list_datasets",
    "list_persona_specs",
    "load_dataset",
    "load_ground_truth",
    "load_messages",
    "load_persona_spec",
    "validate_dataset",
]
