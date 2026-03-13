"""CRI Benchmark dataset loading and generation.

Provides tools for loading canonical datasets, validating dataset
structure, and generating synthetic benchmark datasets.
"""

from cri.datasets.loader import (
    CANONICAL_DATASETS_DIR,
    DatasetInfo,
    list_canonical_datasets,
    load_dataset,
    load_ground_truth,
    load_messages,
    validate_dataset,
)

__all__ = [
    "CANONICAL_DATASETS_DIR",
    "DatasetInfo",
    "list_canonical_datasets",
    "load_dataset",
    "load_ground_truth",
    "load_messages",
    "validate_dataset",
]
