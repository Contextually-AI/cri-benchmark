"""Dataset loader for CRI Benchmark.

API:
    - load_dataset(dataset_dir) -> ConversationDataset
    - load_messages(path) -> list[Message]  (JSONL format)
    - load_ground_truth(path) -> GroundTruth  (JSON format)
    - validate_dataset(dataset) -> list[str]  (returns validation errors)
    - list_canonical_datasets() -> list[DatasetInfo]

Dataset directory structure:
    dataset_dir/
    ├── conversations.jsonl   — One Message JSON per line (required)
    ├── ground_truth.json     — Single GroundTruth JSON object (required)
    └── metadata.json         — DatasetMetadata JSON object (optional)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

from cri.models import (
    ConversationDataset,
    DatasetMetadata,
    GroundTruth,
    Message,
)

logger = logging.getLogger(__name__)

# Default location for canonical datasets (project_root/datasets/canonical)
CANONICAL_DATASETS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "datasets" / "canonical"


# ---------------------------------------------------------------------------
# DatasetInfo model
# ---------------------------------------------------------------------------


class DatasetInfo(BaseModel):
    """Metadata about a discovered canonical dataset.

    Returned by :func:`list_canonical_datasets` to describe each dataset
    found in the canonical directory without fully loading it.
    """

    name: str = Field(description="Dataset directory name")
    path: Path = Field(description="Absolute path to the dataset directory")
    has_ground_truth: bool = Field(description="Whether ground_truth.json exists in the dataset directory")
    message_count: int | None = Field(
        default=None,
        description="Number of messages (lines in conversations.jsonl), or None if file is absent",
    )


# ---------------------------------------------------------------------------
# Module-level loader functions (new API)
# ---------------------------------------------------------------------------


def load_messages(path: Path) -> list[Message]:
    """Load messages from a JSONL file.

    Each non-blank line in the file must be a valid JSON object that can be
    parsed as a :class:`~cri.models.Message`.

    Args:
        path: Path to the JSONL file (e.g. ``conversations.jsonl``).

    Returns:
        Ordered list of Message objects.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If any non-blank line contains invalid JSON or cannot
            be parsed as a Message.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Messages file not found: {path}")

    messages: list[Message] = []
    with open(path, encoding="utf-8") as fh:
        for line_no, raw_line in enumerate(fh, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} of {path}: {exc}") from exc
            try:
                messages.append(Message(**data))
            except (ValidationError, TypeError) as exc:
                raise ValueError(f"Invalid Message on line {line_no} of {path}: {exc}") from exc

    return messages


def load_ground_truth(path: Path) -> GroundTruth:
    """Load a GroundTruth object from a JSON file.

    The file must contain a single JSON object conforming to the
    :class:`~cri.models.GroundTruth` schema.

    Args:
        path: Path to the JSON file (e.g. ``ground_truth.json``).

    Returns:
        A validated GroundTruth instance.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If the file contains invalid JSON or does not conform
            to the GroundTruth schema.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Ground truth file not found: {path}")

    with open(path, encoding="utf-8") as fh:
        try:
            data = json.load(fh)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in ground truth file {path}: {exc}") from exc

    try:
        return GroundTruth(**data)
    except (ValidationError, TypeError) as exc:
        raise ValueError(f"Invalid GroundTruth data in {path}: {exc}") from exc


def load_dataset(dataset_dir: Path) -> ConversationDataset:
    """Load a complete benchmark dataset from a directory.

    The directory must contain at least ``conversations.jsonl`` and
    ``ground_truth.json``. An optional ``metadata.json`` provides dataset
    metadata; if absent, minimal metadata is inferred.

    Args:
        dataset_dir: Path to the dataset directory.

    Returns:
        A fully populated ConversationDataset.

    Raises:
        FileNotFoundError: If *dataset_dir* does not exist or required files
            are missing.
        ValueError: If any file contains invalid data.
    """
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset path is not a directory: {dataset_dir}")

    conversations_path = dataset_dir / "conversations.jsonl"
    ground_truth_path = dataset_dir / "ground_truth.json"
    metadata_path = dataset_dir / "metadata.json"

    if not conversations_path.exists():
        raise FileNotFoundError(f"Required file 'conversations.jsonl' not found in {dataset_dir}")
    if not ground_truth_path.exists():
        raise FileNotFoundError(f"Required file 'ground_truth.json' not found in {dataset_dir}")

    messages = load_messages(conversations_path)
    ground_truth = load_ground_truth(ground_truth_path)

    # Load or infer metadata
    if metadata_path.exists():
        with open(metadata_path, encoding="utf-8") as fh:
            try:
                meta_data = json.load(fh)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in metadata file {metadata_path}: {exc}") from exc
        try:
            metadata = DatasetMetadata(**meta_data)
        except (ValidationError, TypeError) as exc:
            raise ValueError(f"Invalid DatasetMetadata in {metadata_path}: {exc}") from exc
    else:
        # Infer minimal metadata from directory name and message count
        metadata = DatasetMetadata(
            dataset_id=dataset_dir.name,
            persona_id="unknown",
            message_count=len(messages),
            simulated_days=0,
            version="1.0.0",
            seed=None,
        )

    logger.info(
        "Loaded dataset '%s': %d messages, ground_truth loaded",
        dataset_dir.name,
        len(messages),
    )

    return ConversationDataset(
        metadata=metadata,
        messages=messages,
        ground_truth=ground_truth,
    )


def validate_dataset(dataset: ConversationDataset) -> list[str]:
    """Validate a ConversationDataset and return a list of error strings.

    An empty list indicates the dataset is valid. Each string in the returned
    list describes one validation problem.

    Checks performed:
        - Messages list is not empty.
        - Message IDs are unique.
        - Message IDs are sequential (1-indexed, contiguous).
        - Timestamps are non-decreasing.
        - Roles are valid ('user' or 'assistant').
        - Ground truth final_profile is not empty.
        - metadata.message_count matches the actual number of messages.
        - Belief-change references point to valid message IDs.
        - Conflict-scenario references point to valid message IDs.

    Args:
        dataset: The dataset to validate.

    Returns:
        A list of human-readable error strings (empty if valid).
    """
    errors: list[str] = []

    # --- Message-level checks ---
    if not dataset.messages:
        errors.append("Messages list is empty")
        return errors  # no further message checks possible

    # Unique message IDs
    msg_ids = [m.message_id for m in dataset.messages]
    msg_id_set = set(msg_ids)
    if len(msg_ids) != len(msg_id_set):
        duplicates = [mid for mid in msg_id_set if msg_ids.count(mid) > 1]
        errors.append(f"Duplicate message_id(s): {sorted(duplicates)}")

    # Sequential IDs (1-indexed, contiguous)
    expected_ids = list(range(1, len(dataset.messages) + 1))
    if sorted(msg_ids) != expected_ids:
        errors.append(f"Message IDs are not sequential 1..{len(dataset.messages)}; got {sorted(msg_ids)[:10]}{'...' if len(msg_ids) > 10 else ''}")

    # Timestamps non-decreasing
    timestamps = [m.timestamp for m in dataset.messages]
    for i in range(1, len(timestamps)):
        if timestamps[i] < timestamps[i - 1]:
            errors.append(
                f"Timestamps are not non-decreasing: message {dataset.messages[i].message_id} "
                f"('{timestamps[i]}') < message {dataset.messages[i - 1].message_id} "
                f"('{timestamps[i - 1]}')"
            )
            break  # report first violation only

    # Valid roles
    valid_roles = {"user", "assistant"}
    for msg in dataset.messages:
        if msg.role not in valid_roles:
            errors.append(f"Invalid role '{msg.role}' for message {msg.message_id}")

    # --- Ground truth checks ---
    if not dataset.ground_truth.final_profile:
        errors.append("Ground truth final_profile is empty")

    # --- Metadata checks ---
    if dataset.metadata.message_count != len(dataset.messages):
        errors.append(f"metadata.message_count ({dataset.metadata.message_count}) does not match actual message count ({len(dataset.messages)})")

    # --- Cross-reference checks ---
    # Belief changes reference valid message IDs
    for i, change in enumerate(dataset.ground_truth.changes):
        if change.changed_around_msg not in msg_id_set:
            errors.append(f"Belief change [{i}] references invalid message_id changed_around_msg={change.changed_around_msg}")
        for km in change.key_messages:
            if km not in msg_id_set:
                errors.append(f"Belief change [{i}] references invalid key_message id={km}")

    # Conflict scenarios reference valid message IDs
    for conflict in dataset.ground_truth.conflicts:
        for mid in conflict.introduced_at_messages:
            if mid not in msg_id_set:
                errors.append(f"Conflict '{conflict.conflict_id}' references invalid introduced_at_messages id={mid}")

    return errors


def list_canonical_datasets() -> list[DatasetInfo]:
    """Discover all datasets in the canonical datasets directory.

    Scans the canonical directory (``datasets/canonical/`` relative to the
    project root) for subdirectories that contain at least a
    ``conversations.jsonl`` file.

    Returns:
        A sorted list of :class:`DatasetInfo` objects describing each
        discovered dataset. Returns an empty list if the canonical
        directory does not exist.
    """
    if not CANONICAL_DATASETS_DIR.exists():
        return []

    results: list[DatasetInfo] = []
    for entry in sorted(CANONICAL_DATASETS_DIR.iterdir()):
        if not entry.is_dir():
            continue

        conversations_path = entry / "conversations.jsonl"
        ground_truth_path = entry / "ground_truth.json"

        # Only include directories that look like new-format datasets
        # (have conversations.jsonl) OR old-format datasets
        has_conversations = conversations_path.exists()
        has_gt = ground_truth_path.exists()

        # Count messages if conversations file exists
        message_count: int | None = None
        if has_conversations:
            try:
                with open(conversations_path, encoding="utf-8") as fh:
                    message_count = sum(1 for line in fh if line.strip())
            except OSError:
                message_count = None

        # Include if it has conversations.jsonl (new format)
        if has_conversations:
            results.append(
                DatasetInfo(
                    name=entry.name,
                    path=entry,
                    has_ground_truth=has_gt,
                    message_count=message_count,
                )
            )

    return results
