"""Tests for the CRI Benchmark dataset loader — both legacy and new API.

Covers:
- New module-level functions (load_messages, load_ground_truth, load_dataset,
  validate_dataset, list_canonical_datasets)
- JSONL loading edge cases
- JSON ground truth loading and schema validation
- Dataset validation catches all known error types
- list_canonical_datasets discovery behavior
- Integration: round-trip load → validate
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cri.datasets.loader import (
    DatasetInfo,
    list_canonical_datasets,
    load_dataset,
    load_ground_truth,
    load_messages,
    validate_dataset,
)
from cri.models import (
    ConversationDataset,
    DatasetMetadata,
    GroundTruth,
    Message,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _minimal_ground_truth_dict(
    *,
    final_profile: dict | None = None,
) -> dict:
    """Return a minimal valid GroundTruth as a plain dict."""
    if final_profile is None:
        final_profile = {
            "occupation": {
                "dimension_name": "occupation",
                "value": "Engineer",
                "query_topic": "job",
            },
        }
    return {
        "final_profile": final_profile,
        "changes": [],
        "noise_examples": [],
        "signal_examples": [],
        "conflicts": [],
        "temporal_facts": [],
        "query_relevance_pairs": [],
    }


def _minimal_messages_jsonl(n: int = 3) -> str:
    """Return JSONL string with *n* valid messages."""
    lines = []
    for i in range(1, n + 1):
        role = "user" if i % 2 == 1 else "assistant"
        msg = {
            "message_id": i,
            "role": role,
            "content": f"Message {i}",
            "timestamp": f"2026-01-01T{10 + i:02d}:00:00Z",
        }
        lines.append(json.dumps(msg))
    return "\n".join(lines) + "\n"


def _write_dataset(
    base: Path,
    name: str = "test-ds",
    *,
    messages_jsonl: str | None = None,
    ground_truth: dict | None = None,
    metadata: dict | None = None,
    skip_conversations: bool = False,
    skip_ground_truth: bool = False,
) -> Path:
    """Create a dataset directory with standard files. Returns its path."""
    ds_dir = base / name
    ds_dir.mkdir(parents=True, exist_ok=True)

    if not skip_conversations:
        (ds_dir / "conversations.jsonl").write_text(
            messages_jsonl or _minimal_messages_jsonl(),
            encoding="utf-8",
        )

    if not skip_ground_truth:
        gt = ground_truth or _minimal_ground_truth_dict()
        (ds_dir / "ground_truth.json").write_text(
            json.dumps(gt, indent=2),
            encoding="utf-8",
        )

    if metadata is not None:
        (ds_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )

    return ds_dir


def _write_legacy_dataset(
    base: Path,
    name: str = "test-legacy",
    *,
    events: list[dict] | None = None,
    queries: list[dict] | None = None,
    persona: dict | None = None,
) -> Path:
    """Create a legacy-format dataset directory with events.json, queries.json, persona.json."""
    ds_dir = base / name
    ds_dir.mkdir(parents=True, exist_ok=True)

    if events is None:
        events = [
            {
                "id": "evt-001",
                "content": "Alice is a software engineer.",
                "timestamp": "2026-01-01T10:00:00",
                "sequence_number": 1,
            },
        ]
    if queries is None:
        queries = [
            {
                "id": "q-pas-001",
                "question": "What does Alice do?",
                "expected_answer": "Software engineer",
                "dimension": "pas",
            },
        ]
    if persona is None:
        persona = {
            "id": "persona-test",
            "name": "Alice",
            "description": "A test persona",
            "complexity_level": "basic",
            "traits": {"age": 30},
        }

    (ds_dir / "events.json").write_text(json.dumps(events), encoding="utf-8")
    (ds_dir / "queries.json").write_text(json.dumps(queries), encoding="utf-8")
    (ds_dir / "persona.json").write_text(json.dumps(persona), encoding="utf-8")

    return ds_dir


# ===================================================================
# New API — load_messages (JSONL loading)
# ===================================================================


class TestLoadMessages:
    """Test JSONL loading with load_messages."""

    def test_load_valid_jsonl(self, tmp_path: Path) -> None:
        path = tmp_path / "conversations.jsonl"
        path.write_text(_minimal_messages_jsonl(3), encoding="utf-8")

        messages = load_messages(path)
        assert len(messages) == 3
        assert all(isinstance(m, Message) for m in messages)
        assert messages[0].message_id == 1
        assert messages[0].role == "user"
        assert messages[2].message_id == 3

    def test_blank_lines_skipped(self, tmp_path: Path) -> None:
        path = tmp_path / "conversations.jsonl"
        content = (
            '{"message_id": 1, "role": "user",'
            ' "content": "Hi",'
            ' "timestamp": "2026-01-01T10:00:00Z"}\n'
            "\n"
            "   \n"
            '{"message_id": 2, "role": "assistant",'
            ' "content": "Hello",'
            ' "timestamp": "2026-01-01T10:00:05Z"}\n'
        )
        path.write_text(content, encoding="utf-8")

        messages = load_messages(path)
        assert len(messages) == 2

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_messages(tmp_path / "nonexistent.jsonl")

    def test_invalid_json_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.jsonl"
        path.write_text("this is not json\n", encoding="utf-8")

        with pytest.raises(ValueError, match="Invalid JSON"):
            load_messages(path)

    def test_invalid_message_schema_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "bad_schema.jsonl"
        path.write_text('{"foo": "bar"}\n', encoding="utf-8")

        with pytest.raises(ValueError, match="Invalid Message"):
            load_messages(path)

    def test_empty_file_returns_empty_list(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.jsonl"
        path.write_text("", encoding="utf-8")

        messages = load_messages(path)
        assert messages == []

    def test_preserves_optional_fields(self, tmp_path: Path) -> None:
        path = tmp_path / "conversations.jsonl"
        msg = {
            "message_id": 1,
            "role": "user",
            "content": "Hello",
            "timestamp": "2026-01-01T10:00:00Z",
            "session_id": "sess-abc",
            "day": 5,
        }
        path.write_text(json.dumps(msg) + "\n", encoding="utf-8")

        messages = load_messages(path)
        assert messages[0].session_id == "sess-abc"
        assert messages[0].day == 5

    def test_optional_fields_default_none(self, tmp_path: Path) -> None:
        """Messages without optional fields should have None defaults."""
        path = tmp_path / "conversations.jsonl"
        msg = {
            "message_id": 1,
            "role": "user",
            "content": "Hi",
            "timestamp": "2026-01-01T10:00:00Z",
        }
        path.write_text(json.dumps(msg) + "\n", encoding="utf-8")

        messages = load_messages(path)
        assert messages[0].session_id is None
        assert messages[0].day is None

    def test_large_file(self, tmp_path: Path) -> None:
        """Load a JSONL file with 100 messages."""
        path = tmp_path / "large.jsonl"
        path.write_text(_minimal_messages_jsonl(100), encoding="utf-8")

        messages = load_messages(path)
        assert len(messages) == 100
        assert messages[0].message_id == 1
        assert messages[99].message_id == 100

    def test_invalid_json_midfile_reports_line(self, tmp_path: Path) -> None:
        """Invalid JSON on line 3 should report the correct line number."""
        path = tmp_path / "mid_error.jsonl"
        content = (
            '{"message_id": 1, "role": "user",'
            ' "content": "A",'
            ' "timestamp": "2026-01-01T10:00:00Z"}\n'
            '{"message_id": 2, "role": "assistant",'
            ' "content": "B",'
            ' "timestamp": "2026-01-01T10:01:00Z"}\n'
            "NOT VALID JSON\n"
        )
        path.write_text(content, encoding="utf-8")

        with pytest.raises(ValueError, match="line 3"):
            load_messages(path)

    def test_unicode_content(self, tmp_path: Path) -> None:
        """Messages with unicode content should load correctly."""
        path = tmp_path / "unicode.jsonl"
        msg = {
            "message_id": 1,
            "role": "user",
            "content": "Héllo, I like café ☕ and émojis 🎉",
            "timestamp": "2026-01-01T10:00:00Z",
        }
        path.write_text(json.dumps(msg, ensure_ascii=False) + "\n", encoding="utf-8")

        messages = load_messages(path)
        assert "café" in messages[0].content
        assert "🎉" in messages[0].content

    def test_accepts_path_as_string(self, tmp_path: Path) -> None:
        """Function should accept string paths too."""
        path = tmp_path / "conversations.jsonl"
        path.write_text(_minimal_messages_jsonl(1), encoding="utf-8")

        messages = load_messages(str(path))
        assert len(messages) == 1

    def test_whitespace_only_file(self, tmp_path: Path) -> None:
        """File with only whitespace should return empty list."""
        path = tmp_path / "spaces.jsonl"
        path.write_text("  \n  \n\t\n", encoding="utf-8")

        messages = load_messages(path)
        assert messages == []


# ===================================================================
# New API — load_ground_truth (JSON loading)
# ===================================================================


class TestLoadGroundTruth:
    """Test JSON ground truth loading."""

    def test_load_valid_ground_truth(self, tmp_path: Path) -> None:
        path = tmp_path / "ground_truth.json"
        gt_dict = _minimal_ground_truth_dict()
        path.write_text(json.dumps(gt_dict), encoding="utf-8")

        gt = load_ground_truth(path)
        assert isinstance(gt, GroundTruth)
        assert "occupation" in gt.final_profile
        assert gt.final_profile["occupation"].value == "Engineer"

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_ground_truth(tmp_path / "missing.json")

    def test_invalid_json_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        path.write_text("{not valid json}", encoding="utf-8")

        with pytest.raises(ValueError, match="Invalid JSON"):
            load_ground_truth(path)

    def test_invalid_schema_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "bad_schema.json"
        path.write_text('{"wrong": "schema"}', encoding="utf-8")

        with pytest.raises(ValueError, match="Invalid GroundTruth"):
            load_ground_truth(path)

    def test_ground_truth_with_all_sections(self, tmp_path: Path) -> None:
        path = tmp_path / "ground_truth.json"
        gt_dict = {
            "final_profile": {
                "age": {
                    "dimension_name": "age",
                    "value": "30",
                    "query_topic": "age",
                },
            },
            "changes": [
                {
                    "fact": "phone",
                    "old_value": "iPhone",
                    "new_value": "Pixel",
                    "query_topic": "phone",
                    "changed_around_msg": 3,
                    "key_messages": [3],
                },
            ],
            "noise_examples": [
                {"text": "Nice weather today", "reason": "Small talk"},
            ],
            "signal_examples": [
                {"text": "I work as an engineer", "target_fact": "occupation"},
            ],
            "conflicts": [],
            "temporal_facts": [],
            "query_relevance_pairs": [],
        }
        path.write_text(json.dumps(gt_dict), encoding="utf-8")

        gt = load_ground_truth(path)
        assert len(gt.changes) == 1
        assert gt.changes[0].new_value == "Pixel"
        assert len(gt.noise_examples) == 1
        assert len(gt.signal_examples) == 1

    def test_ground_truth_with_conflicts(self, tmp_path: Path) -> None:
        path = tmp_path / "gt.json"
        gt_dict = _minimal_ground_truth_dict()
        gt_dict["conflicts"] = [
            {
                "conflict_id": "c1",
                "topic": "hobby",
                "conflicting_statements": ["I like hiking", "I like biking"],
                "correct_resolution": "biking (most recent)",
                "resolution_type": "recency",
                "introduced_at_messages": [1, 5],
            },
        ]
        path.write_text(json.dumps(gt_dict), encoding="utf-8")

        gt = load_ground_truth(path)
        assert len(gt.conflicts) == 1
        assert gt.conflicts[0].resolution_type == "recency"

    def test_ground_truth_with_temporal_facts(self, tmp_path: Path) -> None:
        path = tmp_path / "gt.json"
        gt_dict = _minimal_ground_truth_dict()
        gt_dict["temporal_facts"] = [
            {
                "fact_id": "tf-1",
                "description": "Job at Company A",
                "value": "Engineer at A",
                "valid_from": "2025-01-01",
                "valid_until": "2026-01-01",
                "query_topic": "previous job",
                "should_be_current": False,
            },
        ]
        path.write_text(json.dumps(gt_dict), encoding="utf-8")

        gt = load_ground_truth(path)
        assert len(gt.temporal_facts) == 1
        assert gt.temporal_facts[0].should_be_current is False

    def test_ground_truth_with_query_relevance_pairs(self, tmp_path: Path) -> None:
        path = tmp_path / "gt.json"
        gt_dict = _minimal_ground_truth_dict()
        gt_dict["query_relevance_pairs"] = [
            {
                "query_id": "qrp-1",
                "query": "What is the user's job?",
                "expected_relevant_facts": ["Engineer"],
                "expected_irrelevant_facts": ["likes dogs"],
            },
        ]
        path.write_text(json.dumps(gt_dict), encoding="utf-8")

        gt = load_ground_truth(path)
        assert len(gt.query_relevance_pairs) == 1
        assert gt.query_relevance_pairs[0].expected_relevant_facts == ["Engineer"]

    def test_profile_dimension_with_list_value(self, tmp_path: Path) -> None:
        """ProfileDimension.value can be a list of strings."""
        path = tmp_path / "gt.json"
        gt_dict = _minimal_ground_truth_dict(
            final_profile={
                "hobbies": {
                    "dimension_name": "hobbies",
                    "value": ["hiking", "photography"],
                    "query_topic": "hobbies",
                    "category": "interests",
                },
            },
        )
        path.write_text(json.dumps(gt_dict), encoding="utf-8")

        gt = load_ground_truth(path)
        assert gt.final_profile["hobbies"].value == ["hiking", "photography"]

    def test_empty_json_object_raises(self, tmp_path: Path) -> None:
        """An empty JSON object {} should fail GroundTruth schema validation."""
        path = tmp_path / "empty.json"
        path.write_text("{}", encoding="utf-8")

        with pytest.raises(ValueError, match="Invalid GroundTruth"):
            load_ground_truth(path)

    def test_accepts_path_as_string(self, tmp_path: Path) -> None:
        path = tmp_path / "gt.json"
        path.write_text(json.dumps(_minimal_ground_truth_dict()), encoding="utf-8")

        gt = load_ground_truth(str(path))
        assert isinstance(gt, GroundTruth)


# ===================================================================
# New API — load_dataset
# ===================================================================


class TestLoadDataset:
    def test_load_valid_dataset(self, tmp_path: Path) -> None:
        ds_dir = _write_dataset(tmp_path)

        dataset = load_dataset(ds_dir)
        assert isinstance(dataset, ConversationDataset)
        assert len(dataset.messages) == 3
        assert isinstance(dataset.ground_truth, GroundTruth)
        assert isinstance(dataset.metadata, DatasetMetadata)

    def test_inferred_metadata(self, tmp_path: Path) -> None:
        ds_dir = _write_dataset(tmp_path, name="my-dataset")

        dataset = load_dataset(ds_dir)
        assert dataset.metadata.dataset_id == "my-dataset"
        assert dataset.metadata.persona_id == "unknown"
        assert dataset.metadata.message_count == 3
        assert dataset.metadata.version == "1.0.0"

    def test_explicit_metadata(self, tmp_path: Path) -> None:
        meta = {
            "dataset_id": "custom-id",
            "persona_id": "persona-42",
            "message_count": 3,
            "simulated_days": 30,
            "version": "2.0.0",
            "seed": 12345,
        }
        ds_dir = _write_dataset(tmp_path, metadata=meta)

        dataset = load_dataset(ds_dir)
        assert dataset.metadata.dataset_id == "custom-id"
        assert dataset.metadata.persona_id == "persona-42"
        assert dataset.metadata.seed == 12345

    def test_missing_directory_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="not found"):
            load_dataset(tmp_path / "nonexistent")

    def test_missing_conversations_raises(self, tmp_path: Path) -> None:
        ds_dir = _write_dataset(tmp_path, skip_conversations=True)
        with pytest.raises(FileNotFoundError, match="conversations.jsonl"):
            load_dataset(ds_dir)

    def test_missing_ground_truth_raises(self, tmp_path: Path) -> None:
        ds_dir = _write_dataset(tmp_path, skip_ground_truth=True)
        with pytest.raises(FileNotFoundError, match="ground_truth.json"):
            load_dataset(ds_dir)

    def test_not_a_directory_raises(self, tmp_path: Path) -> None:
        filepath = tmp_path / "afile.txt"
        filepath.write_text("hi", encoding="utf-8")
        with pytest.raises(FileNotFoundError, match="not a directory"):
            load_dataset(filepath)

    def test_invalid_metadata_json_raises(self, tmp_path: Path) -> None:
        """Invalid JSON in metadata.json should raise ValueError."""
        ds_dir = _write_dataset(tmp_path, name="bad-meta")
        (ds_dir / "metadata.json").write_text("not json", encoding="utf-8")

        with pytest.raises(ValueError, match="Invalid JSON"):
            load_dataset(ds_dir)

    def test_invalid_metadata_schema_raises(self, tmp_path: Path) -> None:
        """Valid JSON but invalid DatasetMetadata schema should raise ValueError."""
        ds_dir = _write_dataset(tmp_path, name="bad-meta-schema")
        (ds_dir / "metadata.json").write_text('{"bad": "schema"}', encoding="utf-8")

        with pytest.raises(ValueError, match="Invalid DatasetMetadata"):
            load_dataset(ds_dir)

    def test_accepts_path_as_string(self, tmp_path: Path) -> None:
        ds_dir = _write_dataset(tmp_path)
        dataset = load_dataset(str(ds_dir))
        assert isinstance(dataset, ConversationDataset)

    def test_dataset_messages_order_preserved(self, tmp_path: Path) -> None:
        """Messages should maintain their JSONL file order."""
        ds_dir = _write_dataset(tmp_path, messages_jsonl=_minimal_messages_jsonl(10))
        dataset = load_dataset(ds_dir)
        ids = [m.message_id for m in dataset.messages]
        assert ids == list(range(1, 11))


# ===================================================================
# New API — validate_dataset
# ===================================================================


class TestValidateDataset:
    """Test that validation catches all known error types."""

    def _make_dataset(
        self,
        *,
        messages: list[dict] | None = None,
        metadata_overrides: dict | None = None,
        ground_truth_overrides: dict | None = None,
    ) -> ConversationDataset:
        if messages is None:
            messages = [
                {
                    "message_id": 1,
                    "role": "user",
                    "content": "Hi",
                    "timestamp": "2026-01-01T10:00:00Z",
                },
                {
                    "message_id": 2,
                    "role": "assistant",
                    "content": "Hello",
                    "timestamp": "2026-01-01T10:00:05Z",
                },
                {
                    "message_id": 3,
                    "role": "user",
                    "content": "Bye",
                    "timestamp": "2026-01-01T10:01:00Z",
                },
            ]

        gt_dict = _minimal_ground_truth_dict()
        if ground_truth_overrides:
            gt_dict.update(ground_truth_overrides)

        meta = {
            "dataset_id": "test",
            "persona_id": "p1",
            "message_count": len(messages),
            "simulated_days": 1,
            "version": "1.0.0",
        }
        if metadata_overrides:
            meta.update(metadata_overrides)

        return ConversationDataset(
            metadata=DatasetMetadata(**meta),
            messages=[Message(**m) for m in messages],
            ground_truth=GroundTruth(**gt_dict),
        )

    def test_valid_dataset_no_errors(self) -> None:
        ds = self._make_dataset()
        errors = validate_dataset(ds)
        assert errors == []

    def test_empty_messages(self) -> None:
        ds = self._make_dataset(
            messages=[],
            metadata_overrides={"message_count": 0},
        )
        errors = validate_dataset(ds)
        assert any("empty" in e.lower() for e in errors)

    def test_duplicate_message_ids(self) -> None:
        msgs = [
            {
                "message_id": 1,
                "role": "user",
                "content": "A",
                "timestamp": "2026-01-01T10:00:00Z",
            },
            {
                "message_id": 1,
                "role": "assistant",
                "content": "B",
                "timestamp": "2026-01-01T10:00:05Z",
            },
        ]
        ds = self._make_dataset(messages=msgs, metadata_overrides={"message_count": 2})
        errors = validate_dataset(ds)
        assert any("duplicate" in e.lower() for e in errors)

    def test_non_sequential_ids(self) -> None:
        msgs = [
            {
                "message_id": 1,
                "role": "user",
                "content": "A",
                "timestamp": "2026-01-01T10:00:00Z",
            },
            {
                "message_id": 3,
                "role": "assistant",
                "content": "B",
                "timestamp": "2026-01-01T10:00:05Z",
            },
        ]
        ds = self._make_dataset(messages=msgs, metadata_overrides={"message_count": 2})
        errors = validate_dataset(ds)
        assert any("sequential" in e.lower() for e in errors)

    def test_mismatched_message_count(self) -> None:
        ds = self._make_dataset(metadata_overrides={"message_count": 999})
        errors = validate_dataset(ds)
        assert any("message_count" in e for e in errors)

    def test_empty_final_profile(self) -> None:
        ds = self._make_dataset(ground_truth_overrides={"final_profile": {}})
        errors = validate_dataset(ds)
        assert any("final_profile" in e for e in errors)

    def test_timestamps_non_decreasing(self) -> None:
        msgs = [
            {
                "message_id": 1,
                "role": "user",
                "content": "A",
                "timestamp": "2026-01-01T12:00:00Z",
            },
            {
                "message_id": 2,
                "role": "assistant",
                "content": "B",
                "timestamp": "2026-01-01T10:00:00Z",
            },
            {
                "message_id": 3,
                "role": "user",
                "content": "C",
                "timestamp": "2026-01-01T11:00:00Z",
            },
        ]
        ds = self._make_dataset(messages=msgs)
        errors = validate_dataset(ds)
        assert any("timestamp" in e.lower() or "non-decreasing" in e.lower() for e in errors)

    def test_invalid_belief_change_reference(self) -> None:
        gt = _minimal_ground_truth_dict()
        gt["changes"] = [
            {
                "fact": "phone",
                "old_value": "iPhone",
                "new_value": "Pixel",
                "query_topic": "phone",
                "changed_around_msg": 999,
                "key_messages": [1, 888],
            },
        ]
        ds = self._make_dataset(ground_truth_overrides=gt)
        errors = validate_dataset(ds)
        assert any("999" in e for e in errors)
        assert any("888" in e for e in errors)

    def test_invalid_conflict_reference(self) -> None:
        gt = _minimal_ground_truth_dict()
        gt["conflicts"] = [
            {
                "conflict_id": "c1",
                "topic": "food",
                "conflicting_statements": ["I love pizza", "I hate pizza"],
                "correct_resolution": "I love pizza",
                "resolution_type": "recency",
                "introduced_at_messages": [1, 777],
            },
        ]
        ds = self._make_dataset(ground_truth_overrides=gt)
        errors = validate_dataset(ds)
        assert any("777" in e for e in errors)

    def test_valid_dataset_with_belief_changes_and_conflicts(self) -> None:
        """No errors when references point to valid message IDs."""
        gt = _minimal_ground_truth_dict()
        gt["changes"] = [
            {
                "fact": "phone",
                "old_value": "iPhone",
                "new_value": "Pixel",
                "query_topic": "phone",
                "changed_around_msg": 2,
                "key_messages": [1, 3],
            },
        ]
        gt["conflicts"] = [
            {
                "conflict_id": "c1",
                "topic": "food",
                "conflicting_statements": ["likes pizza", "hates pizza"],
                "correct_resolution": "likes pizza",
                "resolution_type": "explicit_correction",
                "introduced_at_messages": [1, 2],
            },
        ]
        ds = self._make_dataset(ground_truth_overrides=gt)
        errors = validate_dataset(ds)
        assert errors == []

    def test_multiple_errors_detected(self) -> None:
        """A badly formed dataset should report multiple errors."""
        msgs = [
            {
                "message_id": 1,
                "role": "user",
                "content": "A",
                "timestamp": "2026-01-01T12:00:00Z",
            },
            {
                "message_id": 1,
                "role": "assistant",
                "content": "B",
                "timestamp": "2026-01-01T10:00:00Z",
            },
        ]
        gt = _minimal_ground_truth_dict(final_profile={})
        gt["changes"] = [
            {
                "fact": "x",
                "old_value": "a",
                "new_value": "b",
                "query_topic": "x",
                "changed_around_msg": 999,
                "key_messages": [],
            },
        ]
        ds = self._make_dataset(
            messages=msgs,
            metadata_overrides={"message_count": 5},
            ground_truth_overrides=gt,
        )
        errors = validate_dataset(ds)
        # Should catch: duplicate IDs, non-sequential IDs, timestamp issue,
        # empty final_profile, mismatched message_count, invalid change reference
        assert len(errors) >= 4

    def test_single_message_valid(self) -> None:
        """A dataset with a single message should validate."""
        msgs = [
            {
                "message_id": 1,
                "role": "user",
                "content": "Hi",
                "timestamp": "2026-01-01T10:00:00Z",
            },
        ]
        ds = self._make_dataset(messages=msgs, metadata_overrides={"message_count": 1})
        errors = validate_dataset(ds)
        assert errors == []

    def test_equal_timestamps_valid(self) -> None:
        """Equal timestamps (same time) should be valid (non-decreasing allows equal)."""
        msgs = [
            {
                "message_id": 1,
                "role": "user",
                "content": "A",
                "timestamp": "2026-01-01T10:00:00Z",
            },
            {
                "message_id": 2,
                "role": "assistant",
                "content": "B",
                "timestamp": "2026-01-01T10:00:00Z",
            },
            {
                "message_id": 3,
                "role": "user",
                "content": "C",
                "timestamp": "2026-01-01T10:00:00Z",
            },
        ]
        ds = self._make_dataset(messages=msgs)
        errors = validate_dataset(ds)
        assert errors == []


# ===================================================================
# New API — list_canonical_datasets
# ===================================================================


class TestListCanonicalDatasets:
    """Test list_canonical_datasets discovery behavior."""

    def test_returns_list_of_dataset_info(self) -> None:
        results = list_canonical_datasets()
        assert isinstance(results, list)
        for info in results:
            assert isinstance(info, DatasetInfo)
            assert isinstance(info.name, str)
            assert isinstance(info.path, Path)
            assert isinstance(info.has_ground_truth, bool)

    def test_returns_empty_for_nonexistent_dir(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        import cri.datasets.loader as loader_module

        monkeypatch.setattr(loader_module, "CANONICAL_DATASETS_DIR", tmp_path / "nope")

        results = list_canonical_datasets()
        assert results == []

    def test_discovers_new_format_datasets(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        import cri.datasets.loader as loader_module

        monkeypatch.setattr(loader_module, "CANONICAL_DATASETS_DIR", tmp_path)

        _write_dataset(tmp_path, name="alpha-dataset")
        _write_dataset(tmp_path, name="beta-dataset")

        results = list_canonical_datasets()
        assert len(results) == 2
        names = [r.name for r in results]
        assert "alpha-dataset" in names
        assert "beta-dataset" in names

    def test_dataset_info_has_ground_truth_flag(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        import cri.datasets.loader as loader_module

        monkeypatch.setattr(loader_module, "CANONICAL_DATASETS_DIR", tmp_path)

        _write_dataset(tmp_path, name="with-gt")
        _write_dataset(tmp_path, name="no-gt", skip_ground_truth=True)

        results = list_canonical_datasets()
        by_name = {r.name: r for r in results}
        assert by_name["with-gt"].has_ground_truth is True
        assert by_name["no-gt"].has_ground_truth is False

    def test_message_count_from_jsonl(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        import cri.datasets.loader as loader_module

        monkeypatch.setattr(loader_module, "CANONICAL_DATASETS_DIR", tmp_path)

        _write_dataset(tmp_path, name="counted", messages_jsonl=_minimal_messages_jsonl(7))

        results = list_canonical_datasets()
        assert len(results) == 1
        assert results[0].message_count == 7

    def test_sorted_by_name(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        import cri.datasets.loader as loader_module

        monkeypatch.setattr(loader_module, "CANONICAL_DATASETS_DIR", tmp_path)

        for name in ["charlie", "alpha", "bravo"]:
            _write_dataset(tmp_path, name=name)

        results = list_canonical_datasets()
        names = [r.name for r in results]
        assert names == sorted(names)

    def test_ignores_files_at_top_level(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Files (not dirs) in the canonical directory should be skipped."""
        import cri.datasets.loader as loader_module

        monkeypatch.setattr(loader_module, "CANONICAL_DATASETS_DIR", tmp_path)

        _write_dataset(tmp_path, name="real-ds")
        (tmp_path / "README.md").write_text("# Hello", encoding="utf-8")

        results = list_canonical_datasets()
        assert len(results) == 1
        assert results[0].name == "real-ds"

    def test_ignores_dirs_without_conversations(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Directories without conversations.jsonl should be skipped."""
        import cri.datasets.loader as loader_module

        monkeypatch.setattr(loader_module, "CANONICAL_DATASETS_DIR", tmp_path)

        _write_dataset(tmp_path, name="valid")
        (tmp_path / "empty-dir").mkdir()

        results = list_canonical_datasets()
        assert len(results) == 1
        assert results[0].name == "valid"

    def test_dataset_info_path_is_absolute(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        import cri.datasets.loader as loader_module

        monkeypatch.setattr(loader_module, "CANONICAL_DATASETS_DIR", tmp_path)

        _write_dataset(tmp_path, name="abs-path-ds")

        results = list_canonical_datasets()
        assert results[0].path.is_absolute()


# ===================================================================
# Integration: round-trip load → validate
# ===================================================================


class TestIntegration:
    def test_load_and_validate_round_trip(self, tmp_path: Path) -> None:
        ds_dir = _write_dataset(tmp_path)
        dataset = load_dataset(ds_dir)
        errors = validate_dataset(dataset)
        assert errors == []

    def test_load_with_metadata_and_validate(self, tmp_path: Path) -> None:
        meta = {
            "dataset_id": "roundtrip",
            "persona_id": "p1",
            "message_count": 3,
            "simulated_days": 1,
            "version": "1.0.0",
            "seed": 42,
        }
        ds_dir = _write_dataset(tmp_path, metadata=meta)
        dataset = load_dataset(ds_dir)
        errors = validate_dataset(dataset)
        assert errors == []

    def test_load_validate_large_dataset(self, tmp_path: Path) -> None:
        """Round-trip with a large dataset (50 messages)."""
        meta = {
            "dataset_id": "large",
            "persona_id": "p1",
            "message_count": 50,
            "simulated_days": 30,
            "version": "1.0.0",
        }
        ds_dir = _write_dataset(
            tmp_path,
            name="large-ds",
            messages_jsonl=_minimal_messages_jsonl(50),
            metadata=meta,
        )
        dataset = load_dataset(ds_dir)
        errors = validate_dataset(dataset)
        assert errors == []
        assert len(dataset.messages) == 50

    def test_load_multiple_datasets_independently(self, tmp_path: Path) -> None:
        """Loading two different datasets should produce independent objects."""
        ds_a = _write_dataset(tmp_path, name="ds-a", messages_jsonl=_minimal_messages_jsonl(2))
        ds_b = _write_dataset(tmp_path, name="ds-b", messages_jsonl=_minimal_messages_jsonl(5))

        dataset_a = load_dataset(ds_a)
        dataset_b = load_dataset(ds_b)

        assert len(dataset_a.messages) == 2
        assert len(dataset_b.messages) == 5
        assert dataset_a.metadata.dataset_id == "ds-a"
        assert dataset_b.metadata.dataset_id == "ds-b"
