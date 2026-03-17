"""Tests for the CRI Benchmark CLI runner.

Tests cover:
- Adapter registry (get_adapter_registry, resolve_adapter)
- Dynamic adapter loading (load_adapter_class)
- CLI commands (list-adapters, list-datasets, validate-dataset, run)
- Pipeline smoke test with mocked judge
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from cri.models import Message, StoredFact
from cri.runner import (
    _ADAPTER_ENTRIES,
    get_adapter_registry,
    load_adapter_class,
    main,
    resolve_adapter,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cli_runner() -> CliRunner:
    """Click CLI test runner."""
    return CliRunner()


@pytest.fixture
def tmp_dataset(tmp_path: Path) -> Path:
    """Create a minimal valid dataset directory for testing."""
    ds_dir = tmp_path / "test-dataset"
    ds_dir.mkdir()

    # conversations.jsonl
    messages = [
        {
            "message_id": 1,
            "role": "user",
            "content": "Hi, I'm Alice. I work as an engineer.",
            "timestamp": "2024-01-01T10:00:00Z",
        },
        {
            "message_id": 2,
            "role": "assistant",
            "content": "Nice to meet you, Alice!",
            "timestamp": "2024-01-01T10:01:00Z",
        },
    ]
    with open(ds_dir / "conversations.jsonl", "w") as f:
        for msg in messages:
            f.write(json.dumps(msg) + "\n")

    # ground_truth.json
    ground_truth = {
        "final_profile": {
            "occupation": {
                "dimension_name": "occupation",
                "value": "engineer",
                "query_topic": "current occupation",
            }
        },
        "changes": [],
        "noise_examples": [],
        "signal_examples": [],
        "conflicts": [],
        "temporal_facts": [],
        "query_relevance_pairs": [],
    }
    with open(ds_dir / "ground_truth.json", "w") as f:
        json.dump(ground_truth, f)

    # metadata.json
    metadata = {
        "dataset_id": "test-dataset",
        "persona_id": "alice",
        "message_count": 2,
        "simulated_days": 1,
        "version": "1.0.0",
        "seed": 42,
    }
    with open(ds_dir / "metadata.json", "w") as f:
        json.dump(metadata, f)

    return ds_dir


@pytest.fixture
def invalid_dataset(tmp_path: Path) -> Path:
    """Create a dataset directory with missing required files."""
    ds_dir = tmp_path / "bad-dataset"
    ds_dir.mkdir()
    # Only create an empty conversations.jsonl — no ground_truth.json
    (ds_dir / "conversations.jsonl").write_text("")
    return ds_dir


# ---------------------------------------------------------------------------
# Adapter registry tests
# ---------------------------------------------------------------------------


class TestAdapterRegistry:
    """Tests for adapter registry functions."""

    def test_adapter_entries_contains_expected_names(self) -> None:
        """The static entry list should include the three baseline adapters."""
        assert "no-memory" in _ADAPTER_ENTRIES
        assert "full-context" in _ADAPTER_ENTRIES
        assert "rag" in _ADAPTER_ENTRIES

    def test_get_adapter_registry_returns_dict(self) -> None:
        """get_adapter_registry should return a dict."""
        registry = get_adapter_registry()
        assert isinstance(registry, dict)

    def test_get_adapter_registry_values_are_types(self) -> None:
        """Each value in the registry should be a class (type)."""
        registry = get_adapter_registry()
        for name, cls in registry.items():
            assert isinstance(cls, type), f"'{name}' is not a type: {type(cls)}"

    def test_registry_entries_have_descriptions(self) -> None:
        """Each registry entry should have a non-empty description."""
        for name, (_mod, _cls, desc, _extra) in _ADAPTER_ENTRIES.items():
            assert desc, f"Adapter '{name}' has no description"


# ---------------------------------------------------------------------------
# Dynamic adapter loading tests
# ---------------------------------------------------------------------------


class TestLoadAdapterClass:
    """Tests for load_adapter_class()."""

    def test_colon_syntax(self) -> None:
        """Should load a class using 'module:Class' syntax."""
        cls = load_adapter_class("cri.models:Message")
        assert cls is Message

    def test_dot_syntax(self) -> None:
        """Should load a class using 'module.Class' syntax."""
        cls = load_adapter_class("cri.models.StoredFact")
        assert cls is StoredFact

    def test_invalid_path_no_dots(self) -> None:
        """Should raise ValueError for paths without dots or colons."""
        with pytest.raises(ValueError, match="Invalid adapter path"):
            load_adapter_class("SomeClass")

    def test_missing_module(self) -> None:
        """Should raise ImportError for non-existent modules."""
        with pytest.raises(ImportError, match="Cannot import module"):
            load_adapter_class("nonexistent.module:SomeClass")

    def test_missing_class(self) -> None:
        """Should raise ValueError when the class doesn't exist in the module."""
        with pytest.raises(ValueError, match="has no attribute"):
            load_adapter_class("cri.models:NonExistentClass")

    def test_not_a_class(self) -> None:
        """Should raise ValueError when the target is not a class."""
        with pytest.raises(ValueError, match="not a class"):
            load_adapter_class("cri:__version__")

    def test_resolve_adapter_with_dotted_path(self) -> None:
        """resolve_adapter should fall back to dynamic loading for dotted paths."""
        cls = resolve_adapter("cri.models:Message")
        assert cls is Message


# ---------------------------------------------------------------------------
# CLI command tests
# ---------------------------------------------------------------------------


class TestCLIListAdapters:
    """Tests for the 'cri list-adapters' command."""

    def test_list_adapters_exits_zero(self, cli_runner: CliRunner) -> None:
        result = cli_runner.invoke(main, ["list-adapters"])
        assert result.exit_code == 0

    def test_list_adapters_shows_names(self, cli_runner: CliRunner) -> None:
        result = cli_runner.invoke(main, ["list-adapters"])
        assert "no-memory" in result.output
        assert "full-context" in result.output
        assert "rag" in result.output

    def test_list_adapters_shows_descriptions(self, cli_runner: CliRunner) -> None:
        result = cli_runner.invoke(main, ["list-adapters"])
        assert "Discards all input" in result.output or "lower" in result.output.lower()


class TestCLIListDatasets:
    """Tests for the 'cri list-datasets' command."""

    def test_list_datasets_exits_zero(self, cli_runner: CliRunner) -> None:
        result = cli_runner.invoke(main, ["list-datasets"])
        assert result.exit_code == 0

    def test_list_datasets_shows_header(self, cli_runner: CliRunner) -> None:
        result = cli_runner.invoke(main, ["list-datasets"])
        # Should show the header regardless of whether datasets exist
        assert "Datasets" in result.output or "dataset" in result.output.lower()


class TestCLIValidateDataset:
    """Tests for the 'cri validate-dataset' command."""

    def test_validate_valid_dataset(self, cli_runner: CliRunner, tmp_dataset: Path) -> None:
        result = cli_runner.invoke(main, ["validate-dataset", str(tmp_dataset)])
        assert result.exit_code == 0
        assert "valid" in result.output.lower() or "✓" in result.output

    def test_validate_missing_directory(self, cli_runner: CliRunner) -> None:
        result = cli_runner.invoke(main, ["validate-dataset", "/nonexistent/path"])
        assert result.exit_code != 0

    def test_validate_missing_ground_truth(self, cli_runner: CliRunner, invalid_dataset: Path) -> None:
        result = cli_runner.invoke(main, ["validate-dataset", str(invalid_dataset)])
        assert result.exit_code != 0

    def test_validate_shows_stats(self, cli_runner: CliRunner, tmp_dataset: Path) -> None:
        """Valid dataset validation should show summary stats."""
        result = cli_runner.invoke(main, ["validate-dataset", str(tmp_dataset)])
        assert result.exit_code == 0
        assert "Messages" in result.output or "messages" in result.output.lower()


class TestCLIRun:
    """Tests for the 'cri run' command."""

    def test_run_requires_adapter(self, cli_runner: CliRunner) -> None:
        """Should fail if --adapter is not provided."""
        result = cli_runner.invoke(main, ["run", "--dataset", "/tmp"])
        assert result.exit_code != 0

    def test_run_requires_dataset(self, cli_runner: CliRunner) -> None:
        """Should fail if --dataset is not provided."""
        result = cli_runner.invoke(main, ["run", "--adapter", "no-memory"])
        assert result.exit_code != 0

    def test_run_invalid_adapter_name(self, cli_runner: CliRunner, tmp_dataset: Path) -> None:
        """Should fail gracefully with an invalid adapter name."""
        result = cli_runner.invoke(
            main,
            ["run", "--adapter", "nonexistent-adapter", "--dataset", str(tmp_dataset)],
        )
        assert result.exit_code != 0

    def test_run_nonexistent_dataset(self, cli_runner: CliRunner) -> None:
        """Should fail when dataset path doesn't exist."""
        result = cli_runner.invoke(
            main,
            ["run", "--adapter", "no-memory", "--dataset", "/tmp/nonexistent_dataset_path_xyz"],
        )
        assert result.exit_code != 0


class TestCLIVersion:
    """Tests for --version flag."""

    def test_version_flag(self, cli_runner: CliRunner) -> None:
        result = cli_runner.invoke(main, ["--version"])
        assert result.exit_code == 0
        assert "version" in result.output.lower() or "0." in result.output


class TestCLIHelp:
    """Tests for --help flag."""

    def test_main_help(self, cli_runner: CliRunner) -> None:
        result = cli_runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "CRI Benchmark" in result.output

    def test_run_help(self, cli_runner: CliRunner) -> None:
        result = cli_runner.invoke(main, ["run", "--help"])
        assert result.exit_code == 0
        assert "--adapter" in result.output
        assert "--dataset" in result.output
        assert "--judge-runs" in result.output
        assert "--format" in result.output

    def test_list_adapters_help(self, cli_runner: CliRunner) -> None:
        result = cli_runner.invoke(main, ["list-adapters", "--help"])
        assert result.exit_code == 0

    def test_validate_dataset_help(self, cli_runner: CliRunner) -> None:
        result = cli_runner.invoke(main, ["validate-dataset", "--help"])
        assert result.exit_code == 0
