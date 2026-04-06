"""Tests for the CRI Benchmark dataset generator.

Verifies:
- PersonaSpec loading from canonical dataset
- DatasetGenerator deterministic generation (seed reproducibility)
- Correct message structure (sequential IDs, timestamps, roles)
- Ground truth assembly (profile dims, changes, conflicts)
- Suite generation
- save_dataset produces valid files loadable by the loader
- Round-trip: generate → save → load → validate
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from cri.datasets.generator import DatasetGenerator
from cri.datasets.loader import (
    get_persona,
    list_persona_specs,
    load_dataset,
    validate_dataset,
)
from cri.datasets.personas.specs import PersonaSpec
from cri.models import (
    ConversationDataset,
    GeneratorConfig,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def config() -> GeneratorConfig:
    """A reproducible generator config."""
    return GeneratorConfig(seed=42)


@pytest.fixture
def generator(config: GeneratorConfig) -> DatasetGenerator:
    """A seeded DatasetGenerator instance."""
    return DatasetGenerator(config)


@pytest.fixture
def persona_1_base() -> PersonaSpec:
    """The persona-1-base persona spec loaded from canonical data."""
    return get_persona("persona-1-base")


@pytest.fixture
def persona_1_dataset(generator: DatasetGenerator, persona_1_base: PersonaSpec) -> ConversationDataset:
    """Dataset generated from the persona-1-base persona."""
    return generator.generate(persona_1_base)


# ---------------------------------------------------------------------------
# PersonaSpec tests
# ---------------------------------------------------------------------------


class TestPersonaSpec:
    """Tests for persona specification loading and validity."""

    def test_list_persona_specs(self) -> None:
        specs = list_persona_specs()
        assert len(specs) >= 1

    def test_get_persona(self) -> None:
        spec = get_persona("persona-1-base")
        assert isinstance(spec, PersonaSpec)
        assert spec.persona_id == "persona-1-base"
        assert spec.name == "Marcus Rivera"

    def test_get_persona_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            get_persona("nonexistent-persona")

    def test_persona_has_required_fields(self, persona_1_base: PersonaSpec) -> None:
        assert persona_1_base.persona_id
        assert persona_1_base.name
        assert persona_1_base.description
        assert persona_1_base.complexity_level == "base"
        assert persona_1_base.profile_dimensions
        assert persona_1_base.simulated_days > 0
        assert persona_1_base.target_message_count > 0

    def test_persona_1_base_dimensions(self, persona_1_base: PersonaSpec) -> None:
        """persona-1-base should have 18 profile dimensions."""
        assert len(persona_1_base.profile_dimensions) == 18

    def test_persona_1_base_counts(self, persona_1_base: PersonaSpec) -> None:
        """persona-1-base should have exact counts from ground truth."""
        assert len(persona_1_base.belief_changes) == 7
        assert len(persona_1_base.conflicts) == 8
        assert len(persona_1_base.noise_examples) == 20
        assert len(persona_1_base.signal_examples) == 20
        assert len(persona_1_base.temporal_facts) == 12
        assert len(persona_1_base.query_relevance_pairs) == 20

    def test_profile_dimensions_cover_categories(self, persona_1_base: PersonaSpec) -> None:
        """Persona should have dimensions covering WHO/WHAT/WHERE/WHEN/WHY/HOW."""
        categories = {dim.category for dim in persona_1_base.profile_dimensions.values()}
        assert "who" in categories, "Missing WHO category"
        assert "what" in categories, "Missing WHAT category"
        assert "where" in categories, "Missing WHERE category"
        assert "when" in categories, "Missing WHEN category"
        assert "why" in categories, "Missing WHY category"
        assert "how" in categories, "Missing HOW category"


# ---------------------------------------------------------------------------
# DatasetGenerator tests
# ---------------------------------------------------------------------------


class TestDatasetGenerator:
    """Tests for the DatasetGenerator class."""

    def test_constructor(self, config: GeneratorConfig) -> None:
        gen = DatasetGenerator(config)
        assert gen.config is config

    def test_seed_reproducibility(self, persona_1_base: PersonaSpec) -> None:
        """Same seed must produce identical datasets."""
        g1 = DatasetGenerator(GeneratorConfig(seed=123))
        g2 = DatasetGenerator(GeneratorConfig(seed=123))
        ds1 = g1.generate(persona_1_base)
        ds2 = g2.generate(persona_1_base)

        assert len(ds1.messages) == len(ds2.messages)
        for m1, m2 in zip(ds1.messages, ds2.messages, strict=True):
            assert m1.message_id == m2.message_id
            assert m1.content == m2.content
            assert m1.role == m2.role
            assert m1.timestamp == m2.timestamp

    def test_different_seeds_produce_different_output(self, persona_1_base: PersonaSpec) -> None:
        g1 = DatasetGenerator(GeneratorConfig(seed=1))
        g2 = DatasetGenerator(GeneratorConfig(seed=2))
        ds1 = g1.generate(persona_1_base)
        ds2 = g2.generate(persona_1_base)

        # At least some messages should differ
        contents1 = {m.content for m in ds1.messages}
        contents2 = {m.content for m in ds2.messages}
        assert contents1 != contents2


class TestGenerateFromPersona:
    """Tests for generating a dataset from the canonical persona."""

    def test_message_count_reasonable(self, persona_1_dataset: ConversationDataset, persona_1_base: PersonaSpec) -> None:
        count = len(persona_1_dataset.messages)
        target = persona_1_base.target_message_count
        # Should be within ±50% of target
        assert count >= target * 0.5
        assert count <= target * 1.5

    def test_messages_have_sequential_ids(self, persona_1_dataset: ConversationDataset) -> None:
        ids = [m.message_id for m in persona_1_dataset.messages]
        assert ids == list(range(1, len(ids) + 1))

    def test_messages_have_valid_roles(self, persona_1_dataset: ConversationDataset) -> None:
        for msg in persona_1_dataset.messages:
            assert msg.role in ("user", "assistant")

    def test_messages_alternate_roles(self, persona_1_dataset: ConversationDataset) -> None:
        """Messages should alternate user → assistant."""
        for i in range(0, len(persona_1_dataset.messages) - 1, 2):
            assert persona_1_dataset.messages[i].role == "user"
            assert persona_1_dataset.messages[i + 1].role == "assistant"

    def test_timestamps_non_decreasing(self, persona_1_dataset: ConversationDataset) -> None:
        timestamps = [m.timestamp for m in persona_1_dataset.messages]
        for i in range(1, len(timestamps)):
            assert timestamps[i] >= timestamps[i - 1]

    def test_messages_have_session_ids(self, persona_1_dataset: ConversationDataset) -> None:
        for msg in persona_1_dataset.messages:
            assert msg.session_id is not None
            assert msg.session_id.startswith("session-")

    def test_messages_have_day_numbers(self, persona_1_dataset: ConversationDataset) -> None:
        for msg in persona_1_dataset.messages:
            assert msg.day is not None
            assert msg.day >= 1

    def test_ground_truth_has_profile(self, persona_1_dataset: ConversationDataset, persona_1_base: PersonaSpec) -> None:
        gt = persona_1_dataset.ground_truth
        assert len(gt.final_profile) > 0
        assert len(gt.final_profile) == len(persona_1_base.profile_dimensions)

    def test_ground_truth_has_changes(self, persona_1_dataset: ConversationDataset) -> None:
        assert len(persona_1_dataset.ground_truth.changes) == 7

    def test_ground_truth_has_conflicts(self, persona_1_dataset: ConversationDataset) -> None:
        assert len(persona_1_dataset.ground_truth.conflicts) == 8

    def test_ground_truth_has_temporal_facts(self, persona_1_dataset: ConversationDataset) -> None:
        assert len(persona_1_dataset.ground_truth.temporal_facts) == 12

    def test_ground_truth_has_query_relevance_pairs(self, persona_1_dataset: ConversationDataset) -> None:
        assert len(persona_1_dataset.ground_truth.query_relevance_pairs) == 20

    def test_metadata_correct(self, persona_1_dataset: ConversationDataset, persona_1_base: PersonaSpec) -> None:
        meta = persona_1_dataset.metadata
        assert meta.dataset_id == persona_1_base.persona_id
        assert meta.persona_id == persona_1_base.persona_id
        assert meta.message_count == len(persona_1_dataset.messages)
        assert meta.simulated_days == persona_1_base.simulated_days
        assert meta.seed == 42

    def test_change_references_in_bounds(self, persona_1_dataset: ConversationDataset) -> None:
        msg_count = len(persona_1_dataset.messages)
        for change in persona_1_dataset.ground_truth.changes:
            assert 1 <= change.changed_around_msg <= msg_count
            for km in change.key_messages:
                assert 1 <= km <= msg_count

    def test_conflict_references_in_bounds(self, persona_1_dataset: ConversationDataset) -> None:
        msg_count = len(persona_1_dataset.messages)
        for conflict in persona_1_dataset.ground_truth.conflicts:
            for mid in conflict.introduced_at_messages:
                assert 1 <= mid <= msg_count


class TestGenerateSuite:
    """Tests for the suite generation."""

    def test_returns_datasets(self, generator: DatasetGenerator) -> None:
        suite = generator.generate_suite()
        assert len(suite) >= 1

    def test_datasets_have_different_ids(self, generator: DatasetGenerator) -> None:
        suite = generator.generate_suite()
        ids = {ds.metadata.dataset_id for ds in suite}
        assert len(ids) == len(suite)


# ---------------------------------------------------------------------------
# save_dataset + round-trip tests
# ---------------------------------------------------------------------------


class TestSaveDataset:
    """Tests for persisting datasets to disk."""

    def test_creates_required_files(self, generator: DatasetGenerator, persona_1_base: PersonaSpec) -> None:
        ds = generator.generate(persona_1_base)
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "test-out"
            generator.save_dataset(ds, out)
            assert (out / "conversations.jsonl").exists()
            assert (out / "ground_truth.json").exists()
            assert (out / "metadata.json").exists()

    def test_conversations_jsonl_valid(self, generator: DatasetGenerator, persona_1_base: PersonaSpec) -> None:
        ds = generator.generate(persona_1_base)
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "test-out"
            generator.save_dataset(ds, out)
            with open(out / "conversations.jsonl") as f:
                lines = [line for line in f if line.strip()]
            assert len(lines) == len(ds.messages)
            for line in lines:
                obj = json.loads(line)
                assert "message_id" in obj
                assert "role" in obj
                assert "content" in obj

    def test_ground_truth_json_valid(self, generator: DatasetGenerator, persona_1_base: PersonaSpec) -> None:
        ds = generator.generate(persona_1_base)
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "test-out"
            generator.save_dataset(ds, out)
            with open(out / "ground_truth.json") as f:
                gt = json.load(f)
            assert "final_profile" in gt
            assert "changes" in gt
            assert "conflicts" in gt

    def test_metadata_json_valid(self, generator: DatasetGenerator, persona_1_base: PersonaSpec) -> None:
        ds = generator.generate(persona_1_base)
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "test-out"
            generator.save_dataset(ds, out)
            with open(out / "metadata.json") as f:
                meta = json.load(f)
            assert meta["dataset_id"] == ds.metadata.dataset_id
            assert meta["message_count"] == len(ds.messages)


class TestRoundTrip:
    """Tests that generated datasets survive save → load → validate."""

    @pytest.mark.parametrize("persona", list_persona_specs(), ids=lambda p: p.persona_id)
    def test_round_trip_all_personas(
        self,
        generator: DatasetGenerator,
        persona: PersonaSpec,
    ) -> None:
        ds = generator.generate(persona)

        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / persona.persona_id
            generator.save_dataset(ds, out)

            loaded = load_dataset(out)
            errors = validate_dataset(loaded)
            assert errors == [], f"Validation errors: {errors}"

            assert len(loaded.messages) == len(ds.messages)
            assert loaded.metadata.dataset_id == ds.metadata.dataset_id
            assert len(loaded.ground_truth.final_profile) == len(ds.ground_truth.final_profile)
