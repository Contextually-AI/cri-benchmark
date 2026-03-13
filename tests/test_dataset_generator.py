"""Tests for the CRI Benchmark dataset generator.

Verifies:
- RichPersonaSpec construction and canonical persona validity
- DatasetGenerator deterministic generation (seed reproducibility)
- Correct message structure (sequential IDs, timestamps, roles)
- Ground truth assembly (profile dims, changes, conflicts)
- Canonical suite generation (all 3 personas)
- save_dataset produces valid files loadable by the loader
- Round-trip: generate → save → load → validate
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from cri.datasets.generator import DatasetGenerator
from cri.datasets.loader import load_dataset, validate_dataset
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
def basic_dataset(generator: DatasetGenerator) -> ConversationDataset:
    """Dataset generated from the basic persona."""
    return generator.generate(PERSONA_BASIC)


@pytest.fixture
def intermediate_dataset(generator: DatasetGenerator) -> ConversationDataset:
    """Dataset generated from the intermediate persona."""
    return generator.generate(PERSONA_INTERMEDIATE)


@pytest.fixture
def advanced_dataset(generator: DatasetGenerator) -> ConversationDataset:
    """Dataset generated from the advanced persona."""
    return generator.generate(PERSONA_ADVANCED)


# ---------------------------------------------------------------------------
# RichPersonaSpec tests
# ---------------------------------------------------------------------------


class TestRichPersonaSpec:
    """Tests for persona specification validity."""

    def test_all_personas_list(self) -> None:
        assert len(ALL_PERSONAS) == 3
        assert ALL_PERSONAS[0] is PERSONA_BASIC
        assert ALL_PERSONAS[1] is PERSONA_INTERMEDIATE
        assert ALL_PERSONAS[2] is PERSONA_ADVANCED

    @pytest.mark.parametrize("persona", ALL_PERSONAS, ids=lambda p: p.persona_id)
    def test_persona_has_required_fields(self, persona: RichPersonaSpec) -> None:
        assert persona.persona_id
        assert persona.name
        assert persona.description
        assert persona.complexity_level in ("basic", "intermediate", "advanced")
        assert persona.profile_dimensions
        assert persona.simulated_days > 0
        assert persona.target_message_count > 0

    def test_basic_persona_has_changes_and_conflicts(self) -> None:
        """Basic persona now has belief changes and conflicts."""
        assert len(PERSONA_BASIC.belief_changes) == 3
        assert len(PERSONA_BASIC.conflicts) == 3

    def test_basic_persona_dimensions(self) -> None:
        """Basic persona should have 10 profile dimensions."""
        assert len(PERSONA_BASIC.profile_dimensions) == 10

    def test_basic_persona_counts(self) -> None:
        """Basic persona should have exact counts per spec."""
        assert len(PERSONA_BASIC.noise_examples) == 10
        assert len(PERSONA_BASIC.signal_examples) == 10
        assert len(PERSONA_BASIC.temporal_facts) == 5
        assert len(PERSONA_BASIC.query_relevance_pairs) == 10
        assert PERSONA_BASIC.target_message_count == 1000
        assert PERSONA_BASIC.simulated_days == 30

    def test_intermediate_persona_has_changes(self) -> None:
        assert len(PERSONA_INTERMEDIATE.belief_changes) == 5
        assert len(PERSONA_INTERMEDIATE.conflicts) == 5

    def test_intermediate_persona_dimensions(self) -> None:
        """Intermediate persona should have 14 profile dimensions."""
        assert len(PERSONA_INTERMEDIATE.profile_dimensions) == 14

    def test_intermediate_persona_counts(self) -> None:
        """Intermediate persona should have exact counts per spec."""
        assert len(PERSONA_INTERMEDIATE.noise_examples) == 15
        assert len(PERSONA_INTERMEDIATE.signal_examples) == 15
        assert len(PERSONA_INTERMEDIATE.temporal_facts) == 8
        assert len(PERSONA_INTERMEDIATE.query_relevance_pairs) == 15
        assert PERSONA_INTERMEDIATE.target_message_count == 2000
        assert PERSONA_INTERMEDIATE.simulated_days == 60

    def test_advanced_persona_is_most_complex(self) -> None:
        assert len(PERSONA_ADVANCED.belief_changes) == 7
        assert len(PERSONA_ADVANCED.conflicts) == 8
        assert len(PERSONA_ADVANCED.temporal_facts) == 12
        assert PERSONA_ADVANCED.target_message_count > PERSONA_INTERMEDIATE.target_message_count

    def test_advanced_persona_dimensions(self) -> None:
        """Advanced persona should have 18 profile dimensions."""
        assert len(PERSONA_ADVANCED.profile_dimensions) == 18

    def test_advanced_persona_counts(self) -> None:
        """Advanced persona should have exact counts per spec."""
        assert len(PERSONA_ADVANCED.noise_examples) == 20
        assert len(PERSONA_ADVANCED.signal_examples) == 20
        assert len(PERSONA_ADVANCED.query_relevance_pairs) == 20
        assert PERSONA_ADVANCED.target_message_count == 3000
        assert PERSONA_ADVANCED.simulated_days == 120

    def test_persona_names(self) -> None:
        """Verify persona names match the spec."""
        assert PERSONA_BASIC.name == "Alex Chen"
        assert PERSONA_INTERMEDIATE.name == "Sarah Miller"
        assert PERSONA_ADVANCED.name == "Marcus Rivera"

    def test_profile_dimensions_cover_categories(self) -> None:
        """All personas should have dimensions covering WHO/WHAT/WHERE/WHEN/WHY/HOW."""
        for persona in ALL_PERSONAS:
            categories = {dim.category for dim in persona.profile_dimensions.values()}
            assert "who" in categories, f"{persona.name} missing WHO category"
            assert "what" in categories, f"{persona.name} missing WHAT category"
            assert "where" in categories, f"{persona.name} missing WHERE category"
            assert "when" in categories, f"{persona.name} missing WHEN category"
            assert "why" in categories, f"{persona.name} missing WHY category"
            assert "how" in categories, f"{persona.name} missing HOW category"


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------


class TestHelperFunctions:
    """Tests for the get_persona_* helper functions."""

    def test_get_persona_basic(self) -> None:
        result = get_persona_basic()
        assert result is PERSONA_BASIC
        assert isinstance(result, RichPersonaSpec)
        assert result.name == "Alex Chen"

    def test_get_persona_intermediate(self) -> None:
        result = get_persona_intermediate()
        assert result is PERSONA_INTERMEDIATE
        assert isinstance(result, RichPersonaSpec)
        assert result.name == "Sarah Miller"

    def test_get_persona_advanced(self) -> None:
        result = get_persona_advanced()
        assert result is PERSONA_ADVANCED
        assert isinstance(result, RichPersonaSpec)
        assert result.name == "Marcus Rivera"


# ---------------------------------------------------------------------------
# DatasetGenerator tests
# ---------------------------------------------------------------------------


class TestDatasetGenerator:
    """Tests for the DatasetGenerator class."""

    def test_constructor(self, config: GeneratorConfig) -> None:
        gen = DatasetGenerator(config)
        assert gen.config is config

    def test_seed_reproducibility(self) -> None:
        """Same seed must produce identical datasets."""
        g1 = DatasetGenerator(GeneratorConfig(seed=123))
        g2 = DatasetGenerator(GeneratorConfig(seed=123))
        ds1 = g1.generate(PERSONA_BASIC)
        ds2 = g2.generate(PERSONA_BASIC)

        assert len(ds1.messages) == len(ds2.messages)
        for m1, m2 in zip(ds1.messages, ds2.messages, strict=True):
            assert m1.message_id == m2.message_id
            assert m1.content == m2.content
            assert m1.role == m2.role
            assert m1.timestamp == m2.timestamp

    def test_different_seeds_produce_different_output(self) -> None:
        g1 = DatasetGenerator(GeneratorConfig(seed=1))
        g2 = DatasetGenerator(GeneratorConfig(seed=2))
        ds1 = g1.generate(PERSONA_BASIC)
        ds2 = g2.generate(PERSONA_BASIC)

        # At least some messages should differ
        contents1 = {m.content for m in ds1.messages}
        contents2 = {m.content for m in ds2.messages}
        assert contents1 != contents2


class TestGenerateBasic:
    """Tests for generating a basic-complexity dataset."""

    def test_message_count_reasonable(self, basic_dataset: ConversationDataset) -> None:
        count = len(basic_dataset.messages)
        target = PERSONA_BASIC.target_message_count
        # Should be within ±50% of target
        assert count >= target * 0.5
        assert count <= target * 1.5

    def test_messages_have_sequential_ids(self, basic_dataset: ConversationDataset) -> None:
        ids = [m.message_id for m in basic_dataset.messages]
        assert ids == list(range(1, len(ids) + 1))

    def test_messages_have_valid_roles(self, basic_dataset: ConversationDataset) -> None:
        for msg in basic_dataset.messages:
            assert msg.role in ("user", "assistant")

    def test_messages_alternate_roles(self, basic_dataset: ConversationDataset) -> None:
        """Messages should alternate user → assistant."""
        for i in range(0, len(basic_dataset.messages) - 1, 2):
            assert basic_dataset.messages[i].role == "user"
            assert basic_dataset.messages[i + 1].role == "assistant"

    def test_timestamps_non_decreasing(self, basic_dataset: ConversationDataset) -> None:
        timestamps = [m.timestamp for m in basic_dataset.messages]
        for i in range(1, len(timestamps)):
            assert timestamps[i] >= timestamps[i - 1]

    def test_messages_have_session_ids(self, basic_dataset: ConversationDataset) -> None:
        for msg in basic_dataset.messages:
            assert msg.session_id is not None
            assert msg.session_id.startswith("session-")

    def test_messages_have_day_numbers(self, basic_dataset: ConversationDataset) -> None:
        for msg in basic_dataset.messages:
            assert msg.day is not None
            assert msg.day >= 1

    def test_ground_truth_has_profile(self, basic_dataset: ConversationDataset) -> None:
        gt = basic_dataset.ground_truth
        assert len(gt.final_profile) > 0
        assert len(gt.final_profile) == len(PERSONA_BASIC.profile_dimensions)

    def test_ground_truth_has_changes(self, basic_dataset: ConversationDataset) -> None:
        """Basic persona now has belief changes."""
        assert len(basic_dataset.ground_truth.changes) == 3

    def test_ground_truth_has_conflicts(self, basic_dataset: ConversationDataset) -> None:
        """Basic persona now has conflicts."""
        assert len(basic_dataset.ground_truth.conflicts) == 3

    def test_metadata_correct(self, basic_dataset: ConversationDataset) -> None:
        meta = basic_dataset.metadata
        assert meta.dataset_id == PERSONA_BASIC.persona_id
        assert meta.persona_id == PERSONA_BASIC.persona_id
        assert meta.message_count == len(basic_dataset.messages)
        assert meta.simulated_days == PERSONA_BASIC.simulated_days
        assert meta.version == "1.0.0"
        assert meta.seed == 42


class TestGenerateIntermediate:
    """Tests for generating an intermediate-complexity dataset."""

    def test_has_belief_changes(self, intermediate_dataset: ConversationDataset) -> None:
        gt = intermediate_dataset.ground_truth
        assert len(gt.changes) == 5

    def test_has_conflicts(self, intermediate_dataset: ConversationDataset) -> None:
        gt = intermediate_dataset.ground_truth
        assert len(gt.conflicts) == 5

    def test_change_references_in_bounds(self, intermediate_dataset: ConversationDataset) -> None:
        msg_count = len(intermediate_dataset.messages)
        for change in intermediate_dataset.ground_truth.changes:
            assert 1 <= change.changed_around_msg <= msg_count
            for km in change.key_messages:
                assert 1 <= km <= msg_count

    def test_conflict_references_in_bounds(self, intermediate_dataset: ConversationDataset) -> None:
        msg_count = len(intermediate_dataset.messages)
        for conflict in intermediate_dataset.ground_truth.conflicts:
            for mid in conflict.introduced_at_messages:
                assert 1 <= mid <= msg_count


class TestGenerateAdvanced:
    """Tests for generating an advanced-complexity dataset."""

    def test_message_count_higher(self, advanced_dataset: ConversationDataset) -> None:
        assert len(advanced_dataset.messages) > 200

    def test_has_many_belief_changes(self, advanced_dataset: ConversationDataset) -> None:
        assert len(advanced_dataset.ground_truth.changes) == 7

    def test_has_multiple_conflicts(self, advanced_dataset: ConversationDataset) -> None:
        assert len(advanced_dataset.ground_truth.conflicts) == 8

    def test_has_temporal_facts(self, advanced_dataset: ConversationDataset) -> None:
        assert len(advanced_dataset.ground_truth.temporal_facts) == 12

    def test_has_query_relevance_pairs(self, advanced_dataset: ConversationDataset) -> None:
        assert len(advanced_dataset.ground_truth.query_relevance_pairs) == 20


class TestGenerateCanonicalSuite:
    """Tests for the canonical suite generation."""

    def test_returns_three_datasets(self, generator: DatasetGenerator) -> None:
        suite = generator.generate_canonical_suite()
        assert len(suite) == 3

    def test_datasets_have_different_ids(self, generator: DatasetGenerator) -> None:
        suite = generator.generate_canonical_suite()
        ids = {ds.metadata.dataset_id for ds in suite}
        assert len(ids) == 3

    def test_complexity_ordering(self, generator: DatasetGenerator) -> None:
        suite = generator.generate_canonical_suite()
        # Advanced should have the most messages
        msg_counts = [len(ds.messages) for ds in suite]
        assert msg_counts[2] > msg_counts[0]  # advanced > basic


# ---------------------------------------------------------------------------
# save_dataset + round-trip tests
# ---------------------------------------------------------------------------


class TestSaveDataset:
    """Tests for persisting datasets to disk."""

    def test_creates_required_files(self, generator: DatasetGenerator) -> None:
        ds = generator.generate(PERSONA_BASIC)
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "test-out"
            generator.save_dataset(ds, out)
            assert (out / "conversations.jsonl").exists()
            assert (out / "ground_truth.json").exists()
            assert (out / "metadata.json").exists()

    def test_conversations_jsonl_valid(self, generator: DatasetGenerator) -> None:
        ds = generator.generate(PERSONA_BASIC)
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

    def test_ground_truth_json_valid(self, generator: DatasetGenerator) -> None:
        ds = generator.generate(PERSONA_BASIC)
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "test-out"
            generator.save_dataset(ds, out)
            with open(out / "ground_truth.json") as f:
                gt = json.load(f)
            assert "final_profile" in gt
            assert "changes" in gt
            assert "conflicts" in gt

    def test_metadata_json_valid(self, generator: DatasetGenerator) -> None:
        ds = generator.generate(PERSONA_BASIC)
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "test-out"
            generator.save_dataset(ds, out)
            with open(out / "metadata.json") as f:
                meta = json.load(f)
            assert meta["dataset_id"] == ds.metadata.dataset_id
            assert meta["message_count"] == len(ds.messages)


class TestRoundTrip:
    """Tests that generated datasets survive save → load → validate."""

    @pytest.mark.parametrize("persona", ALL_PERSONAS, ids=lambda p: p.persona_id)
    def test_round_trip_all_personas(
        self,
        generator: DatasetGenerator,
        persona: RichPersonaSpec,
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
