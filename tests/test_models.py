"""Comprehensive tests for CRI core data models.

Covers:
- Round-trip serialization (model_dump → model_validate and model_dump_json → model_validate_json)
- Required field validation (missing required fields raise ValidationError)
- Default values for optional / defaulted fields
- Verdict enum correctness
- ScoringConfig weights sum to 1.0
- Edge cases and boundary conditions
- Instance independence for mutable defaults
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from cri.models import (
    BeliefChange,
    BenchmarkResult,
    ConflictScenario,
    ConversationDataset,
    CRIResult,
    DatasetMetadata,
    Dimension,
    DimensionResult,
    GeneratorConfig,
    GroundTruth,
    JudgmentResult,
    Message,
    NoiseExample,
    PerformanceProfile,
    ProfileDimension,
    QueryRelevancePair,
    ScoringConfig,
    ScoringProfile,
    SignalExample,
    StoredFact,
    TemporalFact,
    Verdict,
)

# ===================================================================
# Helpers
# ===================================================================


def assert_roundtrip(model_cls, instance):
    """Assert dict round-trip: model_dump → model_validate == original."""
    data = instance.model_dump()
    restored = model_cls.model_validate(data)
    assert restored == instance, f"Dict round-trip failed for {model_cls.__name__}"


def assert_json_roundtrip(model_cls, instance):
    """Assert JSON round-trip: model_dump_json → model_validate_json preserves data."""
    json_str = instance.model_dump_json()
    restored = model_cls.model_validate_json(json_str)
    # Compare key fields rather than strict equality (datetime serialisation may differ)
    assert restored.model_dump() == instance.model_dump(), (
        f"JSON round-trip field mismatch for {model_cls.__name__}"
    )


# ===================================================================
# Dimension enum
# ===================================================================


class TestDimension:
    def test_all_dimensions_exist(self) -> None:
        assert len(Dimension) == 7
        dims = {d.value for d in Dimension}
        assert dims == {
            "pas",
            "dbu",
            "tc",
            "crq",
            "qrp",
            "mei",
            "sfc",
        }

    def test_dimension_from_string(self) -> None:
        assert Dimension("pas") == Dimension.PAS
        assert Dimension("dbu") == Dimension.DBU
        assert Dimension("tc") == Dimension.TC
        assert Dimension("crq") == Dimension.CRQ
        assert Dimension("qrp") == Dimension.QRP
        assert Dimension("mei") == Dimension.MEI

    def test_dimension_is_string(self) -> None:
        """Dimension inherits from str, so it IS a string."""
        assert isinstance(Dimension.PAS, str)
        assert Dimension.PAS == "pas"

    def test_invalid_dimension_raises(self) -> None:
        with pytest.raises(ValueError):
            Dimension("nonexistent")

    def test_dimension_iteration(self) -> None:
        dims = list(Dimension)
        assert len(dims) == 7


# ===================================================================
# Verdict enum
# ===================================================================


class TestVerdict:
    def test_verdict_values(self) -> None:
        assert Verdict.YES.value == "YES"
        assert Verdict.NO.value == "NO"

    def test_verdict_members(self) -> None:
        assert len(Verdict) == 2

    def test_verdict_is_not_string(self) -> None:
        """Verdict is a plain Enum, not str Enum."""
        assert not isinstance(Verdict.YES, str)

    def test_verdict_from_value(self) -> None:
        assert Verdict("YES") == Verdict.YES
        assert Verdict("NO") == Verdict.NO

    def test_verdict_invalid_value_raises(self) -> None:
        with pytest.raises(ValueError):
            Verdict("MAYBE")

    def test_verdict_identity(self) -> None:
        assert Verdict.YES is Verdict.YES
        assert Verdict.YES is not Verdict.NO

    def test_verdict_name_vs_value(self) -> None:
        assert Verdict.YES.name == "YES"
        assert Verdict.NO.name == "NO"


# ===================================================================
# Message
# ===================================================================


class TestMessage:
    def test_create_message_minimal(self) -> None:
        msg = Message(
            message_id=1,
            role="user",
            content="Hello, I'm Alice.",
            timestamp="2026-01-01T10:00:00Z",
        )
        assert msg.message_id == 1
        assert msg.role == "user"
        assert msg.content == "Hello, I'm Alice."

    def test_default_optional_fields(self) -> None:
        msg = Message(
            message_id=1,
            role="user",
            content="test",
            timestamp="2026-01-01T00:00:00Z",
        )
        assert msg.session_id is None
        assert msg.day is None

    def test_message_with_all_fields(self) -> None:
        msg = Message(
            message_id=2,
            role="assistant",
            content="Nice to meet you!",
            timestamp="2026-01-01T10:00:01Z",
            session_id="sess-001",
            day=1,
        )
        assert msg.session_id == "sess-001"
        assert msg.day == 1

    def test_message_role_literal_user(self) -> None:
        msg = Message(message_id=1, role="user", content="t", timestamp="2026-01-01T00:00:00Z")
        assert msg.role == "user"

    def test_message_role_literal_assistant(self) -> None:
        msg = Message(message_id=1, role="assistant", content="t", timestamp="2026-01-01T00:00:00Z")
        assert msg.role == "assistant"

    def test_message_invalid_role_raises(self) -> None:
        with pytest.raises(ValidationError):
            Message(
                message_id=1,
                role="system",  # type: ignore[arg-type]
                content="test",
                timestamp="2026-01-01T00:00:00Z",
            )

    def test_message_missing_required_fields(self) -> None:
        with pytest.raises(ValidationError):
            Message(role="user", content="test", timestamp="2026-01-01T00:00:00Z")  # type: ignore[call-arg]

        with pytest.raises(ValidationError):
            Message(message_id=1, content="test", timestamp="2026-01-01T00:00:00Z")  # type: ignore[call-arg]

        with pytest.raises(ValidationError):
            Message(message_id=1, role="user", timestamp="2026-01-01T00:00:00Z")  # type: ignore[call-arg]

        with pytest.raises(ValidationError):
            Message(message_id=1, role="user", content="test")  # type: ignore[call-arg]

    def test_message_roundtrip(self) -> None:
        msg = Message(
            message_id=1,
            role="user",
            content="test",
            timestamp="2026-01-01T00:00:00Z",
            day=5,
            session_id="s1",
        )
        assert_roundtrip(Message, msg)

    def test_message_json_roundtrip(self) -> None:
        msg = Message(
            message_id=99,
            role="assistant",
            content="hi",
            timestamp="2026-06-15T12:30:00Z",
            session_id="sess-x",
            day=42,
        )
        assert_json_roundtrip(Message, msg)

    def test_message_empty_content(self) -> None:
        msg = Message(message_id=1, role="user", content="", timestamp="2026-01-01T00:00:00Z")
        assert msg.content == ""


# ===================================================================
# StoredFact
# ===================================================================


class TestStoredFact:
    def test_create_stored_fact(self) -> None:
        fact = StoredFact(text="User likes hiking")
        assert fact.text == "User likes hiking"
        assert fact.metadata == {}

    def test_stored_fact_with_metadata(self) -> None:
        fact = StoredFact(text="Fact", metadata={"source": "msg-5"})
        assert fact.metadata["source"] == "msg-5"

    def test_stored_fact_default_metadata_is_independent(self) -> None:
        f1 = StoredFact(text="A")
        f2 = StoredFact(text="B")
        f1.metadata["x"] = 1
        assert "x" not in f2.metadata

    def test_stored_fact_missing_text_raises(self) -> None:
        with pytest.raises(ValidationError):
            StoredFact()  # type: ignore[call-arg]

    def test_stored_fact_roundtrip(self) -> None:
        fact = StoredFact(text="Fact X", metadata={"k": "v"})
        assert_roundtrip(StoredFact, fact)

    def test_stored_fact_json_roundtrip(self) -> None:
        fact = StoredFact(text="Fact Y", metadata={"num": 42})
        assert_json_roundtrip(StoredFact, fact)


# ===================================================================
# ProfileDimension
# ===================================================================


class TestProfileDimension:
    def test_create_with_string_value(self) -> None:
        pd = ProfileDimension(
            dimension_name="occupation",
            value="software engineer",
            query_topic="What does the user do for work?",
        )
        assert pd.value == "software engineer"
        assert pd.category is None

    def test_create_with_list_value(self) -> None:
        pd = ProfileDimension(
            dimension_name="hobbies",
            value=["hiking", "photography"],
            query_topic="What are the user's hobbies?",
            category="interests",
        )
        assert isinstance(pd.value, list)
        assert len(pd.value) == 2
        assert pd.category == "interests"

    def test_default_category_is_none(self) -> None:
        pd = ProfileDimension(dimension_name="a", value="b", query_topic="c")
        assert pd.category is None

    def test_missing_required_fields_raises(self) -> None:
        with pytest.raises(ValidationError):
            ProfileDimension(value="x", query_topic="y")  # type: ignore[call-arg]

        with pytest.raises(ValidationError):
            ProfileDimension(dimension_name="x", query_topic="y")  # type: ignore[call-arg]

    def test_roundtrip(self) -> None:
        pd = ProfileDimension(
            dimension_name="age",
            value="30",
            query_topic="How old is the user?",
            category="demo",
        )
        assert_roundtrip(ProfileDimension, pd)

    def test_json_roundtrip(self) -> None:
        pd = ProfileDimension(
            dimension_name="hobbies",
            value=["a", "b"],
            query_topic="hobbies?",
        )
        assert_json_roundtrip(ProfileDimension, pd)


# ===================================================================
# BeliefChange
# ===================================================================


class TestBeliefChange:
    def test_create_belief_change(self) -> None:
        bc = BeliefChange(
            fact="phone_os",
            old_value="iOS",
            new_value="Android",
            query_topic="What phone does the user use?",
            changed_around_msg=42,
        )
        assert bc.old_value == "iOS"
        assert bc.new_value == "Android"
        assert bc.key_messages == []

    def test_default_key_messages_empty(self) -> None:
        bc = BeliefChange(
            fact="f",
            old_value="a",
            new_value="b",
            query_topic="q",
            changed_around_msg=1,
        )
        assert bc.key_messages == []

    def test_key_messages_independence(self) -> None:
        bc1 = BeliefChange(
            fact="f",
            old_value="a",
            new_value="b",
            query_topic="q",
            changed_around_msg=1,
        )
        bc2 = BeliefChange(
            fact="f",
            old_value="a",
            new_value="b",
            query_topic="q",
            changed_around_msg=1,
        )
        bc1.key_messages.append(99)
        assert 99 not in bc2.key_messages

    def test_with_key_messages(self) -> None:
        bc = BeliefChange(
            fact="city",
            old_value="NYC",
            new_value="SF",
            query_topic="Where?",
            changed_around_msg=100,
            key_messages=[98, 99, 100],
        )
        assert bc.key_messages == [98, 99, 100]

    def test_missing_required_raises(self) -> None:
        with pytest.raises(ValidationError):
            BeliefChange(old_value="a", new_value="b", query_topic="q", changed_around_msg=1)  # type: ignore[call-arg]

    def test_roundtrip(self) -> None:
        bc = BeliefChange(
            fact="f",
            old_value="old",
            new_value="new",
            query_topic="q",
            changed_around_msg=5,
            key_messages=[4, 5],
        )
        assert_roundtrip(BeliefChange, bc)

    def test_json_roundtrip(self) -> None:
        bc = BeliefChange(
            fact="f",
            old_value="o",
            new_value="n",
            query_topic="q",
            changed_around_msg=1,
        )
        assert_json_roundtrip(BeliefChange, bc)


# ===================================================================
# ConflictScenario
# ===================================================================


class TestConflictScenario:
    def test_create_conflict(self) -> None:
        cs = ConflictScenario(
            conflict_id="c-001",
            topic="favorite_food",
            conflicting_statements=["I love pizza", "I hate pizza"],
            correct_resolution="I hate pizza",
            resolution_type="recency",
            introduced_at_messages=[10, 25],
        )
        assert cs.resolution_type == "recency"
        assert len(cs.conflicting_statements) == 2

    def test_all_resolution_types(self) -> None:
        for rt in ("recency", "source_authority", "explicit_correction"):
            cs = ConflictScenario(
                conflict_id="c",
                topic="t",
                conflicting_statements=["a", "b"],
                correct_resolution="b",
                resolution_type=rt,  # type: ignore[arg-type]
                introduced_at_messages=[1],
            )
            assert cs.resolution_type == rt

    def test_invalid_resolution_type_raises(self) -> None:
        with pytest.raises(ValidationError):
            ConflictScenario(
                conflict_id="c-002",
                topic="test",
                conflicting_statements=["a", "b"],
                correct_resolution="b",
                resolution_type="random",  # type: ignore[arg-type]
                introduced_at_messages=[1],
            )

    def test_missing_required_raises(self) -> None:
        with pytest.raises(ValidationError):
            ConflictScenario(
                topic="t",
                conflicting_statements=["a"],
                correct_resolution="a",
                resolution_type="recency",
                introduced_at_messages=[1],
            )  # type: ignore[call-arg]

    def test_roundtrip(self) -> None:
        cs = ConflictScenario(
            conflict_id="c",
            topic="t",
            conflicting_statements=["x", "y"],
            correct_resolution="y",
            resolution_type="explicit_correction",
            introduced_at_messages=[1, 2],
        )
        assert_roundtrip(ConflictScenario, cs)


# ===================================================================
# TemporalFact
# ===================================================================


class TestTemporalFact:
    def test_create_temporal_fact(self) -> None:
        tf = TemporalFact(
            fact_id="tf-001",
            description="Job at Acme Corp",
            value="Software Engineer at Acme Corp",
            valid_from="2020-01-01",
            valid_until="2023-06-30",
            query_topic="Where does the user work?",
            should_be_current=False,
        )
        assert tf.valid_from == "2020-01-01"
        assert not tf.should_be_current

    def test_temporal_fact_open_ended(self) -> None:
        tf = TemporalFact(
            fact_id="tf-002",
            description="Current job",
            value="CTO at Startup",
            query_topic="Current role?",
            should_be_current=True,
        )
        assert tf.valid_from is None
        assert tf.valid_until is None
        assert tf.should_be_current

    def test_default_validity_is_none(self) -> None:
        tf = TemporalFact(
            fact_id="x",
            description="d",
            value="v",
            query_topic="q",
            should_be_current=True,
        )
        assert tf.valid_from is None
        assert tf.valid_until is None

    def test_roundtrip(self) -> None:
        tf = TemporalFact(
            fact_id="tf-rt",
            description="d",
            value="v",
            valid_from="2025-01-01",
            valid_until="2026-01-01",
            query_topic="q",
            should_be_current=False,
        )
        assert_roundtrip(TemporalFact, tf)

    def test_json_roundtrip(self) -> None:
        tf = TemporalFact(
            fact_id="tf-json",
            description="d",
            value="v",
            query_topic="q",
            should_be_current=True,
        )
        assert_json_roundtrip(TemporalFact, tf)


# ===================================================================
# QueryRelevancePair
# ===================================================================


class TestNoiseExample:
    def test_create(self) -> None:
        ne = NoiseExample(text="How's the weather?", reason="Small talk")
        assert ne.text == "How's the weather?"

    def test_missing_reason_raises(self) -> None:
        with pytest.raises(ValidationError):
            NoiseExample(text="hi")  # type: ignore[call-arg]

    def test_roundtrip(self) -> None:
        ne = NoiseExample(text="t", reason="r")
        assert_roundtrip(NoiseExample, ne)


class TestSignalExample:
    def test_create(self) -> None:
        se = SignalExample(text="I just moved to SF", target_fact="city: SF")
        assert se.target_fact == "city: SF"

    def test_missing_target_fact_raises(self) -> None:
        with pytest.raises(ValidationError):
            SignalExample(text="hello")  # type: ignore[call-arg]

    def test_roundtrip(self) -> None:
        se = SignalExample(text="t", target_fact="f")
        assert_roundtrip(SignalExample, se)


# ===================================================================
# GroundTruth
# ===================================================================


class TestGroundTruth:
    def _make_gt(self, **overrides) -> GroundTruth:
        defaults = dict(
            final_profile={
                "occupation": ProfileDimension(
                    dimension_name="occupation",
                    value="engineer",
                    query_topic="What does the user do?",
                ),
            },
            changes=[
                BeliefChange(
                    fact="city",
                    old_value="NYC",
                    new_value="SF",
                    query_topic="Where?",
                    changed_around_msg=10,
                ),
            ],
            noise_examples=[NoiseExample(text="hi", reason="greeting")],
            signal_examples=[SignalExample(text="I'm an engineer", target_fact="occupation")],
            conflicts=[
                ConflictScenario(
                    conflict_id="c1",
                    topic="food",
                    conflicting_statements=["pizza", "sushi"],
                    correct_resolution="sushi",
                    resolution_type="recency",
                    introduced_at_messages=[1, 5],
                ),
            ],
            temporal_facts=[
                TemporalFact(
                    fact_id="tf1",
                    description="d",
                    value="v",
                    query_topic="q",
                    should_be_current=True,
                ),
            ],
            query_relevance_pairs=[
                QueryRelevancePair(
                    query_id="qrp1",
                    query="q",
                    expected_relevant_facts=["a"],
                    expected_irrelevant_facts=["b"],
                ),
            ],
        )
        defaults.update(overrides)
        return GroundTruth(**defaults)

    def test_create_ground_truth_full(self) -> None:
        gt = self._make_gt()
        assert "occupation" in gt.final_profile
        assert len(gt.changes) == 1
        assert len(gt.conflicts) == 1
        assert len(gt.temporal_facts) == 1
        assert len(gt.query_relevance_pairs) == 1
        assert len(gt.noise_examples) == 1
        assert len(gt.signal_examples) == 1

    def test_ground_truth_empty(self) -> None:
        gt = GroundTruth(
            final_profile={},
            changes=[],
            noise_examples=[],
            signal_examples=[],
            conflicts=[],
            temporal_facts=[],
            query_relevance_pairs=[],
        )
        assert len(gt.final_profile) == 0
        assert len(gt.changes) == 0

    def test_ground_truth_roundtrip(self) -> None:
        gt = self._make_gt()
        assert_roundtrip(GroundTruth, gt)

    def test_ground_truth_json_roundtrip(self) -> None:
        gt = self._make_gt()
        assert_json_roundtrip(GroundTruth, gt)

    def test_ground_truth_missing_required_raises(self) -> None:
        with pytest.raises(ValidationError):
            GroundTruth(final_profile={})  # type: ignore[call-arg]

    def test_ground_truth_from_fixture(self, sample_ground_truth: GroundTruth) -> None:
        """Verify the sample_ground_truth fixture is valid and complete."""
        gt = sample_ground_truth
        assert len(gt.final_profile) >= 3
        assert len(gt.changes) >= 2
        assert len(gt.noise_examples) >= 1
        assert len(gt.signal_examples) >= 1
        assert len(gt.conflicts) >= 1
        assert len(gt.temporal_facts) >= 1
        assert len(gt.query_relevance_pairs) >= 1


# ===================================================================
# DatasetMetadata
# ===================================================================


class TestDatasetMetadata:
    def test_create(self) -> None:
        dm = DatasetMetadata(
            dataset_id="ds-001",
            persona_id="persona-alice",
            message_count=150,
            simulated_days=90,
            version="1.0.0",
        )
        assert dm.seed is None
        assert dm.message_count == 150

    def test_default_seed_is_none(self) -> None:
        dm = DatasetMetadata(
            dataset_id="ds",
            persona_id="p",
            message_count=1,
            simulated_days=1,
            version="1.0",
        )
        assert dm.seed is None

    def test_with_seed(self) -> None:
        dm = DatasetMetadata(
            dataset_id="ds-002",
            persona_id="persona-bob",
            message_count=200,
            simulated_days=60,
            version="1.0.0",
            seed=42,
        )
        assert dm.seed == 42

    def test_roundtrip(self) -> None:
        dm = DatasetMetadata(
            dataset_id="ds-rt",
            persona_id="p",
            message_count=10,
            simulated_days=5,
            version="2.0",
            seed=99,
        )
        assert_roundtrip(DatasetMetadata, dm)

    def test_missing_required_raises(self) -> None:
        with pytest.raises(ValidationError):
            DatasetMetadata(persona_id="p", message_count=1, simulated_days=1, version="1")  # type: ignore[call-arg]


# ===================================================================
# ConversationDataset
# ===================================================================


class TestConversationDataset:
    def _make_dataset(self) -> ConversationDataset:
        return ConversationDataset(
            metadata=DatasetMetadata(
                dataset_id="ds-001",
                persona_id="p-001",
                message_count=2,
                simulated_days=1,
                version="1.0.0",
            ),
            messages=[
                Message(
                    message_id=1,
                    role="user",
                    content="I'm Alice.",
                    timestamp="2026-01-01T10:00:00Z",
                    day=1,
                ),
                Message(
                    message_id=2,
                    role="assistant",
                    content="Hi Alice!",
                    timestamp="2026-01-01T10:00:01Z",
                    day=1,
                ),
            ],
            ground_truth=GroundTruth(
                final_profile={
                    "name": ProfileDimension(
                        dimension_name="name",
                        value="Alice",
                        query_topic="name?",
                    ),
                },
                changes=[],
                noise_examples=[],
                signal_examples=[],
                conflicts=[],
                temporal_facts=[],
                query_relevance_pairs=[],
            ),
        )

    def test_create_full_dataset(self) -> None:
        ds = self._make_dataset()
        assert ds.metadata.dataset_id == "ds-001"
        assert len(ds.messages) == 2
        assert "name" in ds.ground_truth.final_profile

    def test_dataset_roundtrip(self) -> None:
        ds = self._make_dataset()
        assert_roundtrip(ConversationDataset, ds)

    def test_dataset_json_roundtrip(self) -> None:
        ds = self._make_dataset()
        assert_json_roundtrip(ConversationDataset, ds)

    def test_empty_messages(self) -> None:
        ds = ConversationDataset(
            metadata=DatasetMetadata(
                dataset_id="ds-empty",
                persona_id="p",
                message_count=0,
                simulated_days=1,
                version="1.0",
            ),
            messages=[],
            ground_truth=GroundTruth(
                final_profile={},
                changes=[],
                noise_examples=[],
                signal_examples=[],
                conflicts=[],
                temporal_facts=[],
                query_relevance_pairs=[],
            ),
        )
        assert len(ds.messages) == 0


# ===================================================================
# JudgmentResult (binary verdict)
# ===================================================================


class TestJudgmentResult:
    def test_create_judgment(self) -> None:
        jr = JudgmentResult(
            check_id="chk-001",
            verdict=Verdict.YES,
            votes=[Verdict.YES, Verdict.YES, Verdict.YES],
            unanimous=True,
            prompt="Does the system know the user's name?",
            raw_responses=["YES", "YES", "YES"],
        )
        assert jr.verdict == Verdict.YES
        assert jr.unanimous
        assert len(jr.votes) == 3

    def test_non_unanimous(self) -> None:
        jr = JudgmentResult(
            check_id="chk-002",
            verdict=Verdict.NO,
            votes=[Verdict.YES, Verdict.NO, Verdict.NO],
            unanimous=False,
            prompt="Does the system know the user's phone?",
            raw_responses=["YES", "NO", "NO"],
        )
        assert not jr.unanimous
        assert jr.verdict == Verdict.NO

    def test_single_vote(self) -> None:
        jr = JudgmentResult(
            check_id="chk-single",
            verdict=Verdict.YES,
            votes=[Verdict.YES],
            unanimous=True,
            prompt="test?",
            raw_responses=["YES"],
        )
        assert len(jr.votes) == 1

    def test_roundtrip(self) -> None:
        jr = JudgmentResult(
            check_id="chk-003",
            verdict=Verdict.YES,
            votes=[Verdict.YES],
            unanimous=True,
            prompt="test?",
            raw_responses=["YES"],
        )
        assert_roundtrip(JudgmentResult, jr)

    def test_json_roundtrip(self) -> None:
        jr = JudgmentResult(
            check_id="chk-json",
            verdict=Verdict.NO,
            votes=[Verdict.NO, Verdict.YES, Verdict.NO],
            unanimous=False,
            prompt="prompt?",
            raw_responses=["NO", "YES", "NO"],
        )
        assert_json_roundtrip(JudgmentResult, jr)

    def test_missing_required_raises(self) -> None:
        with pytest.raises(ValidationError):
            JudgmentResult(
                verdict=Verdict.YES,
                votes=[Verdict.YES],
                unanimous=True,
                prompt="p",
                raw_responses=["YES"],
            )  # type: ignore[call-arg]  # missing check_id


# ===================================================================
# DimensionResult
# ===================================================================


class TestDimensionResult:
    def test_create(self) -> None:
        dr = DimensionResult(
            dimension_name="PAS",
            score=0.85,
            passed_checks=17,
            total_checks=20,
        )
        assert dr.score == 0.85
        assert dr.details == []

    def test_default_details_empty(self) -> None:
        dr = DimensionResult(dimension_name="X", score=0.0, passed_checks=0, total_checks=0)
        assert dr.details == []

    def test_details_independence(self) -> None:
        d1 = DimensionResult(dimension_name="A", score=0.5, passed_checks=1, total_checks=2)
        d2 = DimensionResult(dimension_name="B", score=0.5, passed_checks=1, total_checks=2)
        d1.details.append({"x": 1})
        assert len(d2.details) == 0

    def test_with_details(self) -> None:
        dr = DimensionResult(
            dimension_name="DBU",
            score=1.0,
            passed_checks=5,
            total_checks=5,
            details=[{"check_id": "chk-1", "passed": True}],
        )
        assert len(dr.details) == 1

    def test_roundtrip(self) -> None:
        dr = DimensionResult(
            dimension_name="TC",
            score=0.75,
            passed_checks=3,
            total_checks=4,
            details=[{"check_id": "c1", "passed": True}],
        )
        assert_roundtrip(DimensionResult, dr)


# ===================================================================
# CRIResult
# ===================================================================


class TestCRIResult:
    def _make_cri_result(self, **overrides) -> CRIResult:
        defaults = dict(
            system_name="test-memory",
            cri=0.78,
            pas=0.85,
            dbu=0.70,
            mei=0.80,
            tc=0.75,
            crq=0.65,
            qrp=0.90,
            dimension_weights={
                "PAS": 0.25,
                "DBU": 0.20,
                "MEI": 0.20,
                "TC": 0.15,
                "CRQ": 0.10,
                "QRP": 0.10,
            },
            details={
                "PAS": DimensionResult(
                    dimension_name="PAS",
                    score=0.85,
                    passed_checks=17,
                    total_checks=20,
                ),
            },
        )
        defaults.update(overrides)
        return CRIResult(**defaults)

    def test_create(self) -> None:
        cr = self._make_cri_result()
        assert cr.system_name == "test-memory"
        assert cr.cri == 0.78
        assert "PAS" in cr.details

    def test_dimension_scores_accessible(self) -> None:
        cr = self._make_cri_result()
        assert cr.pas == 0.85
        assert cr.dbu == 0.70
        assert cr.mei == 0.80
        assert cr.tc == 0.75
        assert cr.crq == 0.65
        assert cr.qrp == 0.90

    def test_roundtrip(self) -> None:
        cr = self._make_cri_result()
        assert_roundtrip(CRIResult, cr)

    def test_json_roundtrip(self) -> None:
        cr = self._make_cri_result()
        assert_json_roundtrip(CRIResult, cr)


# ===================================================================
# PerformanceProfile
# ===================================================================


class TestPerformanceProfile:
    def _make_profile(self, **overrides) -> PerformanceProfile:
        defaults = dict(
            ingest_latency_ms=5.2,
            query_latency_avg_ms=120.5,
            query_latency_p95_ms=250.0,
            query_latency_p99_ms=480.0,
            total_facts_stored=150,
            memory_growth_curve=[(10, 5), (50, 30), (100, 75)],
            judge_api_calls=60,
        )
        defaults.update(overrides)
        return PerformanceProfile(**defaults)

    def test_create(self) -> None:
        pp = self._make_profile()
        assert pp.judge_total_cost_estimate is None
        assert len(pp.memory_growth_curve) == 3

    def test_default_cost_is_none(self) -> None:
        pp = self._make_profile()
        assert pp.judge_total_cost_estimate is None

    def test_with_cost(self) -> None:
        pp = self._make_profile(judge_total_cost_estimate=1.50)
        assert pp.judge_total_cost_estimate == 1.50

    def test_empty_growth_curve(self) -> None:
        pp = self._make_profile(memory_growth_curve=[])
        assert pp.memory_growth_curve == []

    def test_roundtrip(self) -> None:
        pp = self._make_profile(judge_total_cost_estimate=2.75)
        assert_roundtrip(PerformanceProfile, pp)


# ===================================================================
# BenchmarkResult (new)
# ===================================================================


class TestBenchmarkResult:
    def _make_result(self) -> BenchmarkResult:
        return BenchmarkResult(
            run_id="run-001",
            adapter_name="test-adapter",
            dataset_id="ds-001",
            started_at="2026-01-01T00:00:00Z",
            completed_at="2026-01-01T01:00:00Z",
            cri_result=CRIResult(
                system_name="test-adapter",
                cri=0.80,
                pas=0.85,
                dbu=0.75,
                mei=0.80,
                tc=0.70,
                crq=0.90,
                qrp=0.85,
                dimension_weights={
                    "PAS": 0.25,
                    "DBU": 0.20,
                    "MEI": 0.20,
                    "TC": 0.15,
                    "CRQ": 0.10,
                    "QRP": 0.10,
                },
                details={},
            ),
            performance_profile=PerformanceProfile(
                ingest_latency_ms=5.0,
                query_latency_avg_ms=100.0,
                query_latency_p95_ms=200.0,
                query_latency_p99_ms=400.0,
                total_facts_stored=50,
                memory_growth_curve=[(10, 5)],
                judge_api_calls=20,
            ),
            judge_log=[],
        )

    def test_create(self) -> None:
        br = self._make_result()
        assert br.run_id == "run-001"
        assert br.adapter_name == "test-adapter"
        assert br.cri_result.cri == 0.80

    def test_roundtrip(self) -> None:
        br = self._make_result()
        assert_roundtrip(BenchmarkResult, br)

    def test_json_roundtrip(self) -> None:
        br = self._make_result()
        assert_json_roundtrip(BenchmarkResult, br)

    def test_with_judge_log(self) -> None:
        br = self._make_result()
        # Mutate to add a log entry
        br_data = br.model_dump()
        br_data["judge_log"] = [
            {
                "check_id": "chk-1",
                "verdict": "YES",
                "votes": ["YES"],
                "unanimous": True,
                "prompt": "test?",
                "raw_responses": ["YES"],
            }
        ]
        restored = BenchmarkResult.model_validate(br_data)
        assert len(restored.judge_log) == 1
        assert restored.judge_log[0].verdict == Verdict.YES


# ===================================================================
# ScoringConfig — weights sum to 1.0
# ===================================================================


class TestScoringConfig:
    def test_defaults(self) -> None:
        sc = ScoringConfig()
        assert sc.dimension_weights["PAS"] == 0.25
        assert sc.dimension_weights["DBU"] == 0.20
        assert sc.dimension_weights["MEI"] == 0.20
        assert sc.dimension_weights["TC"] == 0.15
        assert sc.dimension_weights["CRQ"] == 0.10
        assert sc.dimension_weights["QRP"] == 0.10
        assert len(sc.enabled_dimensions) == 6

    def test_default_weights_sum_to_one(self) -> None:
        sc = ScoringConfig()
        total = sum(sc.dimension_weights.values())
        assert abs(total - 1.0) < 1e-9, f"Default weights sum to {total}, expected 1.0"

    def test_all_default_dimensions_enabled(self) -> None:
        sc = ScoringConfig()
        expected = {"PAS", "DBU", "MEI", "TC", "CRQ", "QRP"}
        assert set(sc.enabled_dimensions) == expected

    def test_custom_weights(self) -> None:
        sc = ScoringConfig(
            dimension_weights={"PAS": 0.50, "DBU": 0.50},
            enabled_dimensions=["PAS", "DBU"],
        )
        assert sc.dimension_weights["PAS"] == 0.50
        assert len(sc.enabled_dimensions) == 2

    def test_custom_weights_sum(self) -> None:
        sc = ScoringConfig(
            dimension_weights={"PAS": 0.40, "DBU": 0.30, "MEI": 0.30},
            enabled_dimensions=["PAS", "DBU", "MEI"],
        )
        total = sum(sc.dimension_weights.values())
        assert abs(total - 1.0) < 1e-9

    def test_default_independence(self) -> None:
        sc1 = ScoringConfig()
        sc2 = ScoringConfig()
        sc1.dimension_weights["PAS"] = 0.99
        assert sc2.dimension_weights["PAS"] == 0.25

    def test_enabled_dimensions_independence(self) -> None:
        sc1 = ScoringConfig()
        sc2 = ScoringConfig()
        sc1.enabled_dimensions.append("EXTRA")
        assert "EXTRA" not in sc2.enabled_dimensions

    def test_roundtrip(self) -> None:
        sc = ScoringConfig()
        assert_roundtrip(ScoringConfig, sc)

    def test_fixture_weights_sum_to_one(self, sample_scoring_config: ScoringConfig) -> None:
        """Verify the sample_scoring_config fixture has weights that sum to 1.0."""
        total = sum(sample_scoring_config.dimension_weights.values())
        assert abs(total - 1.0) < 1e-9


# ===================================================================
# ScoringProfile & ScoringConfig factory methods
# ===================================================================


class TestScoringProfile:
    def test_profile_values(self) -> None:
        assert ScoringProfile.CORE.value == "core"
        assert ScoringProfile.EXTENDED.value == "extended"
        assert ScoringProfile.FULL.value == "full"

    def test_profile_from_string(self) -> None:
        assert ScoringProfile("core") == ScoringProfile.CORE
        assert ScoringProfile("extended") == ScoringProfile.EXTENDED
        assert ScoringProfile("full") == ScoringProfile.FULL

    def test_invalid_profile_raises(self) -> None:
        with pytest.raises(ValueError):
            ScoringProfile("nonexistent")

    def test_profile_is_string(self) -> None:
        assert isinstance(ScoringProfile.CORE, str)


class TestScoringConfigProfiles:
    def test_from_profile_core(self) -> None:
        sc = ScoringConfig.from_profile(ScoringProfile.CORE)
        assert set(sc.enabled_dimensions) == {"PAS", "DBU", "MEI", "TC", "CRQ", "QRP"}
        assert len(sc.enabled_dimensions) == 6
        assert sc.profile == ScoringProfile.CORE
        assert not sc.scale_test
        total = sum(sc.dimension_weights.values())
        assert abs(total - 1.0) < 1e-9

    def test_from_profile_extended(self) -> None:
        sc = ScoringConfig.from_profile(ScoringProfile.EXTENDED)
        assert set(sc.enabled_dimensions) == {
            "PAS",
            "DBU",
            "TC",
            "CRQ",
            "QRP",
            "MEI",
            "SFC",
        }
        assert len(sc.enabled_dimensions) == 7
        assert sc.profile == ScoringProfile.EXTENDED
        assert not sc.scale_test
        total = sum(sc.dimension_weights.values())
        assert abs(total - 1.0) < 1e-9

    def test_from_profile_full(self) -> None:
        sc = ScoringConfig.from_profile(ScoringProfile.FULL)
        assert len(sc.enabled_dimensions) == 7
        assert sc.profile == ScoringProfile.FULL
        assert sc.scale_test  # Full profile enables SSI

    def test_from_profile_full_weights_sum(self) -> None:
        sc = ScoringConfig.from_profile(ScoringProfile.FULL)
        total = sum(sc.dimension_weights.values())
        assert abs(total - 1.0) < 1e-9

    def test_from_dimensions_simple(self) -> None:
        sc = ScoringConfig.from_dimensions(["PAS", "MEI"])
        assert set(sc.enabled_dimensions) == {"PAS", "MEI"}
        total = sum(sc.dimension_weights.values())
        assert abs(total - 1.0) < 1e-9
        assert not sc.scale_test

    def test_from_dimensions_all_seven(self) -> None:
        sc = ScoringConfig.from_dimensions(["PAS", "DBU", "TC", "CRQ", "QRP", "MEI", "SFC"])
        assert len(sc.enabled_dimensions) == 7
        total = sum(sc.dimension_weights.values())
        assert abs(total - 1.0) < 1e-9

    def test_from_dimensions_case_insensitive(self) -> None:
        sc = ScoringConfig.from_dimensions(["pas", "dbu"])
        assert set(sc.enabled_dimensions) == {"PAS", "DBU"}

    def test_from_dimensions_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown dimension"):
            ScoringConfig.from_dimensions(["PAS", "FAKE"])

    def test_from_dimensions_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="At least one dimension"):
            ScoringConfig.from_dimensions([])

    def test_from_dimensions_single(self) -> None:
        sc = ScoringConfig.from_dimensions(["SFC"])
        assert sc.enabled_dimensions == ["SFC"]
        assert abs(sc.dimension_weights["SFC"] - 1.0) < 1e-9

    def test_from_dimensions_weights_normalized(self) -> None:
        """Custom dimension selection should produce normalized weights."""
        sc = ScoringConfig.from_dimensions(["PAS", "DBU", "MEI"])
        total = sum(sc.dimension_weights.values())
        assert abs(total - 1.0) < 1e-9
        # All selected dimensions should have positive weight
        for dim in ["PAS", "DBU", "MEI"]:
            assert sc.dimension_weights[dim] > 0

    def test_default_config_is_core(self) -> None:
        """Default ScoringConfig() should match core profile."""
        default = ScoringConfig()
        core = ScoringConfig.from_profile(ScoringProfile.CORE)
        assert set(default.enabled_dimensions) == set(core.enabled_dimensions)
        assert default.dimension_weights == core.dimension_weights

    def test_profile_roundtrip(self) -> None:
        sc = ScoringConfig.from_profile(ScoringProfile.EXTENDED)
        assert_roundtrip(ScoringConfig, sc)

    def test_from_dimensions_roundtrip(self) -> None:
        sc = ScoringConfig.from_dimensions(["PAS", "MEI", "SFC"])
        assert_roundtrip(ScoringConfig, sc)

    def test_validator_fills_missing_weights(self) -> None:
        """If enabled_dimensions has entries not in dimension_weights, validator adds them."""
        sc = ScoringConfig(
            dimension_weights={"PAS": 1.0},
            enabled_dimensions=["PAS", "DBU"],
        )
        assert "DBU" in sc.dimension_weights


# ===================================================================
# GeneratorConfig
# ===================================================================


class TestGeneratorConfig:
    def test_defaults(self) -> None:
        gc = GeneratorConfig()
        assert gc.llm_model == "claude-3-5-sonnet-20241022"
        assert gc.seed is None
        assert gc.simulated_days == 90
        assert gc.messages_per_day_range == (5, 15)

    def test_custom(self) -> None:
        gc = GeneratorConfig(
            llm_model="gpt-4",
            seed=42,
            simulated_days=30,
            messages_per_day_range=(2, 8),
        )
        assert gc.llm_model == "gpt-4"
        assert gc.seed == 42
        assert gc.messages_per_day_range == (2, 8)

    def test_default_seed_is_none(self) -> None:
        gc = GeneratorConfig()
        assert gc.seed is None

    def test_roundtrip(self) -> None:
        gc = GeneratorConfig(seed=7, simulated_days=30)
        assert_roundtrip(GeneratorConfig, gc)


# ===================================================================
# Legacy models
# ===================================================================
