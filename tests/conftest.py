"""Shared test fixtures for CRI Benchmark tests.

Provides reusable fixtures for:
- sample_messages: 12 messages spanning a multi-day conversation
- sample_ground_truth: GroundTruth with all annotation types populated
- mock_judge: Configurable mock that replaces the real LLM Judge
- sample_scoring_config: ScoringConfig with default weights

"""

from __future__ import annotations

from typing import Any

import pytest

from cri.models import (
    BeliefChange,
    ConflictScenario,
    GroundTruth,
    JudgmentResult,
    Message,
    NoiseExample,
    ProfileDimension,
    QueryRelevancePair,
    ScoringConfig,
    SignalExample,
    TemporalFact,
    Verdict,
)

# ---------------------------------------------------------------------------
# New-style fixtures (specified by task input)
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_messages() -> list[Message]:
    """Create a list of 12 sample messages spanning multiple days and sessions.

    Includes both user and assistant messages with varying optional fields.
    """
    return [
        Message(
            message_id=1,
            role="user",
            content="Hi, I'm Alice. I'm a 30-year-old software engineer in New York.",
            timestamp="2026-01-01T10:00:00Z",
            session_id="sess-001",
            day=1,
        ),
        Message(
            message_id=2,
            role="assistant",
            content="Nice to meet you, Alice! Software engineering in NYC sounds exciting.",
            timestamp="2026-01-01T10:00:30Z",
            session_id="sess-001",
            day=1,
        ),
        Message(
            message_id=3,
            role="user",
            content="I love hiking and photography. I go to the Catskills every weekend.",
            timestamp="2026-01-01T10:02:00Z",
            session_id="sess-001",
            day=1,
        ),
        Message(
            message_id=4,
            role="assistant",
            content="The Catskills are beautiful! Do you prefer landscape or nature photography?",
            timestamp="2026-01-01T10:02:30Z",
            session_id="sess-001",
            day=1,
        ),
        Message(
            message_id=5,
            role="user",
            content="Mostly landscape. I just got a new Fujifilm X-T5 camera.",
            timestamp="2026-01-02T14:00:00Z",
            session_id="sess-002",
            day=2,
        ),
        Message(
            message_id=6,
            role="assistant",
            content="Great choice! The X-T5 is excellent for landscapes.",
            timestamp="2026-01-02T14:00:20Z",
            session_id="sess-002",
            day=2,
        ),
        Message(
            message_id=7,
            role="user",
            content="By the way, I recently switched from iOS to Android. Using a Pixel 8 now.",
            timestamp="2026-01-05T09:30:00Z",
            session_id="sess-003",
            day=5,
        ),
        Message(
            message_id=8,
            role="assistant",
            content="How are you finding the switch to Android?",
            timestamp="2026-01-05T09:30:15Z",
            session_id="sess-003",
            day=5,
        ),
        Message(
            message_id=9,
            role="user",
            content="Actually I love it. The Pixel camera is amazing for my photography hobby.",
            timestamp="2026-01-05T09:31:00Z",
            session_id="sess-003",
            day=5,
        ),
        Message(
            message_id=10,
            role="user",
            content="Oh, I should mention — I just got promoted to Senior Engineer!",
            timestamp="2026-01-10T11:00:00Z",
            session_id="sess-004",
            day=10,
        ),
        Message(
            message_id=11,
            role="assistant",
            content="Congratulations on the promotion, Alice! Well deserved.",
            timestamp="2026-01-10T11:00:10Z",
            session_id="sess-004",
            day=10,
        ),
        Message(
            message_id=12,
            role="user",
            content="Thanks! Also, I no longer go hiking every weekend — moved to biking instead.",
            timestamp="2026-01-15T16:00:00Z",
            session_id="sess-005",
            day=15,
        ),
    ]


@pytest.fixture
def sample_ground_truth() -> GroundTruth:
    """Create a fully populated GroundTruth with all annotation types.

    Includes: final_profile, changes, noise_examples, signal_examples,
    conflicts, temporal_facts, and query_relevance_pairs.
    """
    return GroundTruth(
        final_profile={
            "occupation": ProfileDimension(
                dimension_name="occupation",
                value="Senior Software Engineer",
                query_topic="What does the user do for work?",
                category="demographics",
            ),
            "location": ProfileDimension(
                dimension_name="location",
                value="New York",
                query_topic="Where does the user live?",
                category="demographics",
            ),
            "age": ProfileDimension(
                dimension_name="age",
                value="30",
                query_topic="How old is the user?",
                category="demographics",
            ),
            "hobbies": ProfileDimension(
                dimension_name="hobbies",
                value=["biking", "photography"],
                query_topic="What are the user's hobbies?",
                category="interests",
            ),
            "phone_os": ProfileDimension(
                dimension_name="phone_os",
                value="Android",
                query_topic="What phone OS does the user use?",
                category="preferences",
            ),
        },
        changes=[
            BeliefChange(
                fact="phone_os",
                old_value="iOS",
                new_value="Android (Pixel 8)",
                query_topic="What phone does the user use?",
                changed_around_msg=7,
                key_messages=[7, 9],
            ),
            BeliefChange(
                fact="occupation_level",
                old_value="Software Engineer",
                new_value="Senior Software Engineer",
                query_topic="What is the user's job title?",
                changed_around_msg=10,
                key_messages=[10],
            ),
            BeliefChange(
                fact="primary_outdoor_activity",
                old_value="hiking",
                new_value="biking",
                query_topic="What outdoor activity does the user prefer?",
                changed_around_msg=12,
                key_messages=[12],
            ),
        ],
        noise_examples=[
            NoiseExample(
                text="How are you doing today?",
                reason="Generic greeting with no factual content",
            ),
            NoiseExample(
                text="That's interesting, tell me more.",
                reason="Conversational filler with no user-specific info",
            ),
        ],
        signal_examples=[
            SignalExample(
                text="I'm a 30-year-old software engineer in New York.",
                target_fact="occupation: Software Engineer; age: 30; location: New York",
            ),
            SignalExample(
                text="I recently switched from iOS to Android.",
                target_fact="phone_os: Android (updated from iOS)",
            ),
            SignalExample(
                text="I just got promoted to Senior Engineer!",
                target_fact="occupation_level: Senior Engineer",
            ),
        ],
        conflicts=[
            ConflictScenario(
                conflict_id="conflict-hiking-biking",
                topic="primary_outdoor_activity",
                conflicting_statements=[
                    "I go to the Catskills every weekend for hiking.",
                    "I no longer go hiking every weekend — moved to biking instead.",
                ],
                correct_resolution="User now prefers biking over hiking.",
                resolution_type="recency",
                introduced_at_messages=[3, 12],
            ),
            ConflictScenario(
                conflict_id="conflict-phone-os",
                topic="phone_os",
                conflicting_statements=[
                    "Implied iOS user (pre-switch context).",
                    "I recently switched from iOS to Android. Using a Pixel 8 now.",
                ],
                correct_resolution="User now uses Android (Pixel 8).",
                resolution_type="explicit_correction",
                introduced_at_messages=[1, 7],
            ),
        ],
        temporal_facts=[
            TemporalFact(
                fact_id="tf-job-engineer",
                description="Software Engineer role",
                value="Software Engineer",
                valid_from="2026-01-01",
                valid_until="2026-01-10",
                query_topic="What was the user's previous job title?",
                should_be_current=False,
            ),
            TemporalFact(
                fact_id="tf-job-senior",
                description="Senior Software Engineer role",
                value="Senior Software Engineer",
                valid_from="2026-01-10",
                valid_until=None,
                query_topic="What is the user's current job title?",
                should_be_current=True,
            ),
            TemporalFact(
                fact_id="tf-hiking-hobby",
                description="Hiking as primary outdoor hobby",
                value="hiking",
                valid_from="2026-01-01",
                valid_until="2026-01-15",
                query_topic="Did the user used to hike?",
                should_be_current=False,
            ),
        ],
        query_relevance_pairs=[
            QueryRelevancePair(
                query_id="qrp-occupation",
                query="What does the user do for a living?",
                expected_relevant_facts=[
                    "Senior Software Engineer",
                    "promoted from Software Engineer",
                ],
                expected_irrelevant_facts=[
                    "uses Android phone",
                    "likes photography",
                ],
            ),
            QueryRelevancePair(
                query_id="qrp-hobbies",
                query="What are the user's current hobbies?",
                expected_relevant_facts=[
                    "biking",
                    "photography",
                ],
                expected_irrelevant_facts=[
                    "software engineer",
                    "New York",
                ],
            ),
        ],
    )


@pytest.fixture
def sample_scoring_config() -> ScoringConfig:
    """Create a ScoringConfig with the standard default weights.

    Default weights sum to 1.0:
    PAS=0.25, DBU=0.20, MEI=0.20, TC=0.15, CRQ=0.10, QRP=0.10
    """
    return ScoringConfig()


class MockJudge:
    """A configurable mock judge that returns predetermined verdicts.

    Replaces the real LLM-based Judge for deterministic testing.

    Usage in tests::

        judge = MockJudge(default_verdict=Verdict.YES)
        result = await judge.evaluate("check-1", "Does the user like hiking?")
        assert result.verdict == Verdict.YES

        # Override for specific check IDs:
        judge = MockJudge(
            default_verdict=Verdict.YES,
            overrides={"check-fail": Verdict.NO},
        )
    """

    def __init__(
        self,
        default_verdict: Verdict = Verdict.YES,
        overrides: dict[str, Verdict] | None = None,
        num_votes: int = 3,
    ) -> None:
        self.default_verdict = default_verdict
        self.overrides = overrides or {}
        self.num_votes = num_votes
        self.call_log: list[dict[str, Any]] = []

    def _get_verdict(self, check_id: str) -> Verdict:
        return self.overrides.get(check_id, self.default_verdict)

    async def evaluate(
        self,
        check_id: str,
        prompt: str,
    ) -> JudgmentResult:
        """Produce a deterministic JudgmentResult without any LLM calls."""
        verdict = self._get_verdict(check_id)
        votes = [verdict] * self.num_votes

        self.call_log.append({"check_id": check_id, "prompt": prompt, "verdict": verdict})

        return JudgmentResult(
            check_id=check_id,
            verdict=verdict,
            votes=votes,
            unanimous=True,
            prompt=prompt,
            raw_responses=[verdict.value] * self.num_votes,
        )


@pytest.fixture
def mock_judge() -> MockJudge:
    """Create a mock judge that returns YES verdicts by default.

    To customize per test::

        def test_something(mock_judge):
            mock_judge.default_verdict = Verdict.NO
            mock_judge.overrides["special-check"] = Verdict.YES
    """
    return MockJudge(default_verdict=Verdict.YES)
