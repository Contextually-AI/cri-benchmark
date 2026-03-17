"""Tests for the BinaryJudge module.

Test coverage:
- BinaryJudge: constructor defaults, vote parsing, majority voting,
  error handling/retries, log recording, log export, LLM call parameters
- Edge cases: non-binary responses, empty strings, mixed errors
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from cri.judge import (
    BINARY_JUDGE_SYSTEM_PROMPT,
    BinaryJudge,
)
from cri.models import (
    JudgmentResult,
    Verdict,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_llm(content: str = "YES") -> MagicMock:
    """Create a mock BaseChatModel that returns a fixed response."""
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(content=content)
    return mock_llm


def _make_mock_llm_factory(content: str = "YES"):
    """Return a factory that creates a mock LLM returning a fixed response."""

    def factory(temperature: float = 0.0, max_tokens: int = 10) -> MagicMock:
        return _make_mock_llm(content)

    return factory


def _make_mock_llm_factory_sequence(contents: list[str]):
    """Return a factory that creates a mock LLM returning different responses."""

    def factory(temperature: float = 0.0, max_tokens: int = 10) -> MagicMock:
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = [AIMessage(content=c) for c in contents]
        return mock_llm

    return factory


def _make_mock_llm_factory_with_errors(side_effects: list):
    """Return a factory that creates a mock LLM with custom side effects.

    Each element can be a string (returned as AIMessage) or an Exception.
    """

    def factory(temperature: float = 0.0, max_tokens: int = 10) -> MagicMock:
        mock_llm = MagicMock()
        effects = []
        for effect in side_effects:
            if isinstance(effect, Exception):
                effects.append(effect)
            else:
                effects.append(AIMessage(content=effect))
        mock_llm.invoke.side_effect = effects
        return mock_llm

    return factory


class TestBinaryJudgeInit:
    """Test BinaryJudge constructor defaults."""

    def test_default_num_runs(self) -> None:
        j = BinaryJudge(llm_factory=_make_mock_llm_factory())
        assert j.num_runs == 3

    def test_default_temperature(self) -> None:
        j = BinaryJudge(llm_factory=_make_mock_llm_factory())
        assert j.temperature == 0.0

    def test_default_max_tokens(self) -> None:
        j = BinaryJudge(llm_factory=_make_mock_llm_factory())
        assert j.max_tokens == 10

    def test_custom_params(self) -> None:
        j = BinaryJudge(llm_factory=_make_mock_llm_factory(), num_runs=5, temperature=0.5, max_tokens=20)
        assert j.num_runs == 5
        assert j.temperature == 0.5
        assert j.max_tokens == 20

    def test_empty_log_on_init(self) -> None:
        j = BinaryJudge(llm_factory=_make_mock_llm_factory())
        assert j.get_log() == []

    def test_llm_attribute_exists(self) -> None:
        j = BinaryJudge(llm_factory=_make_mock_llm_factory())
        assert j.llm is not None

    def test_factory_called_with_temperature_and_max_tokens(self) -> None:
        factory = MagicMock(return_value=_make_mock_llm())
        BinaryJudge(llm_factory=factory, temperature=0.7, max_tokens=15)
        factory.assert_called_once_with(0.7, 15)


# ---------------------------------------------------------------------------
# Vote parsing
# ---------------------------------------------------------------------------


class TestParseVote:
    """Test BinaryJudge._parse_vote static method."""

    def test_yes_uppercase(self) -> None:
        assert BinaryJudge._parse_vote("YES") is Verdict.YES

    def test_yes_lowercase(self) -> None:
        assert BinaryJudge._parse_vote("yes") is Verdict.YES

    def test_yes_mixed_case(self) -> None:
        assert BinaryJudge._parse_vote("Yes") is Verdict.YES

    def test_no_uppercase(self) -> None:
        assert BinaryJudge._parse_vote("NO") is Verdict.NO

    def test_no_lowercase(self) -> None:
        assert BinaryJudge._parse_vote("no") is Verdict.NO

    def test_no_mixed_case(self) -> None:
        assert BinaryJudge._parse_vote("No") is Verdict.NO

    def test_yes_with_extra_text(self) -> None:
        assert BinaryJudge._parse_vote("YES, the answer is correct") is Verdict.YES

    def test_no_with_extra_text(self) -> None:
        assert BinaryJudge._parse_vote("NO, incomplete response") is Verdict.NO

    def test_multiline_uses_first_line(self) -> None:
        assert BinaryJudge._parse_vote("YES\nSome reasoning") is Verdict.YES

    def test_multiline_no(self) -> None:
        assert BinaryJudge._parse_vote("NO\nBecause reasons") is Verdict.NO

    def test_non_binary_defaults_to_no(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger="cri.judge"):
            result = BinaryJudge._parse_vote("Maybe")
        assert result is Verdict.NO
        assert "non-binary response" in caplog.text

    def test_empty_string_defaults_to_no(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger="cri.judge"):
            result = BinaryJudge._parse_vote("")
        assert result is Verdict.NO

    def test_whitespace_only_defaults_to_no(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger="cri.judge"):
            result = BinaryJudge._parse_vote("   \n  \n  ")
        assert result is Verdict.NO

    def test_leading_blank_lines_skipped(self) -> None:
        assert BinaryJudge._parse_vote("\n\n  YES\n") is Verdict.YES

    def test_number_defaults_to_no(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger="cri.judge"):
            result = BinaryJudge._parse_vote("42")
        assert result is Verdict.NO

    def test_yes_embedded_in_word(self) -> None:
        # "YES" is contained in "YESTERDAY"
        assert BinaryJudge._parse_vote("YESTERDAY") is Verdict.YES

    def test_no_embedded_in_word(self) -> None:
        # "NO" is contained in "NOTE" or "NOTHING"
        assert BinaryJudge._parse_vote("NOTHING") is Verdict.NO


# ---------------------------------------------------------------------------
# Majority voting
# ---------------------------------------------------------------------------


class TestMajorityVote:
    """Test majority-vote logic in judge()."""

    @pytest.fixture
    def judge(self) -> BinaryJudge:
        return BinaryJudge(llm_factory=_make_mock_llm_factory("YES"), num_runs=3)

    def test_2_yes_1_no_yields_yes(self) -> None:
        j = BinaryJudge(
            llm_factory=_make_mock_llm_factory_sequence(["YES", "YES", "NO"]),
            num_runs=3,
        )
        result = j.judge("chk-1", "Is sky blue?")
        assert result.verdict is Verdict.YES
        assert result.votes == [Verdict.YES, Verdict.YES, Verdict.NO]
        assert result.unanimous is False

    def test_1_yes_2_no_yields_no(self) -> None:
        j = BinaryJudge(
            llm_factory=_make_mock_llm_factory_sequence(["YES", "NO", "NO"]),
            num_runs=3,
        )
        result = j.judge("chk-2", "Is sky green?")
        assert result.verdict is Verdict.NO
        assert result.votes == [Verdict.YES, Verdict.NO, Verdict.NO]
        assert result.unanimous is False

    def test_unanimous_yes(self) -> None:
        j = BinaryJudge(llm_factory=_make_mock_llm_factory("YES"), num_runs=3)
        result = j.judge("chk-3", "Test?")
        assert result.verdict is Verdict.YES
        assert result.unanimous is True
        assert all(v is Verdict.YES for v in result.votes)

    def test_unanimous_no(self) -> None:
        j = BinaryJudge(llm_factory=_make_mock_llm_factory("NO"), num_runs=3)
        result = j.judge("chk-4", "Test?")
        assert result.verdict is Verdict.NO
        assert result.unanimous is True
        assert all(v is Verdict.NO for v in result.votes)

    def test_all_non_binary_defaults_to_no(self) -> None:
        j = BinaryJudge(llm_factory=_make_mock_llm_factory("MAYBE"), num_runs=3)
        result = j.judge("chk-5", "Unclear?")
        assert result.verdict is Verdict.NO
        assert all(v is Verdict.NO for v in result.votes)

    def test_3_yes_0_no_is_unanimous_yes(self) -> None:
        j = BinaryJudge(llm_factory=_make_mock_llm_factory("YES"), num_runs=3)
        result = j.judge("u-yes", "p")
        assert result.verdict is Verdict.YES
        assert result.unanimous is True
        assert len(result.votes) == 3

    def test_0_yes_3_no_is_unanimous_no(self) -> None:
        j = BinaryJudge(llm_factory=_make_mock_llm_factory("NO"), num_runs=3)
        result = j.judge("u-no", "p")
        assert result.verdict is Verdict.NO
        assert result.unanimous is True

    def test_single_run_yes(self) -> None:
        j = BinaryJudge(llm_factory=_make_mock_llm_factory("YES"), num_runs=1)
        result = j.judge("single", "p")
        assert result.verdict is Verdict.YES
        assert len(result.votes) == 1

    def test_five_runs_majority(self) -> None:
        j = BinaryJudge(
            llm_factory=_make_mock_llm_factory_sequence(["YES", "YES", "YES", "NO", "NO"]),
            num_runs=5,
        )
        result = j.judge("five", "p")
        assert result.verdict is Verdict.YES
        assert len(result.votes) == 5


# ---------------------------------------------------------------------------
# JudgmentResult fields
# ---------------------------------------------------------------------------


class TestJudgmentResultFields:
    """Verify the returned JudgmentResult has correct metadata."""

    def test_check_id_preserved(self) -> None:
        j = BinaryJudge(llm_factory=_make_mock_llm_factory("YES"), num_runs=3)
        result = j.judge("my-check-42", "prompt text")
        assert result.check_id == "my-check-42"

    def test_prompt_preserved(self) -> None:
        j = BinaryJudge(llm_factory=_make_mock_llm_factory("YES"), num_runs=3)
        result = j.judge("c1", "the evaluation prompt")
        assert result.prompt == "the evaluation prompt"

    def test_raw_responses_collected(self) -> None:
        j = BinaryJudge(
            llm_factory=_make_mock_llm_factory_sequence(["YES", "NO", "YES"]),
            num_runs=3,
        )
        result = j.judge("c2", "p")
        assert result.raw_responses == ["YES", "NO", "YES"]

    def test_result_is_judgment_result_type(self) -> None:
        j = BinaryJudge(llm_factory=_make_mock_llm_factory("NO"), num_runs=3)
        result = j.judge("c3", "p")
        assert isinstance(result, JudgmentResult)

    def test_votes_count_matches_num_runs(self) -> None:
        j = BinaryJudge(llm_factory=_make_mock_llm_factory("YES"), num_runs=3)
        result = j.judge("count", "p")
        assert len(result.votes) == 3
        assert len(result.raw_responses) == 3


# ---------------------------------------------------------------------------
# Error handling / retries
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """Test API error retry and default behavior."""

    def test_retry_once_on_error_then_recover(self) -> None:
        j = BinaryJudge(
            llm_factory=_make_mock_llm_factory_with_errors(
                [
                    # Run 1: fail then succeed on retry
                    Exception("API timeout"),
                    "YES",
                    # Run 2: success
                    "YES",
                    # Run 3: success
                    "NO",
                ]
            ),
            num_runs=3,
        )
        result = j.judge("err-1", "test")
        assert result.verdict is Verdict.YES

    def test_both_attempts_fail_defaults_to_no(self) -> None:
        j = BinaryJudge(
            llm_factory=_make_mock_llm_factory_with_errors(
                [
                    Exception("fail 1"),
                    Exception("fail 2"),
                ]
            ),
            num_runs=1,
        )
        result = j.judge("err-2", "test")
        assert result.verdict is Verdict.NO
        assert result.raw_responses == [""]

    def test_mixed_errors_and_successes(self) -> None:
        j = BinaryJudge(
            llm_factory=_make_mock_llm_factory_with_errors(
                [
                    # Run 1: both attempts fail → ""
                    Exception("fail"),
                    Exception("fail"),
                    # Run 2: success
                    "YES",
                    # Run 3: success
                    "YES",
                ]
            ),
            num_runs=3,
        )
        result = j.judge("err-3", "test")
        # Votes: NO (from empty), YES, YES → majority YES
        assert result.verdict is Verdict.YES

    def test_all_runs_fail_results_in_no(self) -> None:
        def factory(temperature: float = 0.0, max_tokens: int = 10) -> MagicMock:
            mock_llm = MagicMock()
            mock_llm.invoke.side_effect = Exception("always fail")
            return mock_llm

        j = BinaryJudge(llm_factory=factory, num_runs=3)
        result = j.judge("all-fail", "test")
        assert result.verdict is Verdict.NO
        assert all(v is Verdict.NO for v in result.votes)
        assert all(r == "" for r in result.raw_responses)

    def test_none_content_treated_as_empty(self) -> None:
        """If LLM returns None content, it should be treated as empty string."""

        def factory(temperature: float = 0.0, max_tokens: int = 10) -> MagicMock:
            mock_llm = MagicMock()
            mock_response = MagicMock()
            mock_response.content = None
            mock_llm.invoke.return_value = mock_response
            return mock_llm

        j = BinaryJudge(llm_factory=factory, num_runs=1)
        result = j.judge("none-content", "p")
        assert result.verdict is Verdict.NO


# ---------------------------------------------------------------------------
# Log recording
# ---------------------------------------------------------------------------


class TestGetLog:
    """Test get_log() returns accumulated results."""

    def test_log_accumulates(self) -> None:
        j = BinaryJudge(llm_factory=_make_mock_llm_factory("YES"), num_runs=3)
        j.judge("a", "p1")
        j.judge("b", "p2")
        j.judge("c", "p3")
        log = j.get_log()
        assert len(log) == 3
        assert [r.check_id for r in log] == ["a", "b", "c"]

    def test_log_returns_copy(self) -> None:
        j = BinaryJudge(llm_factory=_make_mock_llm_factory("YES"), num_runs=3)
        j.judge("x", "p")
        log1 = j.get_log()
        log2 = j.get_log()
        assert log1 is not log2
        assert log1 == log2

    def test_empty_log_initially(self) -> None:
        j = BinaryJudge(llm_factory=_make_mock_llm_factory())
        assert j.get_log() == []

    def test_log_preserves_verdicts(self) -> None:
        j = BinaryJudge(
            llm_factory=_make_mock_llm_factory_sequence(
                [
                    "YES",
                    "YES",
                    "YES",
                    "NO",
                    "NO",
                    "NO",
                ]
            ),
            num_runs=3,
        )
        j.judge("yes-check", "p1")
        j.judge("no-check", "p2")
        log = j.get_log()
        assert log[0].verdict is Verdict.YES
        assert log[1].verdict is Verdict.NO

    def test_log_not_affected_by_external_mutation(self) -> None:
        j = BinaryJudge(llm_factory=_make_mock_llm_factory("YES"), num_runs=3)
        j.judge("m", "p")
        log = j.get_log()
        log.clear()  # mutate the returned copy
        assert len(j.get_log()) == 1  # internal log unaffected


# ---------------------------------------------------------------------------
# Export log
# ---------------------------------------------------------------------------


class TestExportLog:
    """Test export_log() writes valid JSON."""

    def test_export_creates_json_file(self, tmp_path: Path) -> None:
        j = BinaryJudge(llm_factory=_make_mock_llm_factory("YES"), num_runs=3)
        j.judge("e1", "prompt1")
        j.judge("e2", "prompt2")
        out = tmp_path / "log.json"
        j.export_log(out)
        assert out.exists()
        data = json.loads(out.read_text())
        assert isinstance(data, list)
        assert len(data) == 2
        assert data[0]["check_id"] == "e1"
        assert data[1]["check_id"] == "e2"

    def test_export_json_structure(self, tmp_path: Path) -> None:
        j = BinaryJudge(llm_factory=_make_mock_llm_factory("NO"), num_runs=3)
        j.judge("s1", "some prompt")
        out = tmp_path / "struct.json"
        j.export_log(out)
        data = json.loads(out.read_text())
        record = data[0]
        assert "check_id" in record
        assert "verdict" in record
        assert "votes" in record
        assert "unanimous" in record
        assert "prompt" in record
        assert "raw_responses" in record

    def test_export_empty_log(self, tmp_path: Path) -> None:
        j = BinaryJudge(llm_factory=_make_mock_llm_factory())
        out = tmp_path / "empty.json"
        j.export_log(out)
        data = json.loads(out.read_text())
        assert data == []

    def test_export_is_valid_json(self, tmp_path: Path) -> None:
        j = BinaryJudge(llm_factory=_make_mock_llm_factory("YES"), num_runs=1)
        j.judge("j1", 'prompt with "quotes" and\nnewlines')
        out = tmp_path / "valid.json"
        j.export_log(out)
        # Should not raise
        data = json.loads(out.read_text())
        assert len(data) == 1

    def test_export_overwrites_existing_file(self, tmp_path: Path) -> None:
        j = BinaryJudge(llm_factory=_make_mock_llm_factory("YES"), num_runs=1)
        out = tmp_path / "overwrite.json"
        out.write_text("old content")
        j.judge("new", "p")
        j.export_log(out)
        data = json.loads(out.read_text())
        assert len(data) == 1
        assert data[0]["check_id"] == "new"


# ---------------------------------------------------------------------------
# LLM call parameters
# ---------------------------------------------------------------------------


class TestLLMCallParameters:
    """Verify BaseChatModel.invoke is called with correct params."""

    def test_uses_system_prompt(self) -> None:
        j = BinaryJudge(llm_factory=_make_mock_llm_factory("YES"), num_runs=1)
        j.judge("p1", "user prompt")
        messages = j.llm.invoke.call_args[0][0]
        assert isinstance(messages[0], SystemMessage)
        assert messages[0].content == BINARY_JUDGE_SYSTEM_PROMPT

    def test_uses_user_prompt(self) -> None:
        j = BinaryJudge(llm_factory=_make_mock_llm_factory("YES"), num_runs=1)
        j.judge("p1", "my user prompt")
        messages = j.llm.invoke.call_args[0][0]
        assert isinstance(messages[1], HumanMessage)
        assert messages[1].content == "my user prompt"

    def test_temperature_and_max_tokens_passed_to_factory(self) -> None:
        factory = MagicMock(return_value=_make_mock_llm())
        BinaryJudge(llm_factory=factory, num_runs=1, temperature=0.5, max_tokens=20)
        factory.assert_called_once_with(0.5, 20)

    def test_called_num_runs_times(self) -> None:
        j = BinaryJudge(llm_factory=_make_mock_llm_factory("YES"), num_runs=5)
        j.judge("p1", "prompt")
        assert j.llm.invoke.call_count == 5

    def test_messages_format(self) -> None:
        j = BinaryJudge(llm_factory=_make_mock_llm_factory("YES"), num_runs=1)
        j.judge("fmt", "test prompt")
        messages = j.llm.invoke.call_args[0][0]
        assert len(messages) == 2
        assert isinstance(messages[0], SystemMessage)
        assert isinstance(messages[1], HumanMessage)
