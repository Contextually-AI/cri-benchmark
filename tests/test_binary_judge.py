"""Tests for the BinaryJudge class."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from cri.judge import BINARY_JUDGE_SYSTEM_PROMPT, BinaryJudge
from cri.models import JudgmentResult, Verdict

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_llm(content: str = "YES") -> MagicMock:
    """Create a mock BaseChatModel that returns a fixed response."""
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(content=content)
    mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content=content))
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
        mock_llm.ainvoke = AsyncMock(side_effect=[AIMessage(content=c) for c in contents])
        return mock_llm

    return factory


def _make_mock_llm_factory_with_errors(side_effects: list):
    """Return a factory that creates a mock LLM with custom side effects."""

    def factory(temperature: float = 0.0, max_tokens: int = 10) -> MagicMock:
        mock_llm = MagicMock()
        effects = []
        for effect in side_effects:
            if isinstance(effect, Exception):
                effects.append(effect)
            else:
                effects.append(AIMessage(content=effect))
        mock_llm.invoke.side_effect = list(effects)
        mock_llm.ainvoke = AsyncMock(side_effect=list(effects))
        return mock_llm

    return factory


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def judge() -> BinaryJudge:
    """Return a BinaryJudge with mock LLM and default settings."""
    return BinaryJudge(llm_factory=_make_mock_llm_factory("YES"), num_runs=3)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


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

    def test_num_runs_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="num_runs must be >= 1"):
            BinaryJudge(llm_factory=_make_mock_llm_factory(), num_runs=0)

    def test_num_runs_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="num_runs must be >= 1"):
            BinaryJudge(llm_factory=_make_mock_llm_factory(), num_runs=-1)

    def test_num_runs_one_accepted(self) -> None:
        j = BinaryJudge(llm_factory=_make_mock_llm_factory(), num_runs=1)
        assert j.num_runs == 1

    def test_even_num_runs_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger="cri.judge"):
            BinaryJudge(llm_factory=_make_mock_llm_factory(), num_runs=2)
        assert "even" in caplog.text

    def test_empty_log_on_init(self) -> None:
        j = BinaryJudge(llm_factory=_make_mock_llm_factory())
        assert j.get_log() == []

    def test_llm_attribute_exists(self) -> None:
        j = BinaryJudge(llm_factory=_make_mock_llm_factory())
        assert j.llm is not None

    def test_factory_called_with_temperature_and_max_tokens(self) -> None:
        factory = MagicMock(return_value=_make_mock_llm())
        BinaryJudge(llm_factory=factory, temperature=0.7, max_tokens=15)
        assert factory.call_count == 2
        factory.assert_any_call(0.7, 15)
        factory.assert_any_call(0.7, 200)


# ---------------------------------------------------------------------------
# System prompt constant
# ---------------------------------------------------------------------------


class TestSystemPrompt:
    def test_binary_judge_system_prompt_value(self) -> None:
        assert BINARY_JUDGE_SYSTEM_PROMPT == "You are an evaluation judge. Answer only YES or NO."


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


# ---------------------------------------------------------------------------
# Majority voting
# ---------------------------------------------------------------------------


class TestMajorityVote:
    """Test majority-vote logic in judge()."""

    async def test_majority_yes(self) -> None:
        j = BinaryJudge(
            llm_factory=_make_mock_llm_factory_sequence(["YES", "YES", "NO"]),
            num_runs=3,
        )
        result = await j.judge("chk-1", "Is sky blue?")
        assert result.verdict is Verdict.YES
        assert sum(1 for v in result.votes if v is Verdict.YES) == 2
        assert sum(1 for v in result.votes if v is Verdict.NO) == 1
        assert result.unanimous is False

    async def test_majority_no(self) -> None:
        j = BinaryJudge(
            llm_factory=_make_mock_llm_factory_sequence(["YES", "NO", "NO"]),
            num_runs=3,
        )
        result = await j.judge("chk-2", "Is sky green?")
        assert result.verdict is Verdict.NO
        assert sum(1 for v in result.votes if v is Verdict.YES) == 1
        assert sum(1 for v in result.votes if v is Verdict.NO) == 2
        assert result.unanimous is False

    async def test_unanimous_yes(self) -> None:
        j = BinaryJudge(llm_factory=_make_mock_llm_factory("YES"), num_runs=3)
        result = await j.judge("chk-3", "Test?")
        assert result.verdict is Verdict.YES
        assert result.unanimous is True
        assert all(v is Verdict.YES for v in result.votes)

    async def test_unanimous_no(self) -> None:
        j = BinaryJudge(llm_factory=_make_mock_llm_factory("NO"), num_runs=3)
        result = await j.judge("chk-4", "Test?")
        assert result.verdict is Verdict.NO
        assert result.unanimous is True
        assert all(v is Verdict.NO for v in result.votes)

    async def test_all_non_binary_defaults_to_no(self) -> None:
        j = BinaryJudge(llm_factory=_make_mock_llm_factory("MAYBE"), num_runs=3)
        result = await j.judge("chk-5", "Unclear?")
        assert result.verdict is Verdict.NO
        assert all(v is Verdict.NO for v in result.votes)

    async def test_single_run_yes(self) -> None:
        j = BinaryJudge(llm_factory=_make_mock_llm_factory("YES"), num_runs=1)
        result = await j.judge("chk-single-y", "Test?")
        assert result.verdict is Verdict.YES
        assert len(result.votes) == 1
        assert result.unanimous is True
        assert result.agreement_ratio == 1.0

    async def test_single_run_no(self) -> None:
        j = BinaryJudge(llm_factory=_make_mock_llm_factory("NO"), num_runs=1)
        result = await j.judge("chk-single-n", "Test?")
        assert result.verdict is Verdict.NO
        assert len(result.votes) == 1
        assert result.unanimous is True

    async def test_even_runs_tie_goes_to_no(self) -> None:
        j = BinaryJudge(
            llm_factory=_make_mock_llm_factory_sequence(["YES", "NO"]),
            num_runs=2,
        )
        result = await j.judge("chk-tie", "Tie?")
        assert result.verdict is Verdict.NO
        assert result.unanimous is False

    async def test_five_runs_majority(self) -> None:
        j = BinaryJudge(
            llm_factory=_make_mock_llm_factory_sequence(["YES", "YES", "YES", "NO", "NO"]),
            num_runs=5,
        )
        result = await j.judge("chk-5runs", "Test?")
        assert result.verdict is Verdict.YES
        assert sum(1 for v in result.votes if v is Verdict.YES) == 3
        assert result.agreement_ratio == 3 / 5


# ---------------------------------------------------------------------------
# JudgmentResult fields
# ---------------------------------------------------------------------------


class TestJudgmentResultFields:
    """Verify the returned JudgmentResult has correct metadata."""

    async def test_check_id_preserved(self) -> None:
        j = BinaryJudge(llm_factory=_make_mock_llm_factory("YES"), num_runs=3)
        result = await j.judge("my-check-42", "prompt text")
        assert result.check_id == "my-check-42"

    async def test_prompt_preserved(self) -> None:
        j = BinaryJudge(llm_factory=_make_mock_llm_factory("YES"), num_runs=3)
        result = await j.judge("c1", "the evaluation prompt")
        assert result.prompt == "the evaluation prompt"

    async def test_raw_responses_collected(self) -> None:
        j = BinaryJudge(
            llm_factory=_make_mock_llm_factory_sequence(["YES", "NO", "YES"]),
            num_runs=3,
        )
        result = await j.judge("c2", "p")
        # With concurrent execution, order may vary; check counts instead.
        assert sorted(result.raw_responses) == sorted(["NO", "YES", "YES"])

    async def test_result_is_judgment_result_type(self) -> None:
        j = BinaryJudge(llm_factory=_make_mock_llm_factory("NO"), num_runs=3)
        result = await j.judge("c3", "p")
        assert isinstance(result, JudgmentResult)


# ---------------------------------------------------------------------------
# Error handling / retries
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """Test API error retry and default behavior."""

    async def test_retry_once_on_error_then_recover(self) -> None:
        """First call fails, retry succeeds -> uses successful response."""
        j = BinaryJudge(
            llm_factory=_make_mock_llm_factory_with_errors(
                [
                    # Single run: fail then succeed on retry
                    Exception("API timeout"),
                    "YES",
                ]
            ),
            num_runs=1,
        )
        result = await j.judge("err-1", "test")
        assert result.verdict is Verdict.YES

    async def test_both_attempts_fail_defaults_to_no(self) -> None:
        """All retry attempts fail -> empty string -> NO."""
        j = BinaryJudge(
            llm_factory=_make_mock_llm_factory_with_errors(
                [Exception("always fail")] * 5  # _MAX_RETRIES = 5
            ),
            num_runs=1,
        )
        result = await j.judge("err-2", "test")
        assert result.verdict is Verdict.NO
        assert result.raw_responses == [""]

    async def test_mixed_errors_and_successes(self) -> None:
        """Single run: multiple errors then success."""
        j = BinaryJudge(
            llm_factory=_make_mock_llm_factory_with_errors(
                [
                    # Run 1: two failures then success
                    Exception("fail"),
                    Exception("fail"),
                    "YES",
                ]
            ),
            num_runs=1,
        )
        result = await j.judge("err-3", "test")
        assert result.verdict is Verdict.YES


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


class TestGetLog:
    """Test get_log() returns accumulated results."""

    async def test_log_accumulates(self) -> None:
        j = BinaryJudge(llm_factory=_make_mock_llm_factory("YES"), num_runs=3)
        await j.judge("a", "p1")
        await j.judge("b", "p2")
        await j.judge("c", "p3")
        log = j.get_log()
        assert len(log) == 3
        assert [r.check_id for r in log] == ["a", "b", "c"]

    async def test_log_returns_copy(self) -> None:
        j = BinaryJudge(llm_factory=_make_mock_llm_factory("YES"), num_runs=3)
        await j.judge("x", "p")
        log1 = j.get_log()
        log2 = j.get_log()
        assert log1 is not log2
        assert log1 == log2

    def test_empty_log_initially(self, judge: BinaryJudge) -> None:
        assert judge.get_log() == []


# ---------------------------------------------------------------------------
# Export log
# ---------------------------------------------------------------------------


class TestExportLog:
    """Test export_log() writes valid JSON."""

    async def test_export_creates_json_file(self, tmp_path: Path) -> None:
        j = BinaryJudge(llm_factory=_make_mock_llm_factory("YES"), num_runs=3)
        await j.judge("e1", "prompt1")
        await j.judge("e2", "prompt2")
        out = tmp_path / "log.json"
        j.export_log(out)
        assert out.exists()
        data = json.loads(out.read_text())
        assert isinstance(data, list)
        assert len(data) == 2
        assert data[0]["check_id"] == "e1"
        assert data[1]["check_id"] == "e2"

    async def test_export_json_structure(self, tmp_path: Path) -> None:
        j = BinaryJudge(llm_factory=_make_mock_llm_factory("NO"), num_runs=3)
        await j.judge("s1", "some prompt")
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

    def test_export_empty_log(self, judge: BinaryJudge, tmp_path: Path) -> None:
        out = tmp_path / "empty.json"
        judge.export_log(out)
        data = json.loads(out.read_text())
        assert data == []


# ---------------------------------------------------------------------------
# LLM call parameters
# ---------------------------------------------------------------------------


class TestLLMCallParameters:
    """Verify BaseChatModel.ainvoke is called with correct params."""

    async def test_uses_system_prompt(self) -> None:
        j = BinaryJudge(llm_factory=_make_mock_llm_factory("YES"), num_runs=1)
        await j.judge("p1", "user prompt")
        messages = j.llm.ainvoke.call_args[0][0]
        assert isinstance(messages[0], SystemMessage)
        assert messages[0].content == BINARY_JUDGE_SYSTEM_PROMPT

    async def test_uses_user_prompt(self) -> None:
        j = BinaryJudge(llm_factory=_make_mock_llm_factory("YES"), num_runs=1)
        await j.judge("p1", "my user prompt")
        messages = j.llm.ainvoke.call_args[0][0]
        assert isinstance(messages[1], HumanMessage)
        assert messages[1].content == "my user prompt"

    def test_temperature_and_max_tokens_passed_to_factory(self) -> None:
        factory = MagicMock(return_value=_make_mock_llm())
        BinaryJudge(llm_factory=factory, num_runs=1, temperature=0.5, max_tokens=20)
        assert factory.call_count == 2
        factory.assert_any_call(0.5, 20)
        factory.assert_any_call(0.5, 200)

    async def test_called_num_runs_times(self) -> None:
        j = BinaryJudge(llm_factory=_make_mock_llm_factory("YES"), num_runs=5)
        await j.judge("p1", "prompt")
        assert j.llm.ainvoke.call_count == 5
