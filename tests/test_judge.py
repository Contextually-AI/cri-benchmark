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
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

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


def _make_llm_response(content: str) -> MagicMock:
    """Create a mock litellm completion response."""
    choice = MagicMock()
    choice.message.content = content
    resp = MagicMock()
    resp.choices = [choice]
    return resp


def _make_async_llm_response(content: str) -> AsyncMock:
    """Create a mock async litellm completion response."""
    mock_response = AsyncMock()
    mock_response.choices = [AsyncMock(message=AsyncMock(content=content))]
    return mock_response


class TestBinaryJudgeInit:
    """Test BinaryJudge constructor defaults."""

    def test_default_model(self) -> None:
        j = BinaryJudge()
        assert j.model == "claude-haiku-4-5-20250315"

    def test_default_num_runs(self) -> None:
        j = BinaryJudge()
        assert j.num_runs == 3

    def test_default_temperature(self) -> None:
        j = BinaryJudge()
        assert j.temperature == 0.0

    def test_default_max_tokens(self) -> None:
        j = BinaryJudge()
        assert j.max_tokens == 10

    def test_custom_params(self) -> None:
        j = BinaryJudge(model="gpt-4o", num_runs=5, temperature=0.5, max_tokens=20)
        assert j.model == "gpt-4o"
        assert j.num_runs == 5
        assert j.temperature == 0.5
        assert j.max_tokens == 20

    def test_empty_log_on_init(self) -> None:
        j = BinaryJudge()
        assert j.get_log() == []


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
        return BinaryJudge(model="test-model", num_runs=3)

    @patch("cri.judge.litellm.completion")
    def test_2_yes_1_no_yields_yes(self, mock_comp: MagicMock, judge: BinaryJudge) -> None:
        mock_comp.side_effect = [
            _make_llm_response("YES"),
            _make_llm_response("YES"),
            _make_llm_response("NO"),
        ]
        result = judge.judge("chk-1", "Is sky blue?")
        assert result.verdict is Verdict.YES
        assert result.votes == [Verdict.YES, Verdict.YES, Verdict.NO]
        assert result.unanimous is False

    @patch("cri.judge.litellm.completion")
    def test_1_yes_2_no_yields_no(self, mock_comp: MagicMock, judge: BinaryJudge) -> None:
        mock_comp.side_effect = [
            _make_llm_response("YES"),
            _make_llm_response("NO"),
            _make_llm_response("NO"),
        ]
        result = judge.judge("chk-2", "Is sky green?")
        assert result.verdict is Verdict.NO
        assert result.votes == [Verdict.YES, Verdict.NO, Verdict.NO]
        assert result.unanimous is False

    @patch("cri.judge.litellm.completion")
    def test_unanimous_yes(self, mock_comp: MagicMock, judge: BinaryJudge) -> None:
        mock_comp.return_value = _make_llm_response("YES")
        result = judge.judge("chk-3", "Test?")
        assert result.verdict is Verdict.YES
        assert result.unanimous is True
        assert all(v is Verdict.YES for v in result.votes)

    @patch("cri.judge.litellm.completion")
    def test_unanimous_no(self, mock_comp: MagicMock, judge: BinaryJudge) -> None:
        mock_comp.return_value = _make_llm_response("NO")
        result = judge.judge("chk-4", "Test?")
        assert result.verdict is Verdict.NO
        assert result.unanimous is True
        assert all(v is Verdict.NO for v in result.votes)

    @patch("cri.judge.litellm.completion")
    def test_all_non_binary_defaults_to_no(self, mock_comp: MagicMock, judge: BinaryJudge) -> None:
        mock_comp.return_value = _make_llm_response("MAYBE")
        result = judge.judge("chk-5", "Unclear?")
        assert result.verdict is Verdict.NO
        assert all(v is Verdict.NO for v in result.votes)

    @patch("cri.judge.litellm.completion")
    def test_3_yes_0_no_is_unanimous_yes(self, mock_comp: MagicMock, judge: BinaryJudge) -> None:
        mock_comp.return_value = _make_llm_response("YES")
        result = judge.judge("u-yes", "p")
        assert result.verdict is Verdict.YES
        assert result.unanimous is True
        assert len(result.votes) == 3

    @patch("cri.judge.litellm.completion")
    def test_0_yes_3_no_is_unanimous_no(self, mock_comp: MagicMock, judge: BinaryJudge) -> None:
        mock_comp.return_value = _make_llm_response("NO")
        result = judge.judge("u-no", "p")
        assert result.verdict is Verdict.NO
        assert result.unanimous is True

    @patch("cri.judge.litellm.completion")
    def test_single_run_yes(self, mock_comp: MagicMock) -> None:
        j = BinaryJudge(model="test", num_runs=1)
        mock_comp.return_value = _make_llm_response("YES")
        result = j.judge("single", "p")
        assert result.verdict is Verdict.YES
        assert len(result.votes) == 1

    @patch("cri.judge.litellm.completion")
    def test_five_runs_majority(self, mock_comp: MagicMock) -> None:
        j = BinaryJudge(model="test", num_runs=5)
        mock_comp.side_effect = [
            _make_llm_response("YES"),
            _make_llm_response("YES"),
            _make_llm_response("YES"),
            _make_llm_response("NO"),
            _make_llm_response("NO"),
        ]
        result = j.judge("five", "p")
        assert result.verdict is Verdict.YES
        assert len(result.votes) == 5


# ---------------------------------------------------------------------------
# JudgmentResult fields
# ---------------------------------------------------------------------------


class TestJudgmentResultFields:
    """Verify the returned JudgmentResult has correct metadata."""

    @pytest.fixture
    def judge(self) -> BinaryJudge:
        return BinaryJudge(model="test-model", num_runs=3)

    @patch("cri.judge.litellm.completion")
    def test_check_id_preserved(self, mock_comp: MagicMock, judge: BinaryJudge) -> None:
        mock_comp.return_value = _make_llm_response("YES")
        result = judge.judge("my-check-42", "prompt text")
        assert result.check_id == "my-check-42"

    @patch("cri.judge.litellm.completion")
    def test_prompt_preserved(self, mock_comp: MagicMock, judge: BinaryJudge) -> None:
        mock_comp.return_value = _make_llm_response("YES")
        result = judge.judge("c1", "the evaluation prompt")
        assert result.prompt == "the evaluation prompt"

    @patch("cri.judge.litellm.completion")
    def test_raw_responses_collected(self, mock_comp: MagicMock, judge: BinaryJudge) -> None:
        mock_comp.side_effect = [
            _make_llm_response("YES"),
            _make_llm_response("NO"),
            _make_llm_response("YES"),
        ]
        result = judge.judge("c2", "p")
        assert result.raw_responses == ["YES", "NO", "YES"]

    @patch("cri.judge.litellm.completion")
    def test_result_is_judgment_result_type(self, mock_comp: MagicMock, judge: BinaryJudge) -> None:
        mock_comp.return_value = _make_llm_response("NO")
        result = judge.judge("c3", "p")
        assert isinstance(result, JudgmentResult)

    @patch("cri.judge.litellm.completion")
    def test_votes_count_matches_num_runs(self, mock_comp: MagicMock, judge: BinaryJudge) -> None:
        mock_comp.return_value = _make_llm_response("YES")
        result = judge.judge("count", "p")
        assert len(result.votes) == 3
        assert len(result.raw_responses) == 3


# ---------------------------------------------------------------------------
# Error handling / retries
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """Test API error retry and default behavior."""

    @pytest.fixture
    def judge(self) -> BinaryJudge:
        return BinaryJudge(model="test-model", num_runs=3)

    @patch("cri.judge.litellm.completion")
    def test_retry_once_on_error_then_recover(
        self, mock_comp: MagicMock, judge: BinaryJudge
    ) -> None:
        mock_comp.side_effect = [
            # Run 1: fail then succeed on retry
            Exception("API timeout"),
            _make_llm_response("YES"),
            # Run 2: success
            _make_llm_response("YES"),
            # Run 3: success
            _make_llm_response("NO"),
        ]
        result = judge.judge("err-1", "test")
        assert result.verdict is Verdict.YES

    @patch("cri.judge.litellm.completion")
    def test_both_attempts_fail_defaults_to_no(self, mock_comp: MagicMock) -> None:
        j = BinaryJudge(model="test", num_runs=1)
        mock_comp.side_effect = [
            Exception("fail 1"),
            Exception("fail 2"),
        ]
        result = j.judge("err-2", "test")
        assert result.verdict is Verdict.NO
        assert result.raw_responses == [""]

    @patch("cri.judge.litellm.completion")
    def test_mixed_errors_and_successes(self, mock_comp: MagicMock, judge: BinaryJudge) -> None:
        mock_comp.side_effect = [
            # Run 1: both attempts fail → ""
            Exception("fail"),
            Exception("fail"),
            # Run 2: success
            _make_llm_response("YES"),
            # Run 3: success
            _make_llm_response("YES"),
        ]
        result = judge.judge("err-3", "test")
        # Votes: NO (from empty), YES, YES → majority YES
        assert result.verdict is Verdict.YES

    @patch("cri.judge.litellm.completion")
    def test_all_runs_fail_results_in_no(self, mock_comp: MagicMock) -> None:
        j = BinaryJudge(model="test", num_runs=3)
        mock_comp.side_effect = Exception("always fail")
        result = j.judge("all-fail", "test")
        assert result.verdict is Verdict.NO
        assert all(v is Verdict.NO for v in result.votes)
        assert all(r == "" for r in result.raw_responses)

    @patch("cri.judge.litellm.completion")
    def test_none_content_treated_as_empty(self, mock_comp: MagicMock) -> None:
        """If LLM returns None content, it should be treated as empty string."""
        choice = MagicMock()
        choice.message.content = None
        resp = MagicMock()
        resp.choices = [choice]
        mock_comp.return_value = resp

        j = BinaryJudge(model="test", num_runs=1)
        result = j.judge("none-content", "p")
        assert result.verdict is Verdict.NO


# ---------------------------------------------------------------------------
# Log recording
# ---------------------------------------------------------------------------


class TestGetLog:
    """Test get_log() returns accumulated results."""

    @pytest.fixture
    def judge(self) -> BinaryJudge:
        return BinaryJudge(model="test-model", num_runs=3)

    @patch("cri.judge.litellm.completion")
    def test_log_accumulates(self, mock_comp: MagicMock, judge: BinaryJudge) -> None:
        mock_comp.return_value = _make_llm_response("YES")
        judge.judge("a", "p1")
        judge.judge("b", "p2")
        judge.judge("c", "p3")
        log = judge.get_log()
        assert len(log) == 3
        assert [r.check_id for r in log] == ["a", "b", "c"]

    @patch("cri.judge.litellm.completion")
    def test_log_returns_copy(self, mock_comp: MagicMock, judge: BinaryJudge) -> None:
        mock_comp.return_value = _make_llm_response("YES")
        judge.judge("x", "p")
        log1 = judge.get_log()
        log2 = judge.get_log()
        assert log1 is not log2
        assert log1 == log2

    def test_empty_log_initially(self, judge: BinaryJudge) -> None:
        assert judge.get_log() == []

    @patch("cri.judge.litellm.completion")
    def test_log_preserves_verdicts(self, mock_comp: MagicMock, judge: BinaryJudge) -> None:
        mock_comp.side_effect = [
            _make_llm_response("YES"),
            _make_llm_response("YES"),
            _make_llm_response("YES"),
            _make_llm_response("NO"),
            _make_llm_response("NO"),
            _make_llm_response("NO"),
        ]
        judge.judge("yes-check", "p1")
        judge.judge("no-check", "p2")
        log = judge.get_log()
        assert log[0].verdict is Verdict.YES
        assert log[1].verdict is Verdict.NO

    @patch("cri.judge.litellm.completion")
    def test_log_not_affected_by_external_mutation(
        self, mock_comp: MagicMock, judge: BinaryJudge
    ) -> None:
        mock_comp.return_value = _make_llm_response("YES")
        judge.judge("m", "p")
        log = judge.get_log()
        log.clear()  # mutate the returned copy
        assert len(judge.get_log()) == 1  # internal log unaffected


# ---------------------------------------------------------------------------
# Export log
# ---------------------------------------------------------------------------


class TestExportLog:
    """Test export_log() writes valid JSON."""

    @pytest.fixture
    def judge(self) -> BinaryJudge:
        return BinaryJudge(model="test-model", num_runs=3)

    @patch("cri.judge.litellm.completion")
    def test_export_creates_json_file(
        self, mock_comp: MagicMock, judge: BinaryJudge, tmp_path: Path
    ) -> None:
        mock_comp.return_value = _make_llm_response("YES")
        judge.judge("e1", "prompt1")
        judge.judge("e2", "prompt2")
        out = tmp_path / "log.json"
        judge.export_log(out)
        assert out.exists()
        data = json.loads(out.read_text())
        assert isinstance(data, list)
        assert len(data) == 2
        assert data[0]["check_id"] == "e1"
        assert data[1]["check_id"] == "e2"

    @patch("cri.judge.litellm.completion")
    def test_export_json_structure(
        self, mock_comp: MagicMock, judge: BinaryJudge, tmp_path: Path
    ) -> None:
        mock_comp.return_value = _make_llm_response("NO")
        judge.judge("s1", "some prompt")
        out = tmp_path / "struct.json"
        judge.export_log(out)
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

    @patch("cri.judge.litellm.completion")
    def test_export_is_valid_json(
        self, mock_comp: MagicMock, judge: BinaryJudge, tmp_path: Path
    ) -> None:
        mock_comp.return_value = _make_llm_response("YES")
        judge.judge("j1", 'prompt with "quotes" and\nnewlines')
        out = tmp_path / "valid.json"
        judge.export_log(out)
        # Should not raise
        data = json.loads(out.read_text())
        assert len(data) == 1

    @patch("cri.judge.litellm.completion")
    def test_export_overwrites_existing_file(
        self, mock_comp: MagicMock, judge: BinaryJudge, tmp_path: Path
    ) -> None:
        mock_comp.return_value = _make_llm_response("YES")
        out = tmp_path / "overwrite.json"
        out.write_text("old content")
        judge.judge("new", "p")
        judge.export_log(out)
        data = json.loads(out.read_text())
        assert len(data) == 1
        assert data[0]["check_id"] == "new"


# ---------------------------------------------------------------------------
# LLM call parameters
# ---------------------------------------------------------------------------


class TestLLMCallParameters:
    """Verify litellm.completion is called with correct params."""

    @patch("cri.judge.litellm.completion")
    def test_uses_correct_model(self, mock_comp: MagicMock) -> None:
        j = BinaryJudge(model="my-model", num_runs=1)
        mock_comp.return_value = _make_llm_response("YES")
        j.judge("p1", "prompt")
        call_kwargs = mock_comp.call_args
        assert call_kwargs.kwargs["model"] == "my-model"

    @patch("cri.judge.litellm.completion")
    def test_uses_system_prompt(self, mock_comp: MagicMock) -> None:
        j = BinaryJudge(num_runs=1)
        mock_comp.return_value = _make_llm_response("YES")
        j.judge("p1", "user prompt")
        messages = mock_comp.call_args.kwargs["messages"]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == BINARY_JUDGE_SYSTEM_PROMPT

    @patch("cri.judge.litellm.completion")
    def test_uses_user_prompt(self, mock_comp: MagicMock) -> None:
        j = BinaryJudge(num_runs=1)
        mock_comp.return_value = _make_llm_response("YES")
        j.judge("p1", "my user prompt")
        messages = mock_comp.call_args.kwargs["messages"]
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "my user prompt"

    @patch("cri.judge.litellm.completion")
    def test_temperature_and_max_tokens(self, mock_comp: MagicMock) -> None:
        j = BinaryJudge(num_runs=1, temperature=0.5, max_tokens=20)
        mock_comp.return_value = _make_llm_response("YES")
        j.judge("p1", "prompt")
        kw = mock_comp.call_args.kwargs
        assert kw["temperature"] == 0.5
        assert kw["max_tokens"] == 20

    @patch("cri.judge.litellm.completion")
    def test_called_num_runs_times(self, mock_comp: MagicMock) -> None:
        j = BinaryJudge(model="test", num_runs=5)
        mock_comp.return_value = _make_llm_response("YES")
        j.judge("p1", "prompt")
        assert mock_comp.call_count == 5

    @patch("cri.judge.litellm.completion")
    def test_messages_format(self, mock_comp: MagicMock) -> None:
        j = BinaryJudge(num_runs=1)
        mock_comp.return_value = _make_llm_response("YES")
        j.judge("fmt", "test prompt")
        messages = mock_comp.call_args.kwargs["messages"]
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
