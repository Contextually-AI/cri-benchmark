"""Comprehensive tests for evaluation rubrics — binary verdict functions and legacy rubrics.

Test coverage:
- format_facts: empty, single, multiple, truncation at MAX_FACTS_PER_PROMPT
- All binary rubric functions: non-empty output, correct content, YES/NO instruction,
  semantic equivalence mention, TASK/QUESTION structure
- Edge cases: empty facts, large facts, special characters
- Callable verification: all rubrics callable with correct arg count
- Legacy rubrics: all dimensions covered, content checks, string/enum lookup
"""

from __future__ import annotations

import inspect

from cri.scoring.rubrics import (
    MAX_FACTS_PER_PROMPT,
    crq_resolution_check,
    dbu_recency_check,
    dbu_staleness_check,
    format_facts,
    mei_coverage_check,
    mei_coverage_chunk_check,
    pas_check,
    qrp_irrelevance_check,
    qrp_relevance_check,
    tc_temporal_validity_check,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SAMPLE_FACTS = [
    "User works as a software engineer",
    "User lives in Buenos Aires",
    "User prefers Python over JavaScript",
]

ALL_BINARY_RUBRIC_FUNCS = [
    pas_check,
    dbu_recency_check,
    dbu_staleness_check,
    tc_temporal_validity_check,
    crq_resolution_check,
    qrp_relevance_check,
    qrp_irrelevance_check,
    mei_coverage_check,
    mei_coverage_chunk_check,
]


# ---------------------------------------------------------------------------
# format_facts tests
# ---------------------------------------------------------------------------


class TestFormatFacts:
    def test_empty_list(self) -> None:
        result = format_facts([])
        assert result == "(no facts provided)"

    def test_single_fact(self) -> None:
        result = format_facts(["The user likes Python"])
        assert "1. The user likes Python" in result

    def test_multiple_facts(self) -> None:
        facts = ["Fact A", "Fact B", "Fact C"]
        result = format_facts(facts)
        assert "1. Fact A" in result
        assert "2. Fact B" in result
        assert "3. Fact C" in result

    def test_truncation_at_max(self) -> None:
        facts = [f"Fact {i}" for i in range(MAX_FACTS_PER_PROMPT + 10)]
        result = format_facts(facts)
        # Should include exactly MAX_FACTS_PER_PROMPT facts
        assert f"{MAX_FACTS_PER_PROMPT}. Fact {MAX_FACTS_PER_PROMPT - 1}" in result
        # Should NOT include facts beyond the limit
        assert f"{MAX_FACTS_PER_PROMPT + 1}." not in result
        # Should include truncation note
        assert "10 more facts not shown" in result

    def test_exactly_max_facts_no_truncation_note(self) -> None:
        facts = [f"Fact {i}" for i in range(MAX_FACTS_PER_PROMPT)]
        result = format_facts(facts)
        assert "more facts not shown" not in result
        assert f"{MAX_FACTS_PER_PROMPT}." in result

    def test_numbered_sequentially(self) -> None:
        facts = ["A", "B", "C"]
        result = format_facts(facts)
        lines = [line.strip() for line in result.strip().split("\n")]
        assert lines[0].startswith("1.")
        assert lines[1].startswith("2.")
        assert lines[2].startswith("3.")

    def test_one_over_max_shows_truncation(self) -> None:
        facts = [f"Fact {i}" for i in range(MAX_FACTS_PER_PROMPT + 1)]
        result = format_facts(facts)
        assert "1 more facts not shown" in result

    def test_large_excess_shows_correct_count(self) -> None:
        excess = 42
        facts = [f"Fact {i}" for i in range(MAX_FACTS_PER_PROMPT + excess)]
        result = format_facts(facts)
        assert f"{excess} more facts not shown" in result

    def test_returns_string(self) -> None:
        assert isinstance(format_facts([]), str)
        assert isinstance(format_facts(["a"]), str)
        assert isinstance(format_facts(["a", "b"]), str)


# ---------------------------------------------------------------------------
# Binary verdict rubric function tests — PAS
# ---------------------------------------------------------------------------


class TestPasCheck:
    def test_returns_nonempty_string(self) -> None:
        result = pas_check("occupation", "software engineer", SAMPLE_FACTS)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_contains_dimension_and_value(self) -> None:
        result = pas_check("occupation", "software engineer", SAMPLE_FACTS)
        assert "occupation" in result
        assert "software engineer" in result

    def test_contains_yes_no_instruction(self) -> None:
        result = pas_check("occupation", "software engineer", SAMPLE_FACTS)
        assert "YES" in result
        assert "NO" in result

    def test_contains_semantic_equivalence_note(self) -> None:
        result = pas_check("occupation", "software engineer", SAMPLE_FACTS)
        assert "semantic" in result.lower()

    def test_includes_stored_facts(self) -> None:
        result = pas_check("city", "Buenos Aires", SAMPLE_FACTS)
        assert "Buenos Aires" in result
        assert "software engineer" in result  # from stored facts

    def test_has_task_and_question_sections(self) -> None:
        result = pas_check("name", "Alice", SAMPLE_FACTS)
        assert "TASK" in result
        assert "QUESTION" in result
        assert "Answer YES or NO" in result


# ---------------------------------------------------------------------------
# DBU rubric tests
# ---------------------------------------------------------------------------


class TestDbuRecencyCheck:
    def test_returns_nonempty_string(self) -> None:
        result = dbu_recency_check("job title", "tech lead", SAMPLE_FACTS)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_contains_fact_name_and_value(self) -> None:
        result = dbu_recency_check("job title", "tech lead", SAMPLE_FACTS)
        assert "job title" in result
        assert "tech lead" in result

    def test_contains_yes_no_instruction(self) -> None:
        result = dbu_recency_check("job title", "tech lead", SAMPLE_FACTS)
        assert "YES" in result
        assert "NO" in result

    def test_mentions_updated_or_current(self) -> None:
        result = dbu_recency_check("job title", "tech lead", SAMPLE_FACTS)
        assert "current" in result.lower() or "updated" in result.lower()


class TestDbuStalenessCheck:
    def test_returns_nonempty_string(self) -> None:
        result = dbu_staleness_check("job title", "junior dev", SAMPLE_FACTS)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_contains_old_value(self) -> None:
        result = dbu_staleness_check("job title", "junior dev", SAMPLE_FACTS)
        assert "junior dev" in result

    def test_indicates_yes_means_failure(self) -> None:
        result = dbu_staleness_check("job title", "junior dev", SAMPLE_FACTS)
        assert "FAILED" in result or "failed" in result.lower()

    def test_mentions_historical_acceptable(self) -> None:
        result = dbu_staleness_check("job title", "junior dev", SAMPLE_FACTS)
        assert "historical" in result.lower() or "previously" in result.lower()

    def test_contains_yes_no_instruction(self) -> None:
        result = dbu_staleness_check("job title", "junior dev", SAMPLE_FACTS)
        assert "YES" in result
        assert "NO" in result


# ---------------------------------------------------------------------------
# TC rubric tests
# ---------------------------------------------------------------------------


class TestTcTemporalValidityCheck:
    def test_returns_nonempty_string_current_true(self) -> None:
        result = tc_temporal_validity_check("User lives in BA", True, SAMPLE_FACTS)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_returns_nonempty_string_current_false(self) -> None:
        result = tc_temporal_validity_check("User lived in NYC", False, SAMPLE_FACTS)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_expected_current_true_asks_if_present(self) -> None:
        result = tc_temporal_validity_check("User lives in BA", True, SAMPLE_FACTS)
        assert "currently valid" in result.lower() or "current" in result.lower()
        assert "EXPECTED TO BE CURRENT: Yes" in result

    def test_expected_current_false_asks_if_still_asserted(self) -> None:
        result = tc_temporal_validity_check("User lived in NYC", False, SAMPLE_FACTS)
        assert "EXPECTED TO BE CURRENT: No" in result
        assert "FAILED" in result

    def test_contains_fact_description(self) -> None:
        result = tc_temporal_validity_check("User lives in BA", True, SAMPLE_FACTS)
        assert "User lives in BA" in result

    def test_contains_yes_no_instruction(self) -> None:
        result = tc_temporal_validity_check("User lives in BA", True, SAMPLE_FACTS)
        assert "YES" in result
        assert "NO" in result


# ---------------------------------------------------------------------------
# CRQ rubric tests
# ---------------------------------------------------------------------------


class TestCrqResolutionCheck:
    def test_returns_nonempty_string(self) -> None:
        result = crq_resolution_check("favorite language", "Python", SAMPLE_FACTS)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_contains_topic_and_resolution(self) -> None:
        result = crq_resolution_check("favorite language", "Python", SAMPLE_FACTS)
        assert "favorite language" in result
        assert "Python" in result

    def test_contains_yes_no_instruction(self) -> None:
        result = crq_resolution_check("favorite language", "Python", SAMPLE_FACTS)
        assert "YES" in result
        assert "NO" in result

    def test_mentions_conflict(self) -> None:
        result = crq_resolution_check("topic", "resolution", SAMPLE_FACTS)
        assert "conflict" in result.lower()


# ---------------------------------------------------------------------------
# QRP rubric tests
# ---------------------------------------------------------------------------


class TestQrpRelevanceCheck:
    def test_returns_nonempty_string(self) -> None:
        result = qrp_relevance_check("Where does the user work?", "software engineer", SAMPLE_FACTS)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_contains_query_and_expected_fact(self) -> None:
        result = qrp_relevance_check("Where does the user work?", "software engineer", SAMPLE_FACTS)
        assert "Where does the user work?" in result
        assert "software engineer" in result

    def test_contains_yes_no_instruction(self) -> None:
        result = qrp_relevance_check("Where does the user work?", "software engineer", SAMPLE_FACTS)
        assert "YES" in result
        assert "NO" in result


class TestQrpIrrelevanceCheck:
    def test_returns_nonempty_string(self) -> None:
        result = qrp_irrelevance_check(
            "What is the user's job?",
            "User lives in Buenos Aires",
            SAMPLE_FACTS,
        )
        assert isinstance(result, str)
        assert len(result) > 0

    def test_contains_query_and_irrelevant_fact(self) -> None:
        result = qrp_irrelevance_check(
            "What is the user's job?",
            "User lives in Buenos Aires",
            SAMPLE_FACTS,
        )
        assert "What is the user's job?" in result
        assert "User lives in Buenos Aires" in result

    def test_indicates_yes_means_failure(self) -> None:
        result = qrp_irrelevance_check(
            "What is the user's job?",
            "User lives in Buenos Aires",
            SAMPLE_FACTS,
        )
        assert "FAILED" in result or "failed" in result.lower()

    def test_contains_yes_no_instruction(self) -> None:
        result = qrp_irrelevance_check(
            "What is the user's job?",
            "User lives in Buenos Aires",
            SAMPLE_FACTS,
        )
        assert "YES" in result
        assert "NO" in result


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------


class TestRubricEdgeCases:
    def test_empty_facts_list_pas(self) -> None:
        result = pas_check("name", "Alice", [])
        assert "(no facts provided)" in result

    def test_empty_facts_list_dbu_recency(self) -> None:
        result = dbu_recency_check("job", "dev", [])
        assert "(no facts provided)" in result

    def test_empty_facts_list_dbu_staleness(self) -> None:
        result = dbu_staleness_check("job", "old", [])
        assert "(no facts provided)" in result

    def test_empty_facts_list_tc(self) -> None:
        result = tc_temporal_validity_check("fact", True, [])
        assert "(no facts provided)" in result

    def test_empty_facts_list_crq(self) -> None:
        result = crq_resolution_check("topic", "res", [])
        assert "(no facts provided)" in result

    def test_empty_facts_list_qrp_relevance(self) -> None:
        result = qrp_relevance_check("query", "fact", [])
        assert "(no facts provided)" in result

    def test_empty_facts_list_qrp_irrelevance(self) -> None:
        result = qrp_irrelevance_check("query", "fact", [])
        assert "(no facts provided)" in result

    def test_large_facts_list_truncation(self) -> None:
        big_facts = [f"Fact number {i}" for i in range(50)]
        result = pas_check("name", "Alice", big_facts)
        assert "more facts not shown" in result
        result = crq_resolution_check("topic", "resolved", big_facts)
        assert "more facts not shown" in result

    def test_special_characters_in_inputs(self) -> None:
        result = pas_check(
            "favorite_emoji",
            '🎉 "party" & fun <html>',
            ['User loves 🎉 "party" events'],
        )
        assert isinstance(result, str)
        assert "🎉" in result

    def test_newlines_in_inputs(self) -> None:
        result = pas_check("bio", "line1\nline2", ["fact with\nnewline"])
        assert isinstance(result, str)

    def test_empty_string_inputs(self) -> None:
        result = pas_check("", "", [""])
        assert isinstance(result, str)
        assert "TASK" in result


# ---------------------------------------------------------------------------
# Prompt structure validation (cross-cutting)
# ---------------------------------------------------------------------------


class TestPromptStructure:
    """All prompts should have TASK and QUESTION sections."""

    def test_all_rubrics_have_task_and_question(self) -> None:
        funcs_and_args = [
            (pas_check, ("dim", "val", SAMPLE_FACTS)),
            (dbu_recency_check, ("fact", "val", SAMPLE_FACTS)),
            (dbu_staleness_check, ("fact", "val", SAMPLE_FACTS)),
            (tc_temporal_validity_check, ("fact", True, SAMPLE_FACTS)),
            (tc_temporal_validity_check, ("fact", False, SAMPLE_FACTS)),
            (crq_resolution_check, ("topic", "resolution", SAMPLE_FACTS)),
            (qrp_relevance_check, ("query", "fact", SAMPLE_FACTS)),
            (qrp_irrelevance_check, ("query", "fact", SAMPLE_FACTS)),
            (mei_coverage_check, ("occupation", "software engineer", SAMPLE_FACTS)),
        ]
        for func, args in funcs_and_args:
            result = func(*args)
            assert "TASK" in result, f"{func.__name__} missing TASK section"
            assert "QUESTION" in result, f"{func.__name__} missing QUESTION section"
            assert "Answer YES or NO" in result, f"{func.__name__} missing YES/NO instruction"

    def test_mei_chunk_check_has_task_and_instructions(self) -> None:
        result = mei_coverage_chunk_check(SAMPLE_FACTS, [("occupation", "software engineer")])
        assert "TASK" in result, "mei_coverage_chunk_check missing TASK section"
        assert "INSTRUCTIONS" in result, "mei_coverage_chunk_check missing INSTRUCTIONS section"

    def test_all_rubrics_return_string(self) -> None:
        funcs_and_args = [
            (pas_check, ("dim", "val", SAMPLE_FACTS)),
            (dbu_recency_check, ("fact", "val", SAMPLE_FACTS)),
            (dbu_staleness_check, ("fact", "val", SAMPLE_FACTS)),
            (tc_temporal_validity_check, ("fact", True, SAMPLE_FACTS)),
            (crq_resolution_check, ("topic", "resolution", SAMPLE_FACTS)),
            (qrp_relevance_check, ("query", "fact", SAMPLE_FACTS)),
            (qrp_irrelevance_check, ("query", "fact", SAMPLE_FACTS)),
            (mei_coverage_check, ("occupation", "software engineer", SAMPLE_FACTS)),
            (mei_coverage_chunk_check, (SAMPLE_FACTS, [("occupation", "software engineer")])),
        ]
        for func, args in funcs_and_args:
            result = func(*args)
            assert isinstance(result, str), f"{func.__name__} did not return a string"
            assert len(result) > 0, f"{func.__name__} returned empty string"


# ---------------------------------------------------------------------------
# Callable verification — correct argument counts
# ---------------------------------------------------------------------------


class TestCallableArgCounts:
    """Verify all rubric functions are callable with the correct number of arguments."""

    def test_pas_check_takes_3_args(self) -> None:
        sig = inspect.signature(pas_check)
        assert len(sig.parameters) == 3

    def test_dbu_recency_check_takes_3_args(self) -> None:
        sig = inspect.signature(dbu_recency_check)
        assert len(sig.parameters) == 3

    def test_dbu_staleness_check_takes_3_args(self) -> None:
        sig = inspect.signature(dbu_staleness_check)
        assert len(sig.parameters) == 3

    def test_tc_temporal_validity_check_takes_3_args(self) -> None:
        sig = inspect.signature(tc_temporal_validity_check)
        assert len(sig.parameters) == 3

    def test_crq_resolution_check_takes_3_args(self) -> None:
        sig = inspect.signature(crq_resolution_check)
        assert len(sig.parameters) == 3

    def test_qrp_relevance_check_takes_3_args(self) -> None:
        sig = inspect.signature(qrp_relevance_check)
        assert len(sig.parameters) == 3

    def test_qrp_irrelevance_check_takes_3_args(self) -> None:
        sig = inspect.signature(qrp_irrelevance_check)
        assert len(sig.parameters) == 3

    def test_mei_coverage_check_takes_3_args(self) -> None:
        sig = inspect.signature(mei_coverage_check)
        assert len(sig.parameters) == 3

    def test_mei_coverage_chunk_check_takes_2_args(self) -> None:
        sig = inspect.signature(mei_coverage_chunk_check)
        assert len(sig.parameters) == 2

    def test_all_rubrics_are_callable(self) -> None:
        for func in ALL_BINARY_RUBRIC_FUNCS:
            assert callable(func), f"{func.__name__} is not callable"

    def test_format_facts_takes_1_arg(self) -> None:
        sig = inspect.signature(format_facts)
        assert len(sig.parameters) == 1
