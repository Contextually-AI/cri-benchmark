"""LLM-as-judge evaluation module (async, parallelised).

Uses a large language model to evaluate memory system responses against
expected answers, producing scored judgments for each evaluation dimension.

All judge methods are **async** and use a shared :class:`asyncio.Semaphore`
to cap concurrent API calls, with exponential-backoff retry on transient
errors.  This enables massive parallelisation across dimensions, checks,
chunks, and majority-vote rounds without overwhelming the API.

Provides two judge capabilities:

- :class:`BinaryJudge` -- a binary YES/NO judge with majority-vote
  robustness (used by most CRI evaluation dimensions).
- :meth:`BinaryJudge.judge_coverage` -- a multi-answer coverage judge
  used by MEI.  Given a chunk of stored facts and a list of ground-truth
  facts, it returns the set of GT fact indices covered by that chunk.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable
from pathlib import Path

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from cri.models import (
    JudgmentResult,
    Verdict,
)

logger = logging.getLogger(__name__)

BINARY_JUDGE_SYSTEM_PROMPT = "You are an evaluation judge. Answer only YES or NO."

COVERAGE_JUDGE_SYSTEM_PROMPT = (
    "You are an evaluation judge. You MUST respond with ONLY a JSON array of integers — nothing else. Example responses: [0, 2, 5]  or  []"
)

# Type alias for the LLM factory callable.
# Receives (temperature, max_tokens) and returns a configured BaseChatModel.
LLMFactory = Callable[[float, int], BaseChatModel]

# ---------------------------------------------------------------------------
# Retry / concurrency configuration
# ---------------------------------------------------------------------------
_MAX_RETRIES = 5  # total attempts per API call
_BACKOFF_BASE = 1.0  # seconds; delays: 1, 2, 4, 8, 16


def create_default_llm(temperature: float = 0.0, max_tokens: int = 10) -> BaseChatModel:
    """Create the default LLM for the BinaryJudge.

    Resolution order:

    1. ``ANTHROPIC_API_KEY`` environment variable -> standard ``ChatAnthropic``.
    2. OAuth token file at ``../.auth_token`` -> custom ``AnthropicOAuthChat``.

    Override entirely by passing a custom ``llm_factory`` to
    :class:`BinaryJudge`.

    Args:
        temperature: Sampling temperature.
        max_tokens: Maximum tokens in each LLM response.

    Returns:
        A configured :class:`BaseChatModel` instance.

    Raises:
        RuntimeError: If neither the API key nor the token file is available.
    """
    import os

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model="claude-sonnet-4-6",  # type: ignore[call-arg]
            max_tokens=max_tokens,
            temperature=temperature,
            api_key=api_key,  # type: ignore[arg-type]
        )

    from cri.utils.llm_anthropic_subscription import AnthropicOAuthChat

    token_path = Path("../.auth_token")
    if not token_path.exists():
        raise RuntimeError(
            f"No Anthropic credentials found. Either set the ANTHROPIC_API_KEY environment variable or place an OAuth token at {token_path.resolve()}"
        )
    return AnthropicOAuthChat(
        token_path=str(token_path),
        model_name="claude-sonnet-4-6",
        max_tokens=max_tokens,
        temperature=temperature,
    )


# ---------------------------------------------------------------------------
# Binary YES/NO judge with majority-vote robustness (fully async)
# ---------------------------------------------------------------------------


class BinaryJudge:
    """Async binary YES/NO LLM judge with majority-vote robustness.

    All public evaluation methods are coroutines.  A shared
    :class:`asyncio.Semaphore` (default 15) caps concurrent API calls,
    and every call retries with exponential backoff on transient errors.

    Args:
        llm_factory: A callable ``(temperature, max_tokens) -> BaseChatModel``
            that creates the LLM instance.  Defaults to
            :func:`create_default_llm` (Anthropic OAuth).
        num_runs: Number of independent LLM calls per judgment
            (default ``3``).  Must be a positive integer.  Odd values are
            recommended; with even values, ties resolve to NO.
        temperature: Sampling temperature (default ``0.0`` for
            deterministic output).  Passed to the factory.
        max_tokens: Maximum tokens in each LLM response (default ``10``).
            Passed to the factory.
        max_concurrency: Maximum number of simultaneous API calls
            (default ``15``).
    """

    def __init__(
        self,
        llm_factory: LLMFactory = create_default_llm,
        num_runs: int = 3,
        temperature: float = 0.0,
        max_tokens: int = 10,
        max_concurrency: int = 15,
    ) -> None:
        if num_runs < 1:
            raise ValueError(f"num_runs must be >= 1, got {num_runs}")
        if num_runs % 2 == 0:
            logger.warning(
                "num_runs=%d is even; ties will resolve to NO",
                num_runs,
            )
        self.num_runs = num_runs
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._llm_factory = llm_factory
        self.llm: BaseChatModel = llm_factory(temperature, max_tokens)
        # Separate LLM instance for coverage checks — needs more tokens to
        # return a JSON array of up to ~30 integers.
        self._coverage_llm: BaseChatModel = llm_factory(temperature, 200)
        self._log: list[JudgmentResult] = []
        self._semaphore = asyncio.Semaphore(max_concurrency)

    # -- public API ---------------------------------------------------------

    async def judge(self, check_id: str, prompt: str) -> JudgmentResult:
        """Run the binary evaluation and return a :class:`JudgmentResult`.

        The prompt is sent to the LLM :attr:`num_runs` times **concurrently**.
        Each raw response is parsed into a :class:`Verdict` (``YES`` or
        ``NO``).  The final verdict is determined by majority vote.

        Args:
            check_id: Unique identifier for this evaluation check.
            prompt: The user-facing evaluation prompt to send to the judge.

        Returns:
            A :class:`JudgmentResult` containing the aggregated verdict,
            individual votes, unanimity flag, prompt, and raw responses.
        """
        # Fire all majority-vote calls concurrently.
        raw_responses: list[str] = list(await asyncio.gather(*[self._call_llm(prompt) for _ in range(self.num_runs)]))
        votes = [self._parse_vote(raw) for raw in raw_responses]

        yes_count = sum(1 for v in votes if v is Verdict.YES)
        final_verdict = Verdict.YES if yes_count > self.num_runs / 2 else Verdict.NO

        result = JudgmentResult(
            check_id=check_id,
            verdict=final_verdict,
            votes=votes,
            unanimous=len(set(votes)) == 1,
            prompt=prompt,
            raw_responses=raw_responses,
        )
        self._log.append(result)
        return result

    async def judge_across_chunks(
        self,
        check_id: str,
        stored_facts: list[str],
        prompt_builder: Callable[[list[str]], str],
    ) -> JudgmentResult:
        """Evaluate stored facts in chunks concurrently, YES if any chunk matches.

        Splits *stored_facts* into chunks of ``MAX_FACTS_PER_PROMPT`` and
        evaluates all chunks **concurrently**.  Returns ``YES`` as soon as
        any chunk produces a ``YES`` verdict (remaining tasks are cancelled).
        If all chunks return ``NO``, the overall verdict is ``NO``.

        Args:
            check_id: Base identifier for this evaluation check.
            stored_facts: All fact strings to evaluate.
            prompt_builder: A callable that accepts a chunk of fact strings
                and returns the complete prompt for the judge.

        Returns:
            A :class:`JudgmentResult` with the aggregated verdict.
        """
        from cri.scoring.rubrics import MAX_FACTS_PER_PROMPT

        if len(stored_facts) <= MAX_FACTS_PER_PROMPT:
            prompt = prompt_builder(stored_facts)
            return await self.judge(check_id, prompt)

        chunks = [stored_facts[i : i + MAX_FACTS_PER_PROMPT] for i in range(0, len(stored_facts), MAX_FACTS_PER_PROMPT)]

        async def _eval_chunk(idx: int) -> JudgmentResult:
            chunk_id = f"{check_id}_c{idx}"
            prompt = prompt_builder(chunks[idx])
            return await self.judge(chunk_id, prompt)

        # Launch all chunk evaluations concurrently (semaphore throttles).
        tasks = [asyncio.create_task(_eval_chunk(i)) for i in range(len(chunks))]

        yes_result: JudgmentResult | None = None
        last_result: JudgmentResult | None = None

        try:
            for future in asyncio.as_completed(tasks):
                result = await future
                last_result = result
                if result.verdict is Verdict.YES:
                    yes_result = result
                    break  # early exit — cancel the rest below
        finally:
            # Cancel tasks that haven't finished yet.
            for t in tasks:
                if not t.done():
                    t.cancel()
            # Wait for cancellations to propagate (suppress CancelledError).
            await asyncio.gather(*tasks, return_exceptions=True)

        if yes_result is not None:
            return JudgmentResult(
                check_id=check_id,
                verdict=Verdict.YES,
                votes=yes_result.votes,
                unanimous=yes_result.unanimous,
                prompt=yes_result.prompt,
                raw_responses=yes_result.raw_responses,
            )

        # All chunks returned NO.
        assert last_result is not None
        return JudgmentResult(
            check_id=check_id,
            verdict=Verdict.NO,
            votes=last_result.votes,
            unanimous=last_result.unanimous,
            prompt=last_result.prompt,
            raw_responses=last_result.raw_responses,
        )

    def get_log(self) -> list[JudgmentResult]:
        """Return a copy of all judgment results collected so far."""
        return list(self._log)

    def export_log(self, path: Path) -> None:
        """Serialise the judgment log to a JSON file.

        Args:
            path: Destination file path.  Parent directories must exist.
        """
        data = [r.model_dump(mode="json") for r in self._log]
        path.write_text(json.dumps(data, indent=2))

    async def judge_coverage(self, check_id: str, prompt: str) -> set[int]:
        """Ask the judge which GT facts (by 0-based index) are covered.

        Uses the coverage LLM with semaphore rate-limiting and retry.

        Args:
            check_id: Unique identifier for this chunk evaluation.
            prompt: The coverage prompt built by
                :func:`~cri.scoring.rubrics.mei_coverage_chunk_check`.

        Returns:
            A :class:`set` of 0-based integer indices of the GT facts covered
            by the chunk.  Returns an empty set on parse failure or LLM error.
        """
        for attempt in range(_MAX_RETRIES):
            try:
                async with self._semaphore:
                    response = await self._coverage_llm.ainvoke(
                        [
                            SystemMessage(content=COVERAGE_JUDGE_SYSTEM_PROMPT),
                            HumanMessage(content=prompt),
                        ]
                    )
                raw: str = str(response.content) if response.content else "[]"
                return self._parse_coverage_indices(check_id, raw)
            except Exception as exc:
                if attempt < _MAX_RETRIES - 1:
                    delay = _BACKOFF_BASE * (2**attempt)
                    logger.warning(
                        "BinaryJudge coverage call failed (attempt %d/%d), retrying in %.1fs: %s",
                        attempt + 1,
                        _MAX_RETRIES,
                        delay,
                        exc,
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.warning(
                        "BinaryJudge coverage call failed after %d attempts, returning empty set: %s",
                        _MAX_RETRIES,
                        exc,
                    )
        return set()

    @staticmethod
    def _parse_coverage_indices(check_id: str, raw: str) -> set[int]:
        """Parse a JSON array of integers from the coverage LLM response.

        Searches for the first ``[...]`` block in *raw* and parses it as JSON.
        Non-integer values in the array are silently ignored.

        Args:
            check_id: Used only for the warning log on parse failure.
            raw: Raw string response from the coverage LLM.

        Returns:
            A :class:`set` of integer indices, or an empty set if parsing fails.
        """
        import re

        match = re.search(r"\[.*?\]", raw, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
                return {int(i) for i in data if isinstance(i, (int, float))}
            except (json.JSONDecodeError, ValueError, TypeError):
                pass
        logger.warning(
            "BinaryJudge: failed to parse coverage indices for %s from: %r",
            check_id,
            raw,
        )
        return set()

    # -- internals ----------------------------------------------------------

    async def _call_llm(self, user_prompt: str) -> str:
        """Call the LLM once, rate-limited by semaphore, with retry + backoff.

        On transient API errors the call is retried up to ``_MAX_RETRIES``
        times with exponential backoff.  If all retries fail the method
        returns an empty string (which will be parsed as ``Verdict.NO`` by
        :meth:`_parse_vote`).
        """
        for attempt in range(_MAX_RETRIES):
            try:
                async with self._semaphore:
                    response = await self.llm.ainvoke(
                        [
                            SystemMessage(content=BINARY_JUDGE_SYSTEM_PROMPT),
                            HumanMessage(content=user_prompt),
                        ]
                    )
                content: str = str(response.content) if response.content else ""
                return content
            except Exception as exc:
                if attempt < _MAX_RETRIES - 1:
                    delay = _BACKOFF_BASE * (2**attempt)
                    logger.warning(
                        "BinaryJudge LLM call failed (attempt %d/%d), retrying in %.1fs: %s",
                        attempt + 1,
                        _MAX_RETRIES,
                        delay,
                        exc,
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.warning(
                        "BinaryJudge LLM call failed after %d attempts, defaulting to empty response: %s",
                        _MAX_RETRIES,
                        exc,
                    )
        return ""

    @staticmethod
    def _parse_vote(raw: str) -> Verdict:
        """Extract a YES/NO verdict from the first line of a raw LLM response.

        Rules:
        - Extract the first non-empty line.
        - If it contains ``YES`` (case-insensitive) -> :attr:`Verdict.YES`.
        - If it contains ``NO`` (case-insensitive) -> :attr:`Verdict.NO`.
        - Otherwise default to :attr:`Verdict.NO` and emit a warning.
        """
        first_line = ""
        for line in raw.splitlines():
            stripped = line.strip()
            if stripped:
                first_line = stripped
                break

        upper = first_line.upper()
        if "YES" in upper:
            return Verdict.YES
        if "NO" in upper:
            return Verdict.NO

        logger.warning(
            "BinaryJudge: non-binary response, defaulting to NO. First line: %r",
            first_line,
        )
        return Verdict.NO
