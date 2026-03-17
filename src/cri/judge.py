"""LLM-as-judge evaluation module.

Uses a large language model to evaluate memory system responses against
expected answers, producing scored judgments for each evaluation dimension.

The judge approach enables semantic evaluation that goes beyond simple
string matching, capturing the nuances of contextual understanding.

Provides :class:`BinaryJudge` — a binary YES/NO judge with majority-vote
robustness (synchronous, used by the CRI evaluation pipeline).
"""

from __future__ import annotations

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

# Type alias for the LLM factory callable.
# Receives (temperature, max_tokens) and returns a configured BaseChatModel.
LLMFactory = Callable[[float, int], BaseChatModel]


def create_default_llm(temperature: float = 0.0, max_tokens: int = 10) -> BaseChatModel:
    """Create the default LLM for the BinaryJudge.

    Uses :class:`AnthropicOAuthChat` with an OAuth subscription token
    read from ``../.auth_token``.  Override by passing a custom
    ``llm_factory`` to :class:`BinaryJudge`.

    Args:
        temperature: Sampling temperature.
        max_tokens: Maximum tokens in each LLM response.

    Returns:
        A configured :class:`BaseChatModel` instance.

    Raises:
        RuntimeError: If the token file is not found.
    """
    from cri.utils.llm_anthropic_subscription import AnthropicOAuthChat

    token_path = Path("../.auth_token")
    if not token_path.exists():
        raise RuntimeError(f"Auth token not found at {token_path.resolve()}. Place your Anthropic OAuth token in ../.auth_token")
    return AnthropicOAuthChat(
        token_path=str(token_path),
        model_name="claude-haiku-4-5-20251001",
        max_tokens=max_tokens,
        temperature=temperature,
    )


# ---------------------------------------------------------------------------
# Binary YES/NO judge with majority-vote robustness
# ---------------------------------------------------------------------------


class BinaryJudge:
    """Binary YES/NO LLM judge with majority-vote robustness.

    Sends the same evaluation prompt to an LLM multiple times and aggregates
    the responses via majority vote.  This reduces noise inherent in single
    LLM calls and produces a more reliable binary verdict.

    Args:
        llm_factory: A callable ``(temperature, max_tokens) -> BaseChatModel``
            that creates the LLM instance.  Defaults to
            :func:`create_default_llm` (Anthropic OAuth).
        num_runs: Number of independent LLM calls per judgment
            (default ``3``).  Must be an odd positive integer for a clean
            majority.
        temperature: Sampling temperature (default ``0.0`` for
            deterministic output).  Passed to the factory.
        max_tokens: Maximum tokens in each LLM response (default ``10``).
            Passed to the factory.
    """

    def __init__(
        self,
        llm_factory: LLMFactory = create_default_llm,
        num_runs: int = 3,
        temperature: float = 0.0,
        max_tokens: int = 10,
    ) -> None:
        self.num_runs = num_runs
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.llm: BaseChatModel = llm_factory(temperature, max_tokens)
        self._log: list[JudgmentResult] = []

    # -- public API ---------------------------------------------------------

    def judge(self, check_id: str, prompt: str) -> JudgmentResult:
        """Run the binary evaluation and return a :class:`JudgmentResult`.

        The prompt is sent to the LLM :attr:`num_runs` times.  Each raw
        response is parsed into a :class:`Verdict` (``YES`` or ``NO``).
        The final verdict is determined by majority vote.

        Args:
            check_id: Unique identifier for this evaluation check.
            prompt: The user-facing evaluation prompt to send to the judge.

        Returns:
            A :class:`JudgmentResult` containing the aggregated verdict,
            individual votes, unanimity flag, prompt, and raw responses.
        """
        votes: list[Verdict] = []
        raw_responses: list[str] = []

        for _ in range(self.num_runs):
            raw = self._call_llm(prompt)
            raw_responses.append(raw)
            votes.append(self._parse_vote(raw))

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

    # -- internals ----------------------------------------------------------

    def _call_llm(self, user_prompt: str) -> str:
        """Call the LLM once with a single retry on failure.

        On the first API error the call is retried once.  If the retry also
        fails the method returns an empty string (which will be parsed as
        ``Verdict.NO`` by :meth:`_parse_vote`).
        """
        for attempt in range(2):  # initial + 1 retry
            try:
                response = self.llm.invoke(
                    [
                        SystemMessage(content=BINARY_JUDGE_SYSTEM_PROMPT),
                        HumanMessage(content=user_prompt),
                    ]
                )
                raw_content = response.content or ""
                content: str = raw_content if isinstance(raw_content, str) else str(raw_content)
                return content
            except Exception as exc:
                if attempt == 0:
                    logger.warning(
                        "BinaryJudge LLM call failed (attempt 1/2), retrying: %s",
                        exc,
                    )
                else:
                    logger.warning(
                        "BinaryJudge LLM call failed after retry, defaulting to empty response: %s",
                        exc,
                    )
        return ""

    @staticmethod
    def _parse_vote(raw: str) -> Verdict:
        """Extract a YES/NO verdict from the first line of a raw LLM response.

        Rules:
        - Extract the first non-empty line.
        - If it contains ``YES`` (case-insensitive) → :attr:`Verdict.YES`.
        - If it contains ``NO`` (case-insensitive) → :attr:`Verdict.NO`.
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
