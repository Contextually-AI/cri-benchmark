"""Custom BaseChatModel using Anthropic OAuth tokens (Claude CLI / OpenClaw).

OAuth tokens (``sk-ant-oat01-...``) require special headers and a mandatory
system prompt preamble.  The token is read from a file and refreshed
automatically on 401 responses.

When tools are bound via :meth:`AnthropicOAuthChat.bind_tools`, tool calling
is **simulated**: tool descriptions are injected into the system prompt and the
model's text response is parsed for ``<tool_call>`` XML blocks.
"""

from __future__ import annotations

import asyncio
import json
import re
import uuid
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import httpx
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import PrivateAttr

API_URL = "https://api.anthropic.com/v1/messages"
REQUIRED_PREAMBLE = "You are Claude Code, Anthropic's official CLI for Claude."
TOOL_CALL_PATTERN = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
    re.DOTALL,
)


class AnthropicOAuthChat(BaseChatModel):
    """Chat model that authenticates via an Anthropic OAuth subscription token."""

    token_path: str
    model_name: str = "claude-sonnet-4-6"
    max_tokens: int = 4096
    temperature: float | None = None

    _token: str | None = PrivateAttr(default=None)
    _bound_tools: list[dict[str, Any]] = PrivateAttr(default_factory=list)

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        self._token = self._read_token()

    # ------------------------------------------------------------------
    # Token management
    # ------------------------------------------------------------------

    def _read_token(self) -> str:
        return Path(self.token_path).read_text().strip()

    def _refresh_token(self) -> str:
        self._token = self._read_token()
        return self._token

    # ------------------------------------------------------------------
    # Tool calling simulation
    # ------------------------------------------------------------------

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | Any],
        **kwargs: Any,
    ) -> AnthropicOAuthChat:
        """Bind tools for simulated tool calling.

        Unlike native tool calling, tools are described in the system prompt
        and the model's text response is parsed for ``<tool_call>`` XML
        blocks.  Returns a **copy** of this model with the tools stored.
        """
        from langchain_core.utils.function_calling import convert_to_openai_tool

        formatted = [convert_to_openai_tool(t) for t in tools]

        copy = self.model_copy()
        copy._bound_tools = formatted
        copy._token = self._token
        return copy

    @staticmethod
    def _build_temperature_prompt(temperature: float) -> str:
        """Build a system prompt section describing expected behavior for the given temperature."""
        t = max(0.0, min(1.0, temperature))
        if t <= 0.3:
            creativity = "low"
            description = (
                "Your temperature is set to a low value "
                f"({t:.1f} out of 1.0). This means you should be very "
                "precise, deterministic, and focused in your responses. "
                "Prioritize accuracy and consistency over creativity. "
                "Avoid rambling or exploring unnecessary alternatives. "
                "Be direct and stick strictly to facts and the most "
                "likely information."
            )
        elif t <= 0.7:
            creativity = "moderate"
            description = (
                "Your temperature is set to a medium value "
                f"({t:.1f} out of 1.0). This means you should maintain "
                "a balance between precision and creativity. You may "
                "explore some alternatives when useful, but without "
                "losing focus. Be clear and coherent, allowing yourself "
                "some flexibility in style and suggestions."
            )
        else:
            creativity = "high"
            description = (
                "Your temperature is set to a high value "
                f"({t:.1f} out of 1.0). This means you should be more "
                "creative, exploratory, and diverse in your responses. "
                "Feel free to propose original ideas, explore alternative "
                "approaches, and use a more expressive style. Variety "
                "and originality are encouraged."
            )
        return f"[Creativity: {creativity}] {description}"

    @staticmethod
    def _build_max_tokens_prompt(max_tokens: int) -> str:
        """Build a system prompt section describing response length expectations."""
        if max_tokens <= 500:
            length = "brief"
            description = (
                f"The token limit is set to {max_tokens}. "
                "This is a low value, so you should be very concise "
                "and brief in your responses. Get straight to the point, "
                "use short sentences, and avoid lengthy explanations. "
                "Prioritize the most important information and omit "
                "secondary details."
            )
        elif max_tokens <= 2000:
            length = "moderate"
            description = (
                f"The token limit is set to {max_tokens}. "
                "This allows moderately-sized responses. You may include "
                "reasonable explanations and necessary context, but avoid "
                "being excessively verbose. Maintain a balance between "
                "completeness and conciseness."
            )
        else:
            length = "extended"
            description = (
                f"The token limit is set to {max_tokens}. "
                "This allows detailed and comprehensive responses. You "
                "may elaborate on explanations, include examples, break "
                "down steps, and provide broad context when useful. Take "
                "advantage of the available space to give thorough "
                "answers when the topic warrants it."
            )
        return f"[Expected length: {length}] {description}"

    @staticmethod
    def _build_tool_prompt(tools: list[dict[str, Any]]) -> str:
        """Build a system prompt section describing available tools."""
        lines = [
            "You have access to the following tools. To call a tool, you MUST "
            "use the exact XML format shown below. Do NOT paraphrase or "
            "deviate from this format.",
            "",
            "Available tools:",
        ]
        for tool_def in tools:
            func = tool_def.get("function", tool_def)
            name = func["name"]
            desc = func.get("description", "No description.")
            params = func.get("parameters", {})
            props = params.get("properties", {})
            required = params.get("required", [])

            lines.append("")
            lines.append(f"### {name}")
            lines.append(f"Description: {desc}")
            if props:
                lines.append("Parameters:")
                for pname, pschema in props.items():
                    ptype = pschema.get("type", "any")
                    pdesc = pschema.get("description", "")
                    req = " (required)" if pname in required else " (optional)"
                    lines.append(f"  - {pname} ({ptype}{req}): {pdesc}")

        lines.append("")
        lines.append("When you want to call a tool, output EXACTLY this XML block (you may call multiple tools by including multiple blocks):")
        lines.append("")
        lines.append("<tool_call>")
        lines.append('{"name": "<tool_name>", "arguments": {<JSON arguments>}}')
        lines.append("</tool_call>")
        lines.append("")
        lines.append(
            "IMPORTANT: The JSON inside <tool_call> must be valid JSON. "
            'Always include both "name" and "arguments" keys. '
            "If a tool requires no arguments, use an empty object: "
            '{"name": "tool_name", "arguments": {}}.'
        )
        lines.append("If you do not need to call any tools, respond normally without any <tool_call> blocks.")
        return "\n".join(lines)

    @staticmethod
    def _parse_tool_calls(text: str) -> tuple[str, list[dict[str, Any]]]:
        """Parse ``<tool_call>`` blocks from LLM response text.

        Returns ``(remaining_text, tool_calls)`` where *tool_calls* is a list
        of dicts with keys ``name``, ``args``, ``id``, ``type``.
        """
        tool_calls: list[dict[str, Any]] = []

        for match in TOOL_CALL_PATTERN.finditer(text):
            json_str = match.group(1)
            try:
                parsed = json.loads(json_str)
                name = parsed.get("name", "")
                arguments = parsed.get("arguments", {})
                if not isinstance(arguments, dict):
                    arguments = {}
                tool_calls.append(
                    {
                        "name": name,
                        "args": arguments,
                        "id": f"simulated_{uuid.uuid4().hex[:12]}",
                        "type": "tool_call",
                    }
                )
            except (json.JSONDecodeError, KeyError):
                continue

        remaining = text
        if tool_calls:
            remaining = TOOL_CALL_PATTERN.sub("", text).strip()

        return remaining, tool_calls

    # ------------------------------------------------------------------
    # Message conversion
    # ------------------------------------------------------------------

    def _convert_messages(
        self,
        messages: list[BaseMessage],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Split LangChain messages into Anthropic ``system`` and ``messages``."""
        system_blocks: list[dict[str, Any]] = [
            {"type": "text", "text": REQUIRED_PREAMBLE},
        ]
        api_messages: list[dict[str, Any]] = []

        # Inject temperature and max_tokens behavior hints
        if self.temperature is not None:
            temp_prompt = self._build_temperature_prompt(self.temperature)
            system_blocks.append({"type": "text", "text": temp_prompt})
        max_tokens_prompt = self._build_max_tokens_prompt(self.max_tokens)
        system_blocks.append({"type": "text", "text": max_tokens_prompt})

        # Inject tool descriptions when tools are bound
        if self._bound_tools:
            tool_prompt = self._build_tool_prompt(self._bound_tools)
            system_blocks.append({"type": "text", "text": tool_prompt})

        for msg in messages:
            if isinstance(msg, SystemMessage):
                system_blocks.append({"type": "text", "text": msg.content})
            elif isinstance(msg, HumanMessage):
                api_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                if getattr(msg, "tool_calls", None):
                    # Reconstruct <tool_call> blocks so the LLM sees
                    # consistent history for multi-turn tool loops.
                    parts: list[str] = []
                    if msg.content:
                        parts.append(str(msg.content))
                    for tc in msg.tool_calls:
                        call_json = json.dumps(
                            {"name": tc["name"], "arguments": tc["args"]},
                            ensure_ascii=False,
                        )
                        parts.append(f"<tool_call>\n{call_json}\n</tool_call>")
                    api_messages.append(
                        {
                            "role": "assistant",
                            "content": "\n".join(parts),
                        }
                    )
                else:
                    api_messages.append({"role": "assistant", "content": msg.content})
            elif isinstance(msg, ToolMessage):
                tool_call_id = getattr(msg, "tool_call_id", "unknown")
                tool_name = getattr(msg, "name", None) or "unknown_tool"
                result_text = f'<tool_result tool_call_id="{tool_call_id}" name="{tool_name}">\n{msg.content}\n</tool_result>'
                api_messages.append({"role": "user", "content": result_text})
            else:
                # Fallback – treat as user
                api_messages.append({"role": "user", "content": msg.content})

        return system_blocks, api_messages

    # ------------------------------------------------------------------
    # Headers
    # ------------------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._token}",
            "anthropic-version": "2023-06-01",
            "anthropic-beta": "claude-code-20250219,oauth-2025-04-20",
            "user-agent": "claude-cli/2.1.2 (external, cli)",
            "x-app": "cli",
            "anthropic-dangerous-direct-browser-access": "true",
            "accept": "application/json",
        }

    # ------------------------------------------------------------------
    # BaseChatModel interface
    # ------------------------------------------------------------------

    @property
    def _llm_type(self) -> str:
        return "anthropic-oauth"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop and loop.is_running():
            # Nested event loop — create a new one in a thread
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(
                    asyncio.run,
                    self._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs),
                ).result()
        return asyncio.run(self._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs))

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        system_blocks, api_messages = self._convert_messages(messages)
        body: dict[str, Any] = {
            "model": self.model_name,
            "max_tokens": self.max_tokens,
            "system": system_blocks,
            "messages": api_messages,
        }
        if self.temperature is not None:
            body["temperature"] = self.temperature
        if stop:
            body["stop_sequences"] = stop

        result = await self._post(body)
        return result

    async def _post(self, body: dict[str, Any], *, retried: bool = False) -> ChatResult:
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(API_URL, headers=self._headers(), json=body)

            if resp.status_code == 401 and not retried:
                self._refresh_token()
                return await self._post(body, retried=True)

            resp.raise_for_status()
            data = resp.json()

        text = ""
        for block in data.get("content", []):
            if block.get("type") == "text":
                text += block["text"]

        # When tools are bound, check for simulated tool call patterns.
        if self._bound_tools:
            remaining, tool_calls = self._parse_tool_calls(text)
            message = AIMessage(content=remaining, tool_calls=tool_calls) if tool_calls else AIMessage(content=text)
        else:
            message = AIMessage(content=text)

        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])
