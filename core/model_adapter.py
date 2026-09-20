"""Multi-provider model adapter — isolates provider differences from Agent Loop

Uses litellm (borrowed from mini-swe-agent's model-agnostic design) to support
100+ models through one unified interface: OpenAI, Anthropic, DeepSeek, Gemini,
open-weight and local models — anything litellm routes.

Model name format: "provider/model" (e.g. "deepseek/deepseek-chat",
"anthropic/claude-sonnet-4-6"). Bare names are auto-prefixed from
MINICODE_PROVIDER or name heuristics.
"""

import json
import os
from dataclasses import dataclass
from typing import AsyncIterator


@dataclass
class Usage:
    """Token usage statistics from API response"""
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0


@dataclass
class StreamChunk:
    """Unified streaming event — normalizes across all providers"""
    type: str  # "text_delta" | "tool_use_start" | "message_stop"
    text: str = ""
    name: str = ""
    input: dict | None = None
    tool_call_id: str = ""
    stop_reason: str = ""
    usage: Usage | None = None


class ModelAdapter:
    """Unified LLM interface backed by litellm.

    The Agent Loop only sees clean StreamChunk events — never raw provider
    chunks. Swapping models is a one-line config change (model name), and
    the provider-specific protocol differences are hidden inside litellm.
    """

    def __init__(self, model: str, api_key: str | None = None):
        self.model = self._resolve_model(model)
        self.max_output_tokens = 8192
        # litellm reads provider keys from env (DEEPSEEK_API_KEY, ANTHROPIC_API_KEY,
        # OPENAI_API_KEY, ...). Explicit api_key override is passed through only
        # if set — otherwise rely on environment.
        self._api_key = api_key

    def _resolve_model(self, model: str) -> str:
        """Normalize 'deepseek-chat' → 'deepseek/deepseek-chat' (litellm format)."""
        if "/" in model:
            return model  # already "provider/model"

        provider = os.environ.get("MINICODE_PROVIDER", "").lower()
        if not provider:
            m = model.lower()
            if "deepseek" in m:
                provider = "deepseek"
            elif "claude" in m or "anthropic" in m:
                provider = "anthropic"
            elif "gpt" in m or "openai" in m or "o1" in m or "o3" in m:
                provider = "openai"
            elif "gemini" in m:
                provider = "gemini"
            else:
                provider = "deepseek"  # default (cheapest, no-key friction)
        return f"{provider}/{model}"

    def _convert_tools(self, tools: list[dict] | None) -> list[dict] | None:
        """Convert Anthropic-format tool schemas to OpenAI function-calling format.

        litellm uses the OpenAI format universally, so this conversion runs for
        every provider (Anthropic included).
        """
        if not tools:
            return None
        openai_tools = []
        for t in tools:
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": {
                        "type": "object",
                        "properties": t.get("input_schema", {}).get("properties", {}),
                        "required": t.get("input_schema", {}).get("required", []),
                    },
                },
            })
        return openai_tools

    async def chat(
        self,
        messages: list,
        system: str = "",
        max_tokens: int | None = None,
    ) -> str:
        """Non-streaming chat — used by the security classifier and summarizer."""
        import litellm

        input_msgs = list(messages)
        if system:
            input_msgs = [{"role": "system", "content": system}] + input_msgs

        kwargs = {
            "model": self.model,
            "messages": input_msgs,
            "max_tokens": max_tokens or 1024,
        }
        if self._api_key:
            kwargs["api_key"] = self._api_key

        response = await litellm.acompletion(**kwargs)
        content = response.choices[0].message.content
        return content or ""

    async def chat_streaming(
        self,
        messages: list,
        system: str = "",
        tools: list[dict] | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Stream chat response as unified StreamChunk events (real-time tokens)."""
        import litellm

        input_msgs = list(messages)
        if system:
            input_msgs = [{"role": "system", "content": system}] + input_msgs

        kwargs = {
            "model": self.model,
            "messages": input_msgs,
            "max_tokens": max_tokens or self.max_output_tokens,
            "stream": True,
            # include_usage makes OpenAI-compatible APIs return token counts
            # on the final stream chunk (needed for cost tracking).
            "stream_options": {"include_usage": True},
        }
        if tools:
            kwargs["tools"] = self._convert_tools(tools)
        if self._api_key:
            kwargs["api_key"] = self._api_key

        async for chunk in self._stream_litellm(kwargs):
            yield chunk

    async def _stream_litellm(self, kwargs: dict) -> AsyncIterator[StreamChunk]:
        """Parse litellm streaming chunks (OpenAI format) → unified StreamChunk."""
        import litellm

        accumulated_tool_calls: dict[int, dict] = {}  # index → {id, name, args}
        finish_reason = "stop"
        usage_data: dict = {}

        response = await litellm.acompletion(**kwargs)
        async for chunk in response:
            # 3. Usage — 必须在 choices 检查之前捕获：
            #    include_usage 的最终 chunk 是 choices=[] 但带 usage，
            #    若先 `if not chunk.choices: continue` 会把用量丢掉（计费归零）。
            if getattr(chunk, "usage", None):
                usage_data = {
                    "input_tokens": chunk.usage.prompt_tokens or 0,
                    "output_tokens": chunk.usage.completion_tokens or 0,
                }

            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta

            # 1. Text delta → yield immediately
            if delta.content:
                yield StreamChunk(type="text_delta", text=delta.content)

            # 2. Tool call delta → accumulate
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in accumulated_tool_calls:
                        accumulated_tool_calls[idx] = {"id": "", "name": "", "args": ""}
                    entry = accumulated_tool_calls[idx]
                    if tc.id:
                        entry["id"] = tc.id
                    if tc.function and tc.function.name:
                        entry["name"] = tc.function.name
                    if tc.function and tc.function.arguments:
                        entry["args"] += tc.function.arguments

            # 3. Finish reason (final non-empty chunk)
            if chunk.choices[0].finish_reason:
                finish_reason = chunk.choices[0].finish_reason

        # After stream: yield accumulated tool calls
        for idx, tc in accumulated_tool_calls.items():
            if tc["name"]:
                args_str = tc["args"]
                try:
                    parsed_args = json.loads(args_str) if args_str else {}
                except (json.JSONDecodeError, TypeError):
                    parsed_args = {}
                yield StreamChunk(
                    type="tool_use_start",
                    name=tc["name"],
                    input=parsed_args,
                    tool_call_id=tc["id"] or f"call_auto_{idx}",
                )

        # Yield message stop
        yield StreamChunk(
            type="message_stop",
            stop_reason="end_turn" if finish_reason == "stop" else finish_reason,
            usage=Usage(
                input_tokens=usage_data.get("input_tokens", 0),
                output_tokens=usage_data.get("output_tokens", 0),
            ),
        )
