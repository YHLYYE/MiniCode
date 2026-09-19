"""Agent Loop core engine — while-true + 5 recovery paths

Reference: Claude Code source architecture (how-claude-code-works analysis)
Core principle: The model is the sole decision-maker. No state machines, no DAGs.
"""

import asyncio
import json
from typing import AsyncIterator

from core.state import (
    Message, LoopState, ContinueReason,
    TextDelta, ToolStart, ToolResult, ToolError, DoneEvent,
    BudgetExceeded,
)
from core.model_adapter import ModelAdapter, StreamChunk, Usage
from core.tools.base import Tool, ToolCall
from capabilities.compression import ContextCompressor
from capabilities.memory import MemoryManager
from capabilities.security import PermissionManager, SecurityBlock


# ── Tool result truncation ──
MAX_RESULT_CHARS = 30_000  # 单条工具结果超过此值则截断（防上下文暴涨）


# ── Agent Loop ──

class AgentLoop:
    """Core agent execution engine.

    Double-layer architecture:
    - run() — outer: session lifecycle, streaming yield
    - _query_loop() — inner: per-turn while-true execution

    The loop operates on a simple principle: gather context → call model →
    execute tools → feed results back → repeat. The model decides when to
    stop (by not calling any more tools).
    """

    def __init__(
        self,
        tools: list[Tool],
        model_adapter: ModelAdapter,
        system_prompt: str,
        max_turns: int = 20,
        max_cost_usd: float = 5.0,
        memory_manager: MemoryManager | None = None,
    ):
        self._tools = {t.name: t for t in tools}
        self._model = model_adapter
        self._system_prompt = system_prompt
        self._max_turns = max_turns
        self._max_cost_usd = max_cost_usd
        self._state: LoopState | None = None
        self._current_task: str | None = None
        self._memory = memory_manager
        self._compressor = ContextCompressor()
        self._permission = PermissionManager(model=model_adapter)

    @property
    def state(self) -> LoopState | None:
        return self._state

    async def run(self, task: str, resume_state: LoopState | None = None):
        """Main entry point — async generator yielding UI events.

        Args:
            task: User's task description
            resume_state: Optional state to continue from (defaults to self._state,
                          enabling multi-task conversation in the same loop)
        """
        self._current_task = task
        if resume_state:
            state = resume_state
        elif self._state is not None and self._state.messages:
            # Continue conversation from previous task (interactive REPL mode)
            state = self._state.add_message(
                Message(role="user", content=task)
            )
        else:
            state = LoopState(
                messages=(
                    Message(role="system", content=self._system_prompt),
                    Message(role="user", content=task),
                ),
                transition=ContinueReason.NEXT_TURN,
            )

        async for event in self._query_loop(state):
            yield event

    async def _query_loop(self, state: LoopState):
        """Inner loop: per-turn execution with 5 recovery paths."""
        self._state = state

        while self._state.turn_count < self._max_turns:
            # Increment turn counter (LoopState is immutable → replace)
            self._state = self._state.with_field(
                turn_count=self._state.turn_count + 1
            )

            # ── 0. Context compression check ──
            self._state = await self._compressor.compress_if_needed(self._state)

            # ── 1. Build API request ──
            tool_schemas = [t.to_schema() for t in self._tools.values()]
            messages_list = []
            for m in self._state.messages:
                entry: dict = {"role": m.role, "content": m.content}
                if m.tool_call_id:
                    entry["tool_call_id"] = m.tool_call_id
                if m.tool_calls:
                    entry["tool_calls"] = m.tool_calls
                messages_list.append(entry)

            # ── 2. Streaming API call ──
            system_text = (
                self._state.messages[0].content
                if self._state.messages[0].role == "system"
                else ""
            )
            stream = self._model.chat_streaming(
                messages=messages_list,
                system=system_text,
                tools=tool_schemas,
            )

            assistant_text_parts: list[str] = []
            tool_calls: list[tuple[str, dict, str]] = []  # (name, input, tool_call_id)
            usage = Usage()
            stop_reason = "end_turn"

            try:
                async for chunk in stream:
                    if chunk.type == "text_delta":
                        assistant_text_parts.append(chunk.text)
                        yield TextDelta(chunk.text)
                    elif chunk.type == "tool_use_start":
                        tool_calls.append((chunk.name, chunk.input or {}, chunk.tool_call_id))
                        yield ToolStart(chunk.name)
                    elif chunk.type == "message_stop":
                        if chunk.usage:
                            usage = chunk.usage
                        if chunk.stop_reason:
                            stop_reason = chunk.stop_reason
            except Exception as e:
                if self._is_prompt_too_long(e):
                    # Recovery: context too long → force compact → retry
                    self._state = await self._compressor.force_autocompact(self._state)
                    self._state = self._state.with_transition(
                        ContinueReason.PROMPT_TOO_LONG_RETRY
                    )
                    continue
                raise

            # ── 3. Track costs (fail-fast on budget) ──
            self._state = self._state.accumulate_usage(usage)
            if self._state.total_cost_usd >= self._max_cost_usd:
                raise BudgetExceeded(
                    f"Budget exceeded: ${self._state.total_cost_usd:.4f} "
                    f"> ${self._max_cost_usd:.2f}"
                )

            # ── 3.5 Recovery: output token limit hit (finish_reason == "length") ──
            if stop_reason in ("length", "max_tokens"):
                handled = await self._handle_max_tokens()
                if not handled:
                    await self._record_task_done("terminated: max_tokens")
                    yield DoneEvent(self._state)
                    return
                continue

            # ── 4. Add assistant message (with tool_calls for OpenAI format) ──
            assistant_text = "".join(assistant_text_parts)
            # Build tool_calls attachment for OpenAI/DeepSeek compatibility
            tool_calls_attachments = [
                {"id": tc_id, "type": "function",
                 "function": {"name": tc_name, "arguments": json.dumps(tc_input, ensure_ascii=False)}}
                for tc_name, tc_input, tc_id in tool_calls if tc_id
            ]
            if assistant_text or tool_calls_attachments:
                self._state = self._state.add_message(
                    Message(role="assistant", content=assistant_text,
                            tool_calls=tool_calls_attachments)
                )

            # ── 5. No tool calls → task complete ──
            if not tool_calls and assistant_text:
                self._state = self._state.with_transition(None)
                await self._record_task_done(assistant_text[:300])
                yield DoneEvent(self._state)
                return

            # ── 6. Execute tools sequentially ──
            tool_errors: list[ToolError] = []
            for tool_name, tool_input, tool_call_id in tool_calls:
                tool = self._tools.get(tool_name)
                if tool is None:
                    result = f"Error: Tool '{tool_name}' not found. Available: {list(self._tools.keys())}"
                    yield ToolResult(tool_name, result)
                else:
                    try:
                        # Security check before execution
                        tc = ToolCall(tool_name, tool_input)
                        await self._permission.authorize(tc)
                        result = await tool.execute(**tool_input)
                        # 截断超长结果，防止单条工具输出塞爆上下文
                        if len(result) > MAX_RESULT_CHARS:
                            original = len(result)
                            result = (
                                result[:MAX_RESULT_CHARS - 1000]
                                + f"\n...[省略 {original - MAX_RESULT_CHARS} 字符]...\n"
                                + result[-500:]
                            )
                        yield ToolResult(tool_name, result)
                    except SecurityBlock as e:
                        result = f"Security blocked: {e}"
                        yield ToolResult(tool_name, result)
                    except Exception as e:
                        result = f"Tool error: {e}"
                        tool_errors.append(ToolError(tool_name, str(e)))
                        yield ToolResult(tool_name, result)

                # ── 6.1 集成回调：追踪 active_skills / 最近编辑文件 ──
                # Skill 激活成功 → 记入 active_skills，压缩恢复时可回灌
                if tool_name == "Skill" and not result.startswith(("Skill system not", "Skill '", "No skill ")):
                    skill_name_val = tool_input.get("name", "")
                    if skill_name_val and skill_name_val not in self._state.active_skills:
                        current = list(self._state.active_skills)
                        current.append(skill_name_val)
                        self._state = self._state.with_field(
                            active_skills=tuple(current[-10:])  # 保留最近 10 个
                        )
                # Write 成功 → 通知 ContextCompressor，Autocompact 时能回灌
                if tool_name == "Write":
                    file_path = tool_input.get("file_path", "")
                    content = tool_input.get("content", "")
                    if file_path and content is not None:
                        # 结果前缀 "Created" 或 "Updated" 才视为成功写入
                        if result.startswith(("Created ", "Updated ")):
                            self._compressor.record_edit(file_path, content)
                            if self._memory is not None:
                                await self._memory.record_file_edit(
                                    file_path, "", ""
                                )

                self._state = self._state.add_message(
                    Message(role="tool", content=result, tool_call_id=tool_call_id)
                )

            # ── 7. Error recovery: inject error context and retry ──
            if tool_errors:
                error_lines = "\n".join(
                    f"- {e.tool_name}: {e.error}" for e in tool_errors
                )
                if self._memory is not None:
                    await self._memory.record_episodic(
                        f"Tool errors:\n{error_lines[:500]}",
                        tags=["error"],
                    )
                recovery_msg = Message(
                    role="user",
                    content=(
                        f"Some tool calls failed:\n{error_lines}\n\n"
                        "Analyze the errors and try an alternative approach. "
                        "Do NOT retry the exact same failed calls."
                    ),
                )
                self._state = self._state.add_message(recovery_msg)
                self._state = self._state.with_transition(ContinueReason.NEXT_TURN)
                continue

            # ── 8. Continue to next turn ──
            self._state = self._state.with_transition(ContinueReason.NEXT_TURN)

        # Max turns reached
        await self._record_task_done("terminated: max_turns")
        yield DoneEvent(self._state)

    async def _handle_max_tokens(self) -> bool:
        """Recovery path: output token limit hit.

        Three-tier escalation (returns False when exhausted → caller ends):
        1. Silent upgrade: max_output_tokens 8K → 64K, retry once
        2. Continuation prompt: already at 64K → inject "resume" message
           (up to 3 retries)
        3. Give up: all recovery attempts exhausted
        """
        ESCALATED_MAX_TOKENS = 64_000
        MAX_RECOVERY_RETRIES = 3

        # Tier 1: silent upgrade
        if self._model.max_output_tokens < ESCALATED_MAX_TOKENS:
            self._model.max_output_tokens = ESCALATED_MAX_TOKENS
            self._state = self._state.with_transition(
                ContinueReason.MAX_OUTPUT_TOKENS_UPGRADE
            )
            return True

        # Tier 2: continuation prompt
        if self._state.max_output_tokens_recovery < MAX_RECOVERY_RETRIES:
            self._state = self._state.add_message(Message(
                role="user",
                content=(
                    "Output token limit hit. Resume directly — no apology, "
                    "no recap. Pick up mid-thought if cut off. Break remaining "
                    "work into smaller pieces."
                ),
            ))
            self._state = self._state.with_field(
                max_output_tokens_recovery=self._state.max_output_tokens_recovery + 1
            )
            self._state = self._state.with_transition(
                ContinueReason.MAX_OUTPUT_TOKENS_RECOVERY
            )
            return True

        # Tier 3: give up
        return False

    async def _record_task_done(self, note: str = "") -> None:
        """Persist a task-completion event to episodic memory.

        Called at every DoneEvent site so memory holds a session-level
        trace of what happened, not just isolated tool events.
        """
        if self._memory is None:
            return
        task = self._current_task or "(unnamed task)"
        suffix = f"\n{note}" if note else ""
        await self._memory.record_episodic(
            f"Task: {task} | {self._state.turn_count} turns | "
            f"{self._state.total_tokens} tokens{suffix}",
            tags=["task_summary"],
        )

    def _is_prompt_too_long(self, error: Exception) -> bool:
        """Detect whether an API error indicates context overflow."""
        msg = str(error).lower()
        return any(
            keyword in msg
            for keyword in (
                "too long", "maximum context", "context length",
                "prompt is too long", "context_window", "max tokens",
                "400",  # OpenAI/DeepSeek return 400 for context overflow
            )
        )
