"""Context compression — 3-tier degradation chain.

Reference: Claude Code's compression pipeline (how-claude-code-works ch03).

Token usage thresholds:
  70% → Snip: large old tool outputs → placeholder replacement
  85% → Collapse: middle messages → structured summary
  95% → Autocompact: full session summary (last resort)

Note: single-result truncation (Claude Code's 50% tier) is handled at tool
execution in agent_loop.py via MAX_RESULT_CHARS, not in this compressor.
"""

import hashlib
import tiktoken
from dataclasses import dataclass

from core.state import LoopState, Message

# Token counter — 近似计量，用 OpenAI 的 cl100k_base。
#
# 为什么是"近似"：真正跑的模型可能是 DeepSeek / Claude，它们各有自己的分词器。
# 实测同一句中文，cl100k 比 XLM-R 系（bge-m3 / bge-reranker 同族词表）多 50%~90%
# （短句 19 vs 10、长句 27 vs 18），中英混排几乎相等（25 vs 25）。
# 方向是**保守**的：中文长会话会略微提前触发压缩，不会算少了把窗口撑爆；
# 真的超了还有 PROMPT_TOO_LONG_RETRY 兜底。
# 早先这里写的是 "cl100k_base (Claude + GPT compatible)" —— 不准确，Claude 有自己的分词器。
_ENCODER = tiktoken.get_encoding("cl100k_base")


def count_tokens(messages: tuple[Message, ...]) -> int:
    """Accurate token count for conversation messages."""
    total = 0
    for msg in messages:
        total += 4  # per-message format overhead
        total += len(_ENCODER.encode(msg.content))
    return total


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _strip_orphaned_tool_calls(messages: tuple[Message, ...]) -> tuple[Message, ...]:
    """压缩边界可能切断 assistant.tool_calls 与其 tool 结果的配对。

    OpenAI/DeepSeek 要求：assistant 带 tool_calls 时，其后必须紧跟对应的
    tool 消息（否则 API 400）。跨压缩边界时：
    - assistant 的 tool 结果被压掉 → 去掉该 assistant 的 tool_calls
    - tool 消息对应的 assistant 被压掉 → 丢掉该 tool 消息
    """
    tool_result_ids = {
        m.tool_call_id for m in messages
        if m.role == "tool" and m.tool_call_id
    }
    referenced_ids = {
        tc.get("id")
        for m in messages if m.role == "assistant" and m.tool_calls
        for tc in m.tool_calls if tc.get("id")
    }
    out: list[Message] = []
    for m in messages:
        if m.role == "assistant" and m.tool_calls:
            kept = [tc for tc in m.tool_calls if tc.get("id") in tool_result_ids]
            out.append(Message(
                role="assistant", content=m.content, tool_calls=kept or None,
            ))
        elif m.role == "tool":
            if m.tool_call_id in referenced_ids:
                out.append(m)
        else:
            out.append(m)
    return tuple(out)


def summarize_messages(messages: tuple[Message, ...]) -> str:
    """规则式摘要：从消息历史里抽出「改过哪些文件 / 跑过哪些命令 / 出过哪些错」。

    纯本地、零 API 成本 —— 压缩触发时上下文已经满了，调不动模型；
    会话结束时用它生成跨会话摘要也一样。
    """
    file_tools: list[str] = []
    commands: list[str] = []
    errors: list[str] = []

    for msg in messages:
        content = msg.content[:500]
        if any(word in content for word in ["Created", "Updated", "Edited"]):
            file_tools.append(content[:200])
        elif "exit code" in content:
            commands.append(content[:200])
        elif "Error" in content or "error" in content.lower():
            errors.append(content[:200])

    parts = []
    if file_tools:
        recent = file_tools[-10:]
        parts.append(
            f"Files modified ({len(file_tools)} total):\n"
            + "\n".join(f"  - {f}" for f in recent)
        )
    if commands:
        recent = commands[-5:]
        parts.append(
            f"Commands run ({len(commands)} total):\n"
            + "\n".join(f"  - {c}" for c in recent)
        )
    if errors:
        recent = errors[-5:]
        parts.append(
            f"Errors ({len(errors)} total):\n"
            + "\n".join(f"  - {e}" for e in recent)
        )

    return "\n\n".join(parts) if parts else f"({len(messages)} messages processed)"


@dataclass
class CompressionConfig:
    max_context_tokens: int = 60_000  # DeepSeek V3 64K window, leave 4K margin
    snip_threshold_ratio: float = 0.70
    collapse_threshold_ratio: float = 0.85
    autocompact_threshold_ratio: float = 0.95
    autocompact_circuit_breaker: int = 3
    restore_recent_edits: int = 5


class ContextCompressor:
    """Manages context window budget through progressive compression.

    Each tier is triggered at a specific token usage ratio. Tiers are
    applied in order (lightest → heaviest). The compressor never removes
    the System Prompt or the most recent messages.
    """

    def __init__(self, config: CompressionConfig | None = None):
        self._config = config or CompressionConfig()
        self._cache: dict[str, str] = {}
        self._recent_edits: list[tuple[str, str]] = []
        self._task_list: str = ""

    def record_edit(self, file_path: str, content: str):
        """Track recent file edits for post-compression restoration."""
        self._recent_edits.append((file_path, content))
        max_edits = self._config.restore_recent_edits
        if len(self._recent_edits) > max_edits:
            self._recent_edits = self._recent_edits[-max_edits:]

    def record_task_list(self, content: str):
        """记住最近一份 TodoWrite 清单。

        它是长任务里的锚点（总共几件事、做到哪一步），所以有两个特殊待遇：
        Snip 时跳过它、Autocompact 后回灌它。否则跑过几个工具之后，这份清单
        就会被替换成占位符，而且取不回来。
        """
        self._task_list = content

    async def compress_if_needed(self, state: LoopState) -> LoopState:
        """Check token usage and apply appropriate compression tier."""
        tokens = count_tokens(state.messages)
        ratio = tokens / self._config.max_context_tokens

        # 已经压到最轻一档阈值以下 → 熔断器复原（说明上一轮压缩真的起作用了）
        if ratio < self._config.snip_threshold_ratio:
            if state.auto_compact_attempts:
                return state.with_field(auto_compact_attempts=0)
            return state

        if ratio >= self._config.autocompact_threshold_ratio:
            return await self._autocompact(state)
        if ratio >= self._config.collapse_threshold_ratio:
            return await self._collapse(state)
        if ratio >= self._config.snip_threshold_ratio:
            return await self._snip(state)
        return state

    # ── T2: Snip — placeholder replacement for old large tool outputs ──

    async def _snip(self, state: LoopState) -> LoopState:
        """Replace large old tool outputs with [[snip:KEY]] placeholders.

        Triggered when tool results account for > 50% of total tokens.
        Keeps the 5 most recent tool results, snips older ones by size.
        """
        TOOL_RESULT_RATIO = 0.50
        KEEP_RECENT = 5

        total = count_tokens(state.messages)
        tool_entries = [
            (i, msg) for i, msg in enumerate(state.messages)
            if msg.role == "tool"
        ]
        if not tool_entries:
            return state

        tool_tokens = sum(count_tokens((msg,)) for _, msg in tool_entries)
        if tool_tokens / total < TOOL_RESULT_RATIO:
            return state

        # 保留最近 5 个工具结果，裁剪更早的（tool_entries 已按消息顺序排列）
        to_snip = (
            tool_entries[:-KEEP_RECENT]
            if len(tool_entries) > KEEP_RECENT
            else []
        )

        new_messages = list(state.messages)
        for i, msg in to_snip:
            # 任务清单是锚点，不能变成占位符
            if self._task_list and msg.content == self._task_list:
                continue
            key = sha256(msg.content)[:8]
            self._cache[key] = msg.content
            new_messages[i] = Message(
                role="tool",
                content=f"[[snip:{key}]] ({len(msg.content)} chars — cached)",
                tool_call_id=msg.tool_call_id,  # 保留配对，否则 API 400
            )

        return state.with_field(messages=tuple(new_messages))

    # ── T3: Collapse — structured summary of middle messages ──

    async def _collapse(self, state: LoopState) -> LoopState:
        """Archive middle conversation into a structured summary.

        Keeps: first 3 messages (system + initial task + first reply)
        Keeps: last 20 messages (current working context)
        Summarizes: everything in between
        """
        HEAD_COUNT = 3
        TAIL_COUNT = 20

        if len(state.messages) <= HEAD_COUNT + TAIL_COUNT:
            return state

        head = state.messages[:HEAD_COUNT]
        tail = state.messages[-TAIL_COUNT:]
        middle = state.messages[HEAD_COUNT:-TAIL_COUNT]

        summary = self._summarize_sync(middle)
        summary_msg = Message(
            role="user",
            content=f"[Collapsed {len(middle)} messages]\n{summary}",
        )

        # 任务清单是长任务锚点，但它**不在摘要的抽取范围里**（摘要只抽
        # 「改过的文件 / 跑过的命令 / 出过的错」），所以清单落在中间段时会
        # 被永久压掉 —— 而 Collapse 的阈值（85%）低于 Autocompact（95%），
        # 长会话往往先经过这里，恰好是最需要"总共几件事"的阶段。
        # 所以像 Autocompact 一样显式回灌；已经在尾部保留的就不重复灌。
        restored: tuple[Message, ...] = (summary_msg,)
        if self._task_list and not any(
                m.content == self._task_list for m in tail):
            restored = (
                summary_msg,
                Message(role="user",
                        content=f"[Current task list]\n{self._task_list}"),
            )

        return state.with_field(
            messages=_strip_orphaned_tool_calls(head + restored + tail)
        )

    def _summarize_sync(self, messages: tuple[Message, ...]) -> str:
        """Generate a rule-based summary of conversation history.

        Extracts: file modifications, shell commands, and errors.
        Falls back gracefully when LLM is unavailable.
        """
        return summarize_messages(messages)

    # ── T4: Autocompact — last resort full-session compression ──

    async def _autocompact(self, state: LoopState) -> LoopState:
        """Ultra-compact: summarize entire session into a single message.

        Circuit breaker: stops trying after N consecutive compactions that
        failed to bring usage back under the snip threshold. Without the
        attempt counter being incremented here, the breaker was unreachable —
        compression would re-run every turn and burn a model call each time.
        Production data from Claude Code: 1,279 sessions had 50+
        consecutive failures before this breaker was added.
        """
        if state.auto_compact_attempts >= self._config.autocompact_circuit_breaker:
            return state  # circuit breaker: stop trying

        # Rule-based summary (LLM-free — safe even when context is full)
        summary = self._summarize_sync(state.messages)

        compacted = [
            state.messages[0],  # System Prompt always preserved
            Message(
                role="user",
                content=f"[Session Compressed]\n{summary}",
            ),
        ]

        # Restore recently edited files (post-compression amnesia prevention)
        for path, content in self._recent_edits[
            -self._config.restore_recent_edits:
        ]:
            compacted.append(
                Message(
                    role="user",
                    content=f"[Restored file]\n{path}:\n{content[:5000]}",
                )
            )

        # Restore active skill context
        for skill_name in state.active_skills[-3:]:
            compacted.append(
                Message(
                    role="user",
                    content=f"[Active skill: {skill_name}]",
                )
            )

        # Restore the task list — 长任务的进度锚点
        if self._task_list:
            compacted.append(Message(
                role="user",
                content=f"[Current task list]\n{self._task_list}",
            ))

        return state.with_field(
            messages=tuple(compacted),
            auto_compact_attempts=state.auto_compact_attempts + 1,
        )

    async def force_autocompact(self, state: LoopState) -> LoopState:
        """Force full-session compaction for prompt-too-long recovery.

        Bypasses the circuit breaker — the API has already rejected the
        context as too long, so we MUST compress regardless of prior
        failures. Falls back to collapse if summarization yields nothing.
        """
        summary = self._summarize_sync(state.messages)

        if summary:
            compacted = [
                state.messages[0],  # System Prompt always preserved
                Message(
                    role="user",
                    content=f"[Session Compressed]\n{summary}",
                ),
            ]

            # Restore recently edited files
            for path, content in self._recent_edits[
                -self._config.restore_recent_edits:
            ]:
                compacted.append(
                    Message(
                        role="user",
                        content=f"[Restored file]\n{path}:\n{content[:5000]}",
                    )
                )

            # Restore active skill context
            for skill_name in state.active_skills[-3:]:
                compacted.append(
                    Message(
                        role="user",
                        content=f"[Active skill: {skill_name}]",
                    )
                )

            # Restore the task list — 长任务的进度锚点
            if self._task_list:
                compacted.append(Message(
                    role="user",
                    content=f"[Current task list]\n{self._task_list}",
                ))

            return state.with_field(
                messages=tuple(compacted), auto_compact_attempts=0
            )

        # Summarization failed → fall back to collapse (head+tail+summary)
        return await self._collapse(state)
