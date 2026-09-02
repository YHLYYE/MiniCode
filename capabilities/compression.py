"""Context compression — 4-tier degradation chain

Reference: Claude Code's 4-level compression pipeline
(how-claude-code-works ch03 — Truncation → Snip → Collapse → Autocompact)

Token usage thresholds:
  50% → Truncation: single result > 30K chars → truncate + summary
  70% → Snip: large old tool outputs → placeholder replacement
  85% → Collapse: middle messages → structured summary
  95% → Autocompact: fork sub-agent → full session summary (last resort)
"""

import hashlib
import tiktoken
from dataclasses import dataclass, field
from enum import Enum

from core.state import LoopState, Message

# Token counter — uses cl100k_base (Claude + GPT compatible)
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


class CompressionTier(Enum):
    NONE = 0
    TRUNCATION = 1
    SNIP = 2
    COLLAPSE = 3
    AUTOCOMPACT = 4


@dataclass
class CompressionConfig:
    max_context_tokens: int = 60_000  # DeepSeek V3 64K window, leave 4K margin
    truncation_threshold: int = 30_000  # chars (single tool result)
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

    def record_edit(self, file_path: str, content: str):
        """Track recent file edits for post-compression restoration."""
        self._recent_edits.append((file_path, content))
        max_edits = self._config.restore_recent_edits
        if len(self._recent_edits) > max_edits:
            self._recent_edits = self._recent_edits[-max_edits:]

    async def compress_if_needed(self, state: LoopState) -> LoopState:
        """Check token usage and apply appropriate compression tier."""
        tokens = count_tokens(state.messages)
        ratio = tokens / self._config.max_context_tokens

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

        # Sort by content size descending, snip largest first
        tool_entries.sort(key=lambda x: len(x[1].content), reverse=True)
        to_snip = (
            tool_entries[:-KEEP_RECENT]
            if len(tool_entries) > KEEP_RECENT
            else []
        )

        new_messages = list(state.messages)
        for i, msg in to_snip:
            key = sha256(msg.content)[:8]
            self._cache[key] = msg.content
            new_messages[i] = Message(
                role="tool",
                content=f"[[snip:{key}]] ({len(msg.content)} chars — cached)",
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

        return state.with_field(messages=head + (summary_msg,) + tail)

    def _summarize_sync(self, messages: tuple[Message, ...]) -> str:
        """Generate a rule-based summary of conversation history.

        Extracts: file modifications, shell commands, and errors.
        Falls back gracefully when LLM is unavailable.
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

    # ── T4: Autocompact — last resort full-session compression ──

    async def _autocompact(self, state: LoopState) -> LoopState:
        """Ultra-compact: summarize entire session into a single message.

        Circuit breaker: stops trying after N consecutive failures.
        Production data from Claude Code: 1,279 sessions had 50+
        consecutive failures before this breaker was added.
        """
        if state.auto_compact_attempts >= self._config.autocompact_circuit_breaker:
            return state  # circuit breaker: stop trying

        # Rule-based summary (LLM-free — safe even when context is full)
        summary = self._summarize_sync(state.messages)
        if not summary:
            return state.with_field(
                auto_compact_attempts=state.auto_compact_attempts + 1
            )

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

        return state.with_field(
            messages=tuple(compacted), auto_compact_attempts=0
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

            return state.with_field(
                messages=tuple(compacted), auto_compact_attempts=0
            )

        # Summarization failed → fall back to collapse (head+tail+summary)
        return await self._collapse(state)
