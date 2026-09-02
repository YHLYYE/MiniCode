"""Shared agent state types — no circular dependencies"""
import time
from dataclasses import dataclass, field, replace
from enum import Enum


class ContinueReason(Enum):
    NEXT_TURN = "next_turn"
    PROMPT_TOO_LONG_RETRY = "ptl_retry"
    MAX_OUTPUT_TOKENS_UPGRADE = "mot_upgrade"
    MAX_OUTPUT_TOKENS_RECOVERY = "mot_recovery"


@dataclass(frozen=True)
class Message:
    role: str       # "system" | "user" | "assistant" | "tool"
    content: str
    tool_call_id: str | None = field(default=None)  # for OpenAI/DeepSeek format
    tool_calls: list | None = field(default=None)   # assistant's tool_calls (OpenAI format)


@dataclass(frozen=True)
class LoopState:
    messages: tuple[Message, ...] = ()
    turn_count: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    max_output_tokens_recovery: int = 0
    auto_compact_attempts: int = 0
    transition: ContinueReason | None = None
    active_skills: tuple[str, ...] = ()

    def add_message(self, msg: Message) -> "LoopState":
        return replace(self, messages=self.messages + (msg,))

    def add_messages(self, msgs: list[Message]) -> "LoopState":
        return replace(self, messages=self.messages + tuple(msgs))

    def with_transition(self, reason: ContinueReason) -> "LoopState":
        return replace(self, transition=reason)

    def with_field(self, **kwargs) -> "LoopState":
        return replace(self, **kwargs)

    def accumulate_usage(self, usage) -> "LoopState":
        new_tokens = self.total_tokens + usage.input_tokens + usage.output_tokens
        cost = (
            usage.input_tokens * 3 / 1_000_000 +
            usage.output_tokens * 15 / 1_000_000
        )
        return replace(
            self,
            total_tokens=new_tokens,
            total_cost_usd=round(self.total_cost_usd + cost, 6),
        )


@dataclass
class TextDelta:
    text: str

@dataclass
class ToolStart:
    tool_name: str

@dataclass
class ToolResult:
    tool_name: str
    output: str

@dataclass
class ToolError:
    tool_name: str
    error: str

@dataclass
class DoneEvent:
    state: LoopState


class BudgetExceeded(Exception):
    pass


class UnrecoverableError(Exception):
    pass
