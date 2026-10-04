"""Shared agent state types — no circular dependencies"""
from dataclasses import dataclass, field, replace
from enum import Enum


class ContinueReason(Enum):
    NEXT_TURN = "next_turn"
    PROMPT_TOO_LONG_RETRY = "ptl_retry"
    MAX_OUTPUT_TOKENS_UPGRADE = "mot_upgrade"
    MAX_OUTPUT_TOKENS_RECOVERY = "mot_recovery"
    # 下面三个以前都写成 NEXT_TURN，跟"正常继续"分不出来 ——
    # 于是"这一轮为什么继续"这个信号对这三条恢复路径是失效的。
    STREAM_RETRY = "stream_retry"           # 网络瞬态错误 → 退避重试
    TOOL_ERROR_RETRY = "tool_error_retry"   # 工具报错 → 把错误回喂给模型
    EMPTY_RESPONSE_RETRY = "empty_retry"    # 模型返回空响应 → 提醒它继续


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

    def add_external_usage(self, tokens: int, cost_usd: float) -> "LoopState":
        """累加「不落在本 loop 账本里」的用量 —— 目前是子 Agent 的消耗。

        子 Agent 是独立的 AgentLoop，用量记在它自己的 state 上。父级如果
        不收回这份用量，`--max-cost` 就只管得住主 Agent 自己，最后那行
        「完成: N tokens, $X」也会少算。
        """
        return replace(
            self,
            total_tokens=self.total_tokens + tokens,
            total_cost_usd=round(self.total_cost_usd + cost_usd, 6),
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


@dataclass
class RecoveryNotice:
    """某条恢复路径刚刚把这一轮救了回来。

    为什么要有这个事件：恢复路径的设计目标是"对用户无感"，但完全静默的代价
    是**连使用者自己都不知道它触发过**。这个事件让恢复变得可观测，且不打断流程。
    """
    reason: ContinueReason
    detail: str = ""


class BudgetExceeded(Exception):
    pass
