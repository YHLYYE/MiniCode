"""Shared agent state types — no circular dependencies"""
from dataclasses import dataclass, field, replace
from enum import Enum


# 兜底计价：每 1M token 的美元价，取 Claude Sonnet 3.5 档（$3 / $15）。
#
# 这是**兜底**，不是主口径 —— 真实计价走 `ModelAdapter.estimate_cost_usd()`
# （litellm 自带价目表）。写死单一价格是错的，而且错在两个方向：
# 1M input + 1M output 按这里是 $18，而 litellm 的价目表里
#   deepseek-chat 是 $0.70（高估约 26 倍）、claude-sonnet-4-5 是 $28.50（低估 1.6 倍）。
# 这个数字下游是 `--max-cost` 预算护栏，所以口径错了护栏就是错的。
FALLBACK_INPUT_PRICE_PER_M = 3.0
FALLBACK_OUTPUT_PRICE_PER_M = 15.0


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
    # 注意：frozen=True 给的是**不可变性**，不是可哈希性。tool_calls 是 list，
    # 所以带 tool_calls 的 Message（以及含它的 LoopState）hash() 会 TypeError。
    # 现在没有任何地方拿它们当 dict key / 放进 set，所以保持 list 不改成 tuple
    # —— 它要原样进 OpenAI/DeepSeek 的请求体，也必须能直接 JSON 序列化。


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

    def accumulate_usage(self, usage, cost_usd: float | None = None) -> "LoopState":
        """累计一轮的 token 与花费。

        `cost_usd` 由调用方按**真实模型**价目给出（`ModelAdapter`），
        给不出（适配器没实现 / 价目表没有这个模型）才退回 FALLBACK_* 单价。
        以前这里只有写死单价：换模型之后账本会错到离谱，而 `--max-cost`
        正是拿这个账本当护栏。
        """
        new_tokens = self.total_tokens + usage.input_tokens + usage.output_tokens
        if cost_usd is None:
            cost_usd = (
                usage.input_tokens * FALLBACK_INPUT_PRICE_PER_M
                + usage.output_tokens * FALLBACK_OUTPUT_PRICE_PER_M
            ) / 1_000_000
        return replace(
            self,
            total_tokens=new_tokens,
            total_cost_usd=round(self.total_cost_usd + cost_usd, 6),
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
