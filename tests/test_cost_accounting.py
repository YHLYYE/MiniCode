"""子 Agent 用量回流测试。

此前 AgentTool 只把子 Agent 的 token 拼进返回字符串给模型看，
父级的 LoopState 完全不包含这份消耗 —— `--max-cost` 只管得住主 Agent，
最后的用量报告也是少算的。
"""
import pytest

from capabilities.multi_agent import AgentTool
from core.agent_loop import AgentLoop
from core.state import DoneEvent, LoopState, TextDelta
from core.tools.base import Tool


class _Chunk:
    def __init__(self, type, text="", name="", input=None, stop_reason="",
                 usage=None, tool_call_id=""):
        self.type = type
        self.text = text
        self.name = name
        self.input = input
        self.stop_reason = stop_reason
        self.usage = usage
        self.tool_call_id = tool_call_id


class _Usage:
    def __init__(self, input_tokens=10, output_tokens=5):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class _MockModel:
    model = "mock"
    max_output_tokens = 8192

    def __init__(self, plan):
        self.plan = plan
        self.idx = 0

    async def chat_streaming(self, messages, system="", tools=None,
                             max_tokens=None):
        if self.idx >= len(self.plan):
            yield _Chunk(type="text_delta", text="done")
            yield _Chunk(type="message_stop", stop_reason="end_turn",
                         usage=_Usage())
            return
        for c in self.plan[self.idx]:
            yield c
        self.idx += 1


def _subagent_factory(tokens=1234, cost=0.5):
    """假子 Agent：直接报一份用量。"""
    def factory(tools, system_prompt, max_turns):
        class FakeLoop:
            async def run(self, task):
                yield TextDelta("sub-agent summary")
                yield DoneEvent(LoopState(turn_count=2, total_tokens=tokens,
                                          total_cost_usd=cost))
        return FakeLoop()
    return factory


# ── AgentTool.drain_usage ──

@pytest.mark.asyncio
async def test_drain_usage_returns_and_resets():
    at = AgentTool(agent_loop_factory=_subagent_factory(1000, 0.25),
                   tool_registry={})
    await at.execute("task", agent_type="general")

    assert at.drain_usage() == (1000, 0.25)
    # 取过一次就清零，避免重复计费
    assert at.drain_usage() is None


@pytest.mark.asyncio
async def test_drain_usage_accumulates_across_team_roles():
    at = AgentTool(agent_loop_factory=_subagent_factory(100, 0.1),
                   tool_registry={})
    await at.execute("task", agent_type="team")

    tokens, cost = at.drain_usage()
    assert tokens == 300        # 三个角色
    assert cost == pytest.approx(0.3)


def test_drain_usage_empty_before_any_run():
    assert AgentTool(agent_loop_factory=_subagent_factory()).drain_usage() is None


# ── 端到端：父级账本要包含子 Agent 的消耗 ──

@pytest.mark.asyncio
async def test_subagent_usage_lands_in_parent_state():
    plan = [
        [
            _Chunk(type="tool_use_start", name="Agent",
                   input={"task": "look around", "agent_type": "general"},
                   tool_call_id="c1"),
            _Chunk(type="message_stop", stop_reason="end_turn", usage=_Usage()),
        ],
        [
            _Chunk(type="text_delta", text="done"),
            _Chunk(type="message_stop", stop_reason="end_turn", usage=_Usage()),
        ],
    ]
    agent_tool = AgentTool(agent_loop_factory=_subagent_factory(5000, 0.75),
                           tool_registry={})
    loop = AgentLoop(tools=[agent_tool], model_adapter=_MockModel(plan),
                     system_prompt="test", max_cost_usd=5.0)

    async for _ in loop.run("go"):
        pass

    assert loop.state.total_tokens >= 5000
    assert loop.state.total_cost_usd >= 0.75


@pytest.mark.asyncio
async def test_subagent_spend_counts_toward_parent_budget():
    """子 Agent 花超预算时，父级下一轮必须被预算拦住。"""
    plan = [
        [
            _Chunk(type="tool_use_start", name="Agent",
                   input={"task": "expensive", "agent_type": "general"},
                   tool_call_id="c1"),
            _Chunk(type="message_stop", stop_reason="end_turn", usage=_Usage()),
        ],
        [
            _Chunk(type="text_delta", text="should not get here"),
            _Chunk(type="message_stop", stop_reason="end_turn", usage=_Usage()),
        ],
    ]
    agent_tool = AgentTool(agent_loop_factory=_subagent_factory(10, 9.0),
                           tool_registry={})
    loop = AgentLoop(tools=[agent_tool], model_adapter=_MockModel(plan),
                     system_prompt="test", max_cost_usd=1.0)

    from core.state import BudgetExceeded
    with pytest.raises(BudgetExceeded):
        async for _ in loop.run("go"):
            pass


# ── 工具没有 drain_usage 时不能出问题 ──

@pytest.mark.asyncio
async def test_tool_without_drain_usage_is_fine():
    class PlainTool(Tool):
        name = "plain"
        input_schema = {}
        is_readonly = True

        async def execute(self):
            return "ok"

    plan = [
        [_Chunk(type="tool_use_start", name="plain", input={},
                tool_call_id="c1"),
         _Chunk(type="message_stop", stop_reason="end_turn", usage=_Usage())],
        [_Chunk(type="text_delta", text="done"),
         _Chunk(type="message_stop", stop_reason="end_turn", usage=_Usage())],
    ]
    loop = AgentLoop(tools=[PlainTool()], model_adapter=_MockModel(plan),
                     system_prompt="test")
    events = [e async for e in loop.run("go")]
    assert any(isinstance(e, DoneEvent) for e in events)


# ── 计价口径必须跟着模型走 ──
# 此前 LoopState.accumulate_usage 里写死了单一价格（$3/$15 每 1M），
# 而 .env.example 首推的是 deepseek-chat：1M in + 1M out 记 $18，
# litellm 的真实价目是 $0.70 —— 高估约 26 倍。这个数字下游是 --max-cost
# 预算护栏，所以口径错了护栏就是错的（另一头 claude-sonnet-4-5 反而低估 1.6 倍）。

def test_accumulate_usage_uses_the_given_cost():
    """调用方给了真实价格就用它，不再套兜底单价。"""
    state = LoopState().accumulate_usage(_Usage(1_000_000, 1_000_000),
                                         cost_usd=0.70)
    assert state.total_cost_usd == 0.70
    assert state.total_tokens == 2_000_000


def test_accumulate_usage_fallback_is_the_documented_constant():
    """算不出价时才退兜底，而且兜底单价只有一处定义（改一处即全局生效）。"""
    from core.state import (FALLBACK_INPUT_PRICE_PER_M,
                            FALLBACK_OUTPUT_PRICE_PER_M)

    state = LoopState().accumulate_usage(_Usage(1_000_000, 0))
    assert state.total_cost_usd == FALLBACK_INPUT_PRICE_PER_M

    state = LoopState().accumulate_usage(_Usage(0, 1_000_000))
    assert state.total_cost_usd == FALLBACK_OUTPUT_PRICE_PER_M


def test_model_adapter_prices_a_real_model():
    """真实适配器要能查到 litellm 的价目表，而且比兜底单价便宜得多。

    只断言相对关系（有价、比 Claude 档兜底便宜），不钉具体数字 ——
    价目表是外部数据，会随官方调价变化。
    """
    from core.model_adapter import ModelAdapter
    from core.state import FALLBACK_INPUT_PRICE_PER_M

    adapter = ModelAdapter("deepseek-chat")
    cost = adapter.estimate_cost_usd(1_000_000, 0)

    assert cost is not None, "litellm 价目表里应该有 deepseek-chat"
    assert 0 < cost < FALLBACK_INPUT_PRICE_PER_M, (
        f"deepseek 比 Claude 档便宜应该是常识，实测 {cost}")


def test_model_adapter_returns_none_for_unpriced_model():
    """查不到价目的模型返回 None（调用方好退回兜底），而不是抛异常。"""
    from core.model_adapter import ModelAdapter

    adapter = ModelAdapter("definitely-not-a-real-model-xyz")
    assert adapter.estimate_cost_usd(1000, 500) is None


@pytest.mark.asyncio
async def test_loop_charges_the_adapter_price_not_the_fallback():
    """端到端：loop 记账用的是适配器报的价，不是兜底单价。"""
    class PricedModel(_MockModel):
        model = "mock"

        def estimate_cost_usd(self, input_tokens, output_tokens):
            return 1.25          # 一眼能认出来的数字

    plan = [
        [_Chunk(type="text_delta", text="done"),
         _Chunk(type="message_stop", stop_reason="end_turn",
                usage=_Usage(1000, 500))],
    ]
    loop = AgentLoop(tools=[], model_adapter=PricedModel(plan),
                     system_prompt="test")
    _ = [e async for e in loop.run("go")]

    assert loop.state.total_cost_usd == 1.25


@pytest.mark.asyncio
async def test_loop_falls_back_when_adapter_cannot_price():
    """适配器没有计价能力时不能崩，退回兜底单价（现有 mock 全是这种）。"""
    from core.state import FALLBACK_INPUT_PRICE_PER_M

    plan = [
        [_Chunk(type="text_delta", text="done"),
         _Chunk(type="message_stop", stop_reason="end_turn",
                usage=_Usage(1_000_000, 0))],
    ]
    loop = AgentLoop(tools=[], model_adapter=_MockModel(plan),
                     system_prompt="test")
    _ = [e async for e in loop.run("go")]

    assert loop.state.total_cost_usd == FALLBACK_INPUT_PRICE_PER_M
