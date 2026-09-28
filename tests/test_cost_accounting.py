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
