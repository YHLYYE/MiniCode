"""多 Agent 协作测试 — explore/general/team"""
import asyncio

import pytest

from capabilities.multi_agent import (MAX_PROJECT_RULES_CHARS, AgentTool,
                                      compose_subagent_prompt)
from core.agent_loop import AgentLoop
from core.state import DoneEvent, LoopState, TextDelta


def test_agent_type_enum():
    """agent_type 支持三种模式"""
    at = AgentTool()
    assert list(at.input_schema["agent_type"]["enum"]) == [
        "explore", "general", "team"
    ]


def test_profiles_exist():
    """两个 profile 都定义了（explore/general）"""
    at = AgentTool()
    for key in ["explore", "general"]:
        assert key in at.AGENT_PROFILES
    # team 用 TEAM_ROLES，不在 AGENT_PROFILES
    assert "team" not in at.AGENT_PROFILES


def test_team_roles_are_tool_scoped():
    """research/testing 不能写文件；只有 coding 拿全量工具"""
    at = AgentTool()
    assert at.TEAM_ROLE_TOOLS["coding"] is None
    for role in ("research", "testing"):
        assert "Write" not in at.TEAM_ROLE_TOOLS[role]
        assert "Edit" not in at.TEAM_ROLE_TOOLS[role]
    # testing 需要跑命令来验证，所以给 Bash
    assert "Bash" in at.TEAM_ROLE_TOOLS["testing"]
    assert "Bash" not in at.TEAM_ROLE_TOOLS["research"]


def _role_of(system_prompt: str) -> str:
    if "research sub-agent" in system_prompt:
        return "research"
    if "coding sub-agent" in system_prompt:
        return "coding"
    return "testing"


def _pipeline_factory(seen, fail_on=None):
    """记录每个角色的 (name, tools, task)，并返回一个假 AgentLoop。"""
    def factory(tools, system_prompt, max_turns):
        entry = {"role": _role_of(system_prompt), "tools": tools, "task": None}
        seen.append(entry)

        class FakeLoop:
            async def run(self, task):
                from core.state import TextDelta, DoneEvent, LoopState
                entry["task"] = task
                if fail_on == entry["role"]:
                    raise RuntimeError(f"{entry['role']} exploded")
                yield TextDelta(f"{entry['role']}-output")
                yield DoneEvent(LoopState(turn_count=1))

        return FakeLoop()

    return factory


@pytest.mark.asyncio
async def test_team_runs_as_a_pipeline_and_passes_output_downstream():
    """team 是 research → coding → testing 串行流水线，每步拿到上一步产出"""
    seen = []

    at = AgentTool(agent_loop_factory=_pipeline_factory(seen),
                   tool_registry={})
    result = await at.execute("some task", agent_type="team")

    assert [e["role"] for e in seen] == ["research", "coding", "testing"]
    # 第一步只拿到原任务
    assert seen[0]["task"] == "some task"
    # 第二步带上 research 的产出
    assert "research-output" in seen[1]["task"]
    # 第三步带上 coding 的产出
    assert "coding-output" in seen[2]["task"]
    assert "Agent Team" in result and "3/3" in result


@pytest.mark.asyncio
async def test_team_stops_when_a_step_fails():
    """上一步失败 → 流水线中断，不再往下派发"""
    seen = []

    at = AgentTool(agent_loop_factory=_pipeline_factory(seen, fail_on="coding"),
                   tool_registry={})
    result = await at.execute("some task", agent_type="team")

    assert [e["role"] for e in seen] == ["research", "coding"]
    assert "2/3" in result
    assert "coding exploded" in result


@pytest.mark.asyncio
async def test_team_pipeline_filters_tools_per_role():
    """research 拿不到 Write/Edit，coding 拿到全量"""
    from core.tools.base import Tool

    class _Stub(Tool):
        input_schema = {"x": {"type": "string"}}
        async def execute(self, **kw): return "ok"

    registry = {}
    for name in ("Read", "Grep", "Glob", "Write", "Edit", "Bash"):
        stub = type(f"Stub_{name}", (_Stub,), {"name": name})()
        registry[name] = stub

    seen = []
    at = AgentTool(agent_loop_factory=_pipeline_factory(seen),
                   tool_registry=registry)
    await at.execute("some task", agent_type="team")

    research_tools = {t.name for t in seen[0]["tools"]}
    coding_tools = {t.name for t in seen[1]["tools"]}
    testing_tools = {t.name for t in seen[2]["tools"]}

    assert "Write" not in research_tools and "Edit" not in research_tools
    assert "Write" not in testing_tools and "Bash" in testing_tools
    assert {"Read", "Write", "Edit", "Bash"} <= coding_tools


@pytest.mark.asyncio
async def test_general_subagent_isolated_context():
    """子 Agent 拥有独立上下文，只返回摘要"""
    captured_task = []

    def factory(tools, system_prompt, max_turns):
        class FakeLoop:
            async def run(self, task):
                captured_task.append(task)
                from core.state import TextDelta, DoneEvent, LoopState
                yield TextDelta("sub-agent summary")
                yield DoneEvent(LoopState(turn_count=3))

        return FakeLoop()

    at = AgentTool(agent_loop_factory=factory, tool_registry={})
    result = await at.execute("do independent work", agent_type="general")

    assert captured_task == ["do independent work"]
    assert "sub-agent summary" in result
    assert "Sub-agent: general" in result


# ── 并发安全按 agent_type 判定 ──
# 回归背景：`is_concurrency_safe = True` 是**无条件**声明的，于是
# general / team（都能写文件）也被当成并发安全 —— 同一回合里两个子 Agent
# 会被 asyncio.gather 并行跑起来，可能同时改同一批文件。

def test_agent_concurrency_safety_depends_on_agent_type():
    """explore 只读 → 可并行；general / team / 未知 → 一律不安全（fail-closed）。"""
    at = AgentTool()
    assert at.check_concurrency_safe({"task": "x", "agent_type": "explore"}) is True
    for kind in ("general", "team", "agent-typo"):
        assert at.check_concurrency_safe(
            {"task": "x", "agent_type": kind}) is False, kind
    # 没写 agent_type 时 default 是 general → 不安全
    assert at.check_concurrency_safe({}) is False
    assert at.check_concurrency_safe(None) is False


class _Chunk:
    def __init__(self, type, text="", name="", input=None, stop_reason="",
                 usage=None, tool_call_id=""):
        self.type, self.text, self.name = type, text, name
        self.input, self.stop_reason = input, stop_reason
        self.usage, self.tool_call_id = usage, tool_call_id


class _Usage:
    input_tokens, output_tokens = 10, 5


class _MockModel:
    model = "mock"
    max_output_tokens = 8192

    def __init__(self, plan):
        self.plan, self.idx = plan, 0

    async def chat_streaming(self, messages, system="", tools=None, max_tokens=None):
        if self.idx >= len(self.plan):
            yield _Chunk(type="text_delta", text="done")
            yield _Chunk(type="message_stop", stop_reason="end_turn", usage=_Usage())
            return
        for c in self.plan[self.idx]:
            yield c
        self.idx += 1


def _two_agent_plan(agent_type: str) -> list:
    """一个回合里发两个 Agent 调用（任务 A 与 B）。"""
    return [
        [
            _Chunk(type="tool_use_start", name="Agent",
                   input={"task": "A", "agent_type": agent_type}, tool_call_id="c1"),
            _Chunk(type="tool_use_start", name="Agent",
                   input={"task": "B", "agent_type": agent_type}, tool_call_id="c2"),
            _Chunk(type="message_stop", stop_reason="end_turn", usage=_Usage()),
        ],
        [
            _Chunk(type="text_delta", text="done"),
            _Chunk(type="message_stop", stop_reason="end_turn", usage=_Usage()),
        ],
    ]


def _traced_factory(trace: list, delay: float = 0.15):
    """假子 Agent：把 start/end 记进 trace，好判断有没有真的重叠。"""
    def factory(tools, system_prompt, max_turns):
        class FakeLoop:
            async def run(self, task):
                trace.append(("start", task))
                await asyncio.sleep(delay)
                trace.append(("end", task))
                yield TextDelta(f"<{task}-summary>")
                yield DoneEvent(LoopState(turn_count=1))
        return FakeLoop()
    return factory


async def _run_two_agents(agent_type: str, trace: list):
    at = AgentTool(agent_loop_factory=_traced_factory(trace), tool_registry={})
    loop = AgentLoop(tools=[at],
                     model_adapter=_MockModel(_two_agent_plan(agent_type)),
                     system_prompt="t")
    async for _ in loop.run("go"):
        pass


@pytest.mark.asyncio
async def test_two_explore_agents_still_run_in_parallel():
    """不能因为这次修复把只读的 explore 也串行化 —— 那就丢了原本的收益。"""
    trace: list = []
    await _run_two_agents("explore", trace)
    first_end = next(i for i, (phase, _) in enumerate(trace) if phase == "end")
    assert first_end > 1, f"两个 explore 没有重叠，被串行化了：{trace}"


@pytest.mark.asyncio
async def test_two_general_agents_are_serialized():
    """general 能写文件 → 必须一个一个来（判定看执行轨迹，不看结果顺序）。"""
    trace: list = []
    await _run_two_agents("general", trace)
    assert trace == [("start", "A"), ("end", "A"),
                     ("start", "B"), ("end", "B")], trace


# ── 失败与非法输入都要变成文本，而不是抛异常 ──

@pytest.mark.asyncio
async def test_unknown_agent_type_returns_text_instead_of_raising():
    at = AgentTool(agent_loop_factory=_pipeline_factory([]), tool_registry={})
    out = await at.execute("whatever", agent_type="agent-typo")
    assert "Unknown agent_type" in out
    assert "agent-typo" in out


@pytest.mark.asyncio
async def test_missing_factory_degrades_without_raising():
    """没配工厂时要给出提示，而不是在拼用量抬头那一步崩掉。"""
    at = AgentTool(tool_registry={})
    out = await at.execute("x", agent_type="general")
    assert "not configured" in out
    assert "$0.0000" in out
    assert at.drain_usage() is None


# ── 项目约定要显式拼进子 Agent 的角色提示词 ──
# 回归背景：子 Agent 的上下文只有「角色提示词 + 任务」，父级的系统提示词
# （含 CLAUDE.md）不会传下去 —— 于是 research 定下的规范约束不到 coding。

def test_project_rules_are_appended_to_the_role_prompt():
    prompt = compose_subagent_prompt("You are a research sub-agent.",
                                     "禁止用 Tab 缩进；测试必须能离线跑。")
    assert prompt.startswith("You are a research sub-agent.")
    assert "禁止用 Tab 缩进" in prompt
    assert "项目约定" in prompt          # 有分段标题，模型能看出哪部分是规范


def test_no_project_rules_leaves_the_prompt_untouched():
    """没有 CLAUDE.md 时不许拼一个空壳，更不许把占位文案当约定灌进去。"""
    for empty in (None, "", "   \n  "):
        assert compose_subagent_prompt("You are a coder.", empty) == "You are a coder."


def test_project_rules_are_truncated():
    """超长 CLAUDE.md 不能把每个子 Agent 的上下文都吃掉。"""
    long_rules = "x" * (MAX_PROJECT_RULES_CHARS + 500)
    prompt = compose_subagent_prompt("role", long_rules)
    assert prompt.count("x") == MAX_PROJECT_RULES_CHARS
