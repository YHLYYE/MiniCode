"""多 Agent 协作测试 — explore/general/team"""
import pytest
from capabilities.multi_agent import AgentTool


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
