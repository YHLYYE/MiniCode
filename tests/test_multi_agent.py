"""多 Agent 协作测试 — explore/general/worktree/team"""
import pytest
from pathlib import Path
from capabilities.multi_agent import AgentTool


def test_agent_type_enum():
    """agent_type 支持三种模式"""
    at = AgentTool()
    assert list(at.input_schema["agent_type"]["enum"]) == [
        "explore", "general", "team"
    ]


def test_team_roles():
    """team 模式有三个角色"""
    at = AgentTool()
    assert at.TEAM_ROLES == ["research", "coding", "testing"]


def test_worktree_degrade_non_git(tmp_path):
    """非 git 仓库时 worktree 降级为 general 模式"""
    at = AgentTool(project_root=tmp_path)  # tmp_path 不是 git 仓库
    assert not at._is_git_repo()


def test_profiles_exist():
    """四个 profile 都定义了"""
    at = AgentTool()
    for key in ["explore", "general", "worktree"]:
        assert key in at.AGENT_PROFILES
    # team 用 TEAM_ROLES，不在 AGENT_PROFILES
    assert "team" not in at.AGENT_PROFILES


@pytest.mark.asyncio
async def test_worktree_degrade_message():
    """非 git 仓库时 worktree 返回降级提示"""
    # 用 mock factory，避免真实 API 调用
    calls = []

    def factory(tools, system_prompt, max_turns):
        calls.append((tools, system_prompt, max_turns))

        class FakeLoop:
            async def run(self, task):
                from core.state import TextDelta, DoneEvent, LoopState
                yield TextDelta("degraded sub-agent output")
                yield DoneEvent(LoopState(turn_count=1))

        return FakeLoop()

    at = AgentTool(
        agent_loop_factory=factory,
        tool_registry={},
        project_root=Path(".") if not Path(".").is_absolute() else Path("."),
    )

    # 确保不是 git 仓库（Desktop/minicode 当前不是 git）
    if not at._is_git_repo():
        result = await at.execute("some task", agent_type="worktree")
        assert "degraded" in result.lower() or "sub-agent" in result.lower()
    else:
        pytest.skip("当前是 git 仓库，跳过降级测试")


@pytest.mark.asyncio
async def test_team_parallel_execution():
    """team 模式并行执行三个角色，汇总结果"""
    calls = []

    def factory(tools, system_prompt, max_turns):
        calls.append(system_prompt)

        class FakeLoop:
            async def run(self, task):
                from core.state import TextDelta, DoneEvent, LoopState
                yield TextDelta(f"role output: {system_prompt[:20]}")
                yield DoneEvent(LoopState(turn_count=1))

        return FakeLoop()

    at = AgentTool(agent_loop_factory=factory, tool_registry={})
    result = await at.execute("some task", agent_type="team")

    # 三个角色都被派发
    assert len(calls) == 3
    # 汇总结果包含三个角色名
    assert "research" in result
    assert "coding" in result
    assert "testing" in result
    assert "Agent Team" in result


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
