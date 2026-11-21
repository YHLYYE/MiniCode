import subprocess

import pytest

from capabilities.multi_agent import AgentTool
from capabilities.worktree import WorktreeManager
from core.state import DoneEvent, LoopState, TextDelta
from core.tools.files import WriteTool
from core.tools.shell import BashTool


def git(repo, *args):
    return subprocess.check_output(
        ["git", *args], cwd=repo, text=True, stderr=subprocess.STDOUT
    ).strip()


@pytest.fixture
def repo(tmp_path):
    git(tmp_path, "init")
    git(tmp_path, "config", "user.email", "test@example.com")
    git(tmp_path, "config", "user.name", "Test")
    (tmp_path / "a.txt").write_text("a")
    git(tmp_path, "add", ".")
    git(tmp_path, "commit", "-m", "init")
    return tmp_path


@pytest.mark.asyncio
async def test_general_agent_runs_in_worktree_when_enabled(repo):
    seen = {}

    def factory(tools, system_prompt, max_turns):
        seen["tools"] = {tool.name: tool for tool in tools}

        class FakeLoop:
            async def run(self, task):
                result = await seen["tools"]["Write"].execute(
                    file_path="generated.txt", content="isolated"
                )
                seen["write_result"] = result
                yield TextDelta("wrote generated.txt")
                yield DoneEvent(LoopState(turn_count=1))

        return FakeLoop()

    agent_tool = AgentTool(
        agent_loop_factory=factory,
        tool_registry={},
        worktree_manager=WorktreeManager(repo),
        workspace_tool_factory=lambda root: {
            "Write": WriteTool(workspace_root=root),
        },
        isolate_subagents=True,
    )

    result = await agent_tool.execute("write generated file", agent_type="general")

    assert "branch: minicode/" in result
    assert "changed_files: ['generated.txt']" in result
    assert not (repo / "generated.txt").exists()
    assert seen["write_result"].startswith("Created")


@pytest.mark.asyncio
async def test_explore_never_uses_worktree(repo):
    def factory(tools, system_prompt, max_turns):
        class FakeLoop:
            async def run(self, task):
                yield TextDelta("inspected")
                yield DoneEvent(LoopState(turn_count=1))

        return FakeLoop()

    agent_tool = AgentTool(
        agent_loop_factory=factory,
        tool_registry={},
        worktree_manager=WorktreeManager(repo),
        isolate_subagents=True,
    )

    await agent_tool.execute("inspect repo", agent_type="explore")

    assert not (repo / ".minicode" / "worktrees").exists()


@pytest.mark.asyncio
async def test_team_coding_and_testing_share_one_worktree(repo):
    seen = {}

    def factory(tools, system_prompt, max_turns):
        by_name = {tool.name: tool for tool in tools}
        role = (
            "research" if "research sub-agent" in system_prompt
            else "coding" if "coding sub-agent" in system_prompt
            else "testing"
        )
        seen[role] = by_name

        class FakeLoop:
            async def run(self, task):
                if role == "coding":
                    await by_name["Write"].execute(
                        file_path="team.txt", content="done"
                    )
                yield TextDelta(f"{role}-output")
                yield DoneEvent(LoopState(turn_count=1))

        return FakeLoop()

    agent_tool = AgentTool(
        agent_loop_factory=factory,
        tool_registry={},
        worktree_manager=WorktreeManager(repo),
        workspace_tool_factory=lambda root: {
            "Write": WriteTool(workspace_root=root),
            "Bash": BashTool(workspace_root=root),
        },
        isolate_subagents=True,
    )

    result = await agent_tool.execute("team change", agent_type="team")

    coding_root = seen["coding"]["Write"]._backend.root
    testing_root = seen["testing"]["Bash"]._workspace_root
    assert coding_root == testing_root
    assert "changed_files: ['team.txt']" in result
    assert not (repo / "team.txt").exists()
