import subprocess
import sys

import pytest

from capabilities.multi_agent import AgentTool
from capabilities.worktree import WorktreeManager
from capabilities.verification import CommandVerifier, VerificationResult
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

    manager = WorktreeManager(repo)
    agent_tool = AgentTool(
        agent_loop_factory=factory,
        tool_registry={},
        worktree_manager=manager,
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

    manager = WorktreeManager(repo)
    agent_tool = AgentTool(
        agent_loop_factory=factory,
        tool_registry={},
        worktree_manager=manager,
        isolate_subagents=True,
    )

    await agent_tool.execute("inspect repo", agent_type="explore")

    assert not (repo / ".minicode" / "worktrees").exists()


@pytest.mark.asyncio
async def test_team_coding_and_testing_share_one_worktree(repo):
    seen = {}

    class FakeVerifier:
        def __init__(self, root):
            self.root = root
            seen["verify_root"] = root

        async def verify(self):
            return VerificationResult("fake", True, 0, "ok")

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
        verifier_factory=FakeVerifier,
        isolate_subagents=True,
    )

    result = await agent_tool.execute("team change", agent_type="team")

    coding_root = seen["coding"]["Write"]._backend.root
    testing_root = seen["testing"]["Bash"]._workspace_root
    assert coding_root == testing_root
    assert seen["verify_root"] == coding_root
    assert "changed_files: ['team.txt']" in result
    assert "verification: passed" in result
    assert not (repo / "team.txt").exists()


@pytest.mark.asyncio
async def test_general_failure_reports_preserved_worktree(repo):
    def factory(tools, system_prompt, max_turns):
        by_name = {tool.name: tool for tool in tools}

        class FakeLoop:
            async def run(self, task):
                await by_name["Write"].execute(
                    file_path="partial.txt", content="partial"
                )
                raise RuntimeError("subagent crashed")
                yield  # pragma: no cover

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

    result = await agent_tool.execute("crash after write", agent_type="general")

    assert "branch: minicode/" in result
    assert "path:" in result
    assert not (repo / "partial.txt").exists()


@pytest.mark.asyncio
async def test_general_worktree_verification_runs_before_commit(repo):
    seen = {}

    class FakeVerifier:
        def __init__(self, root):
            self.root = root
            seen["verify_root"] = root

        async def verify(self):
            return VerificationResult("fake", True, 0, "ok")

    def factory(tools, system_prompt, max_turns):
        by_name = {tool.name: tool for tool in tools}

        class FakeLoop:
            async def run(self, task):
                await by_name["Write"].execute(
                    file_path="verified.txt", content="verified"
                )
                yield TextDelta("done")
                yield DoneEvent(LoopState(turn_count=1))

        return FakeLoop()

    manager = WorktreeManager(repo)
    agent_tool = AgentTool(
        agent_loop_factory=factory,
        tool_registry={},
        worktree_manager=manager,
        workspace_tool_factory=lambda root: {
            "Write": WriteTool(workspace_root=root),
        },
        verifier_factory=FakeVerifier,
        isolate_subagents=True,
    )

    result = await agent_tool.execute("verified change", agent_type="general")

    assert seen["verify_root"].name.startswith("verified-change-")
    assert "verification: passed" in result
    assert "commit: none" not in result
    assert agent_tool.completion_blocker() is None
    assert manager.list_entries()[0]["status"] == "ready"
    assert manager.list_entries()[0]["verification"] == "passed"
    assert not (repo / "verified.txt").exists()


@pytest.mark.asyncio
async def test_failed_worktree_verification_keeps_uncommitted_worktree(repo):
    class FakeVerifier:
        def __init__(self, root):
            self.root = root

        async def verify(self):
            return VerificationResult("fake", False, 1, "test failed")

    def factory(tools, system_prompt, max_turns):
        by_name = {tool.name: tool for tool in tools}

        class FakeLoop:
            async def run(self, task):
                await by_name["Write"].execute(
                    file_path="bad.txt", content="bad"
                )
                yield TextDelta("done")
                yield DoneEvent(LoopState(turn_count=1))

        return FakeLoop()

    manager = WorktreeManager(repo)
    agent_tool = AgentTool(
        agent_loop_factory=factory,
        tool_registry={},
        worktree_manager=manager,
        workspace_tool_factory=lambda root: {
            "Write": WriteTool(workspace_root=root),
        },
        verifier_factory=FakeVerifier,
        isolate_subagents=True,
    )

    result = await agent_tool.execute("bad change", agent_type="general")

    assert "verification: failed" in result
    assert "commit: none" in result
    assert "uncommitted: true" in result
    assert "changed_files: ['bad.txt']" in result
    assert "test failed" in result
    assert "isolated subagent verification failed" in (
        agent_tool.completion_blocker() or ""
    )
    assert manager.list_entries()[0]["status"] == "verify_failed"
    assert manager.list_entries()[0]["verification"] == "failed"
    assert not (repo / "bad.txt").exists()


@pytest.mark.asyncio
async def test_real_command_verifier_sees_worktree_files(repo):
    def factory(tools, system_prompt, max_turns):
        by_name = {tool.name: tool for tool in tools}

        class FakeLoop:
            async def run(self, task):
                await by_name["Write"].execute(
                    file_path="evidence.txt", content="present"
                )
                yield TextDelta("done")
                yield DoneEvent(LoopState(turn_count=1))

        return FakeLoop()

    command = (
        f'"{sys.executable}" -c '
        '"from pathlib import Path; '
        'raise SystemExit(0 if Path(\'evidence.txt\').exists() else 1)"'
    )
    agent_tool = AgentTool(
        agent_loop_factory=factory,
        tool_registry={},
        worktree_manager=WorktreeManager(repo),
        workspace_tool_factory=lambda root: {
            "Write": WriteTool(workspace_root=root),
        },
        verifier_factory=lambda root: CommandVerifier(
            command, cwd=root, timeout_seconds=5
        ),
        isolate_subagents=True,
    )

    result = await agent_tool.execute("verify evidence", agent_type="general")

    assert "verification: passed" in result
    assert "verification: failed" not in result


@pytest.mark.asyncio
async def test_team_verification_failure_has_explicit_status(repo):
    class FailingVerifier:
        def __init__(self, root):
            self.root = root

        async def verify(self):
            return VerificationResult("fake", False, 1, "team tests failed")

    def factory(tools, system_prompt, max_turns):
        by_name = {tool.name: tool for tool in tools}
        role = (
            "research" if "research sub-agent" in system_prompt
            else "coding" if "coding sub-agent" in system_prompt
            else "testing"
        )

        class FakeLoop:
            async def run(self, task):
                if role == "coding":
                    await by_name["Write"].execute(
                        file_path="team-bad.txt", content="bad"
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
        },
        verifier_factory=FailingVerifier,
        isolate_subagents=True,
    )

    result = await agent_tool.execute("bad team", agent_type="team")

    assert "status: verification_failed" in result
    assert "verification: failed" in result
    assert "commit: none" in result
