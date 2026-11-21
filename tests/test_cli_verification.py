from click.testing import CliRunner

import main as main_module
from capabilities.docker_sandbox import DockerBashTool
from config import Config
from main import main


def test_help_lists_verification_and_isolation_flags():
    result = CliRunner().invoke(main, ["--help"])

    assert result.exit_code == 0
    assert "--verify-command" in result.output
    assert "--verify-timeout" in result.output
    assert "--verify-retries" in result.output
    assert "--isolate-subagents" in result.output
    assert "--sandbox-bash" in result.output
    assert "--sandbox-image" in result.output
    assert "--sandbox-network" in result.output


def test_build_tools_wires_subagent_isolation(monkeypatch, tmp_path):
    class FakeSkillSystem:
        def register_from_source(self, *args, **kwargs):
            return None

        def get_index_for_system_prompt(self):
            return ""

        def routing_hint(self, task, top_k=3):
            return ""

    class FakeMemory:
        def project_rules(self):
            return ""

        def load_claude_md(self):
            return ""

        def load_session_summary(self):
            return ""

        def search_sync(self, *args, **kwargs):
            return []

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(main_module, "SkillSystem", FakeSkillSystem)
    monkeypatch.setattr(main_module, "MemoryManager", FakeMemory)

    tools, *_ = main_module._build_tools(
        "normal", Config(model="test"), isolate_subagents=True,
        subagent_verifier_factory=lambda root: "verifier",
    )
    agent_tool = next(tool for tool in tools if tool.name == "Agent")
    tool_names = {tool.name for tool in tools}

    assert agent_tool._isolate_subagents is True
    assert agent_tool._worktree_manager is not None
    assert agent_tool._workspace_tool_factory is not None
    assert agent_tool._verifier_factory is not None
    assert {"WorktreeList", "WorktreeApply", "WorktreeRemove"} <= tool_names


def test_subagent_verifier_factory_is_none_without_command():
    assert main_module._make_subagent_verifier_factory(None, 120) is None
    assert main_module._make_subagent_verifier_factory("", 120) is None


def test_build_tools_selects_docker_bash(monkeypatch, tmp_path):
    class FakeSkillSystem:
        def register_from_source(self, *args, **kwargs):
            return None

        def get_index_for_system_prompt(self):
            return ""

        def routing_hint(self, task, top_k=3):
            return ""

    class FakeMemory:
        def project_rules(self):
            return ""

        def load_claude_md(self):
            return ""

        def load_session_summary(self):
            return ""

        def search_sync(self, *args, **kwargs):
            return []

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(main_module, "SkillSystem", FakeSkillSystem)
    monkeypatch.setattr(main_module, "MemoryManager", FakeMemory)

    tools, *_ = main_module._build_tools(
        "normal", Config(model="test"), sandbox_bash=True,
        sandbox_image="alpine:3.20", sandbox_network="none",
    )

    assert isinstance(next(tool for tool in tools if tool.name == "Bash"), DockerBashTool)
