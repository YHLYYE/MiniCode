from click.testing import CliRunner

import main as main_module
from config import Config
from main import main


def test_help_lists_verification_and_isolation_flags():
    result = CliRunner().invoke(main, ["--help"])

    assert result.exit_code == 0
    assert "--verify-command" in result.output
    assert "--verify-timeout" in result.output
    assert "--verify-retries" in result.output
    assert "--isolate-subagents" in result.output


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
        "normal", Config(model="test"), isolate_subagents=True
    )
    agent_tool = next(tool for tool in tools if tool.name == "Agent")

    assert agent_tool._isolate_subagents is True
    assert agent_tool._worktree_manager is not None
    assert agent_tool._workspace_tool_factory is not None
