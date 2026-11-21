"""Offline benchmark for verification and Worktree lifecycle behavior."""
from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from capabilities.multi_agent import AgentTool
from capabilities.verification import CommandVerifier
from capabilities.worktree import WorktreeManager
from core.agent_loop import AgentLoop
from core.state import (
    DoneEvent,
    LoopState,
    TerminationReason,
    TextDelta,
)
from core.tools.files import WriteTool


class BenchUsage:
    input_tokens = 10
    output_tokens = 5
    cache_read_tokens = 0
    cache_write_tokens = 0


class BenchChunk:
    def __init__(self, type, text="", name="", input=None,
                 stop_reason="", usage=None, tool_call_id=""):
        self.type = type
        self.text = text
        self.name = name
        self.input = input or {}
        self.stop_reason = stop_reason
        self.usage = usage
        self.tool_call_id = tool_call_id


class ScriptedChatAdapter:
    model = "benchmark"
    max_output_tokens = 8192

    def __init__(self, plans):
        self.plans = list(plans)

    async def chat_streaming(self, messages, system="", tools=None, max_tokens=None):
        plan = self.plans.pop(0) if self.plans else [
            BenchChunk("text_delta", text="done"),
            BenchChunk("message_stop", stop_reason="end_turn", usage=BenchUsage()),
        ]
        for chunk in plan:
            yield chunk


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=repo, text=True, errors="replace",
        stderr=subprocess.STDOUT,
    ).strip()


def _init_repo(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    _git(path, "init")
    _git(path, "config", "user.email", "bench@example.com")
    _git(path, "config", "user.name", "Benchmark")
    (path / "a.txt").write_text("base", encoding="utf-8")
    _git(path, "add", ".")
    _git(path, "commit", "-m", "base")


def _capability_factory(filename: str, content: str):
    def factory(tools, system_prompt, max_turns):
        tool_map = {tool.name: tool for tool in tools}

        class FakeLoop:
            async def run(self, task):
                await tool_map["Write"].execute(
                    file_path=filename, content=content
                )
                yield TextDelta(f"wrote {filename}")
                yield DoneEvent(LoopState(turn_count=1))

        return FakeLoop()

    return factory


def _agent_tool_with_verifier(repo: Path, verifier: CommandVerifier):
    manager = WorktreeManager(repo)
    return manager, AgentTool(
        agent_loop_factory=_capability_factory("feature.txt", "feature"),
        tool_registry={},
        worktree_manager=manager,
        workspace_tool_factory=lambda root: {
            "Write": WriteTool(workspace_root=root),
        },
        verifier_factory=lambda root: CommandVerifier(
            verifier.command,
            cwd=root,
            timeout_seconds=verifier.timeout_seconds,
        ),
        isolate_subagents=True,
    )


def _run_verified_and_apply(root: Path) -> dict:
    repo = root / "verified"
    _init_repo(repo)
    command = (
        f'"{sys.executable}" -c '
        '"from pathlib import Path; '
        'raise SystemExit(0 if Path(\'feature.txt\').exists() else 1)"'
    )
    manager, tool = _agent_tool_with_verifier(
        repo, CommandVerifier(command, timeout_seconds=5)
    )
    result = asyncio.run(tool.execute("verified change", agent_type="general"))
    entry = manager.list_entries()[0]
    mutations_before_apply = int((repo / "feature.txt").exists())
    applied_commit = manager.apply(entry["branch"])
    return {
        "case_id": "verified_and_apply",
        "passed": (
            "verification: passed" in result
            and mutations_before_apply == 0
            and entry["status"] == "ready"
            and (repo / "feature.txt").read_text(encoding="utf-8") == "feature"
        ),
        "verified_commits": int(bool(entry["commit"])),
        "main_workspace_mutations_before_apply": mutations_before_apply,
        "explicit_apply_count": int(bool(applied_commit)),
        "auto_merges": 0,
    }


def _run_failed_verification(root: Path) -> dict:
    repo = root / "failed"
    _init_repo(repo)
    manager, tool = _agent_tool_with_verifier(
        repo,
        CommandVerifier(
            f'"{sys.executable}" -c "raise SystemExit(1)"',
            timeout_seconds=5,
        ),
    )
    result = asyncio.run(tool.execute("bad change", agent_type="general"))
    entry = manager.list_entries()[0]
    return {
        "case_id": "failed_verification_preserved",
        "passed": (
            "verification: failed" in result
            and entry["status"] == "verify_failed"
            and not (repo / "feature.txt").exists()
            and tool.completion_blocker() is not None
        ),
        "preserved_uncommitted_worktree": int(entry["status"] == "verify_failed"),
        "main_workspace_mutations_before_apply": 0,
        "auto_merges": 0,
    }


def _run_parent_completion_gate(root: Path) -> dict:
    repo = root / "parent_gate"
    _init_repo(repo)
    manager = WorktreeManager(repo)
    agent_tool = AgentTool(
        agent_loop_factory=_capability_factory("feature.txt", "bad"),
        tool_registry={},
        worktree_manager=manager,
        workspace_tool_factory=lambda root: {
            "Write": WriteTool(workspace_root=root),
        },
        verifier_factory=lambda root: CommandVerifier(
            f'"{sys.executable}" -c "raise SystemExit(1)"',
            cwd=root,
            timeout_seconds=5,
        ),
        isolate_subagents=True,
    )
    plans = [
        [
            BenchChunk(
                "tool_use_start",
                name="Agent",
                input={"task": "bad isolated task", "agent_type": "general"},
                tool_call_id="call_agent",
            ),
            BenchChunk("message_stop", stop_reason="tool_use", usage=BenchUsage()),
        ],
        [
            BenchChunk("text_delta", text="done"),
            BenchChunk("message_stop", stop_reason="end_turn", usage=BenchUsage()),
        ],
        [
            BenchChunk("text_delta", text="still done"),
            BenchChunk("message_stop", stop_reason="end_turn", usage=BenchUsage()),
        ],
    ]
    loop = AgentLoop(
        tools=[agent_tool],
        model_adapter=ScriptedChatAdapter(plans),
        system_prompt="Benchmark.",
        max_turns=5,
        verify_retries=1,
    )

    async def collect():
        return [event async for event in loop.run("run isolated task")]

    events = asyncio.run(collect())
    done = next(event for event in events if isinstance(event, DoneEvent))
    return {
        "case_id": "parent_completion_blocked",
        "passed": done.state.termination_reason == TerminationReason.VERIFY_FAILED,
        "false_completion_blocked": int(
            done.state.termination_reason == TerminationReason.VERIFY_FAILED
        ),
        "main_workspace_mutations_before_apply": 0,
        "auto_merges": 0,
    }


def _run_conflict_rollback(root: Path) -> dict:
    repo = root / "conflict"
    _init_repo(repo)
    manager = WorktreeManager(repo)
    handle = manager.create("conflict")
    (handle.path / "a.txt").write_text("feature", encoding="utf-8")
    manager.commit(handle, "feature", verification_status="passed")
    manager.mark_verification(handle.branch, "passed", "ok")
    (repo / "a.txt").write_text("main", encoding="utf-8")
    _git(repo, "add", "a.txt")
    _git(repo, "commit", "-m", "main")

    raised = False
    try:
        manager.apply(handle.branch)
    except RuntimeError:
        raised = True
    entry = manager.find(handle.branch)
    rollback_ok = (
        raised
        and entry["status"] == "ready"
        and handle.path.exists()
        and _git(repo, "status", "--porcelain") == ""
        and not (repo / ".git" / "CHERRY_PICK_HEAD").exists()
    )
    return {
        "case_id": "conflict_rollback",
        "passed": rollback_ok,
        "rollback_ok": int(rollback_ok),
        "main_workspace_mutations_before_apply": 0,
        "auto_merges": 0 if rollback_ok else 1,
    }


def run_benchmark(output_dir: Path) -> dict:
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="minicode-bench-") as tmp:
        root = Path(tmp)
        cases = [
            _run_verified_and_apply(root),
            _run_failed_verification(root),
            _run_parent_completion_gate(root),
            _run_conflict_rollback(root),
        ]

    summary = {
        "mode": "scripted-offline",
        "cases_total": len(cases),
        "cases_passed": sum(1 for case in cases if case["passed"]),
        "false_completion_blocked": sum(
            case.get("false_completion_blocked", 0) for case in cases
        ),
        "verified_commits": sum(case.get("verified_commits", 0) for case in cases),
        "main_workspace_mutations_before_apply": sum(
            case.get("main_workspace_mutations_before_apply", 0)
            for case in cases
        ),
        "conflict_rollbacks": sum(
            case.get("rollback_ok", 0) for case in cases
        ),
        "auto_merges": sum(case.get("auto_merges", 0) for case in cases),
    }
    report = {"summary": summary, "cases": cases}
    (output_dir / "worktree_verification_bench.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
        newline="\n",
    )
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="artifacts")
    args = parser.parse_args()
    summary = run_benchmark(Path(args.output))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
