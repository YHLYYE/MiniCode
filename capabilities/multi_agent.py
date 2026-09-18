"""Multi-agent collaboration — AgentTool unified routing

Reference: Claude Code's AgentTool (how-claude-code-works ch07)
Core insight: Sub-agents reuse the same turn engine with different params.

Collaboration modes:
- explore / general: single sub-agent with toolset filter
- team: coordinator decomposes task → parallel role sub-agents → merge
- (worktree retired: cwd isolation not implemented, removed from the enum)
"""

import asyncio
import subprocess
from pathlib import Path

from core.tools.base import Tool


class AgentTool(Tool):
    """Launch sub-agents to handle independent tasks.

    The model calls this tool when a task can be decomposed into
    independent sub-tasks. Sub-agents have isolated context windows
    and return only result summaries — preventing context pollution.
    """

    name = "Agent"
    description = (
        "Launch a sub-agent (or team of sub-agents) for independent work. "
        "agent_type: explore (read-only search) | general (full toolset) | "
        "team (coordinated roles)."
    )
    input_schema = {
        "task": {
            "type": "string",
            "description": "The task for the sub-agent(s) to complete",
        },
        "agent_type": {
            "type": "string",
            "enum": ["explore", "general", "team"],
            "description": "explore | general | team",
        },
    }
    is_concurrency_safe = True

    AGENT_PROFILES = {
        "explore": {
            "tools_filter": ["Read", "RecallMemory"],
            "max_turns": 10,
            "system_prompt": (
                "You are a code explorer. Find relevant code and report "
                "findings. Be thorough but concise. Your output IS the "
                "deliverable — do not ask follow-up questions."
            ),
        },
        "general": {
            "tools_filter": None,  # None = all tools
            "max_turns": 20,
            "system_prompt": (
                "You are a sub-agent. Complete the assigned task "
                "independently. Return a concise summary of what you "
                "did and what you found. Your output IS the deliverable."
            ),
        },
        "worktree": {
            "tools_filter": None,
            "max_turns": 20,
            "system_prompt": (
                "You are a sub-agent working in an isolated git worktree. "
                "Complete the assigned task independently. Your changes are "
                "isolated from the main branch. Return a summary of changes."
            ),
        },
    }

    # Team mode roles — coordinator dispatches to these in parallel
    TEAM_ROLES = ["research", "coding", "testing"]

    _semaphore = asyncio.Semaphore(5)  # Max concurrent sub-agents

    def __init__(self, agent_loop_factory=None, tool_registry=None,
                 project_root: Path | None = None):
        super().__init__()
        self._agent_factory = agent_loop_factory
        self._tool_registry = tool_registry or {}
        self._project_root = project_root or Path.cwd()

    async def execute(self, task: str, agent_type: str = "general") -> str:
        if agent_type == "team":
            return await self._run_team(task)
        if agent_type == "worktree":
            return await self._run_worktree(task)

        profile = self.AGENT_PROFILES[agent_type]
        async with self._semaphore:
            try:
                result = await self._run_subagent(task, profile)
            except Exception as e:
                return f"Sub-agent ({agent_type}) failed: {e}"

        return (
            f"[Sub-agent: {agent_type}, {result['turns']} turns, "
            f"{result['tokens']} tokens]\n\n{result['output']}"
        )

    # ── Worktree mode: git isolation for parallel write tasks ──

    async def _run_worktree(self, task: str) -> str:
        """Run a sub-agent in an isolated git worktree.

        RETIRED — removed from the agent_type enum. Worktree cwd isolation
        is not implemented (sub-agents run in the parent working directory,
        so this path gives a false sense of isolation). Kept only for
        reference; not reachable via normal tool invocation.
        """
        if not self._is_git_repo():
            return (
                "[Worktree mode degraded: not a git repository. "
                "Running as general sub-agent instead.]\n\n"
                + await self.execute(task, agent_type="general")
            )

        profile = self.AGENT_PROFILES["worktree"]
        branch = f"minicode-worktree-{int(asyncio.get_event_loop().time())}"

        try:
            # Create isolated worktree
            self._git("worktree", "add", "-b", branch, "worktree_dir")
            worktree_path = self._project_root / "worktree_dir"

            async with self._semaphore:
                result = await self._run_subagent(task, profile, cwd=worktree_path)

            # Report diff summary
            diff = self._git("diff", "--stat", "main", branch)

        except Exception as e:
            return f"Worktree mode failed: {e}"
        finally:
            # Clean up worktree
            try:
                self._git("worktree", "remove", "worktree_dir", "--force")
                self._git("branch", "-D", branch)
            except Exception:
                pass  # cleanup is best-effort

        return (
            f"[Worktree mode, {result['turns']} turns]\n\n"
            f"{result['output']}\n\n[Changes (main vs {branch})]\n{diff}"
        )

    # ── Team mode: coordinator + parallel role sub-agents ──

    async def _run_team(self, task: str) -> str:
        """Coordinator decomposes task → parallel role sub-agents → merge results.

        Uses 3 role sub-agents (research/coding/testing) running in parallel,
        each with a role-specific system prompt. Merges their outputs into
        a coordinated result summary.
        """
        role_prompts = {
            "research": (
                "You are a research sub-agent. Analyze the task and identify "
                "what needs to change, relevant files, and dependencies. "
                "Return a structured analysis."
            ),
            "coding": (
                "You are a coding sub-agent. Implement the required changes "
                "based on the task. Return what you changed and why."
            ),
            "testing": (
                "You are a testing sub-agent. Determine how to verify the "
                "changes work. Return a test plan and expected results."
            ),
        }

        async def run_role(role: str) -> dict:
            profile = {
                "tools_filter": None,
                "max_turns": 10,
                "system_prompt": role_prompts[role],
            }
            async with self._semaphore:
                try:
                    result = await self._run_subagent(task, profile)
                    return {"role": role, "ok": True, **result}
                except Exception as e:
                    return {"role": role, "ok": False, "output": str(e),
                            "turns": 0, "tokens": 0}

        # Run all roles in parallel
        results = await asyncio.gather(
            *(run_role(role) for role in self.TEAM_ROLES)
        )

        # Merge into coordinated summary
        lines = [f"[Agent Team: {len(results)} roles, task: {task[:80]}]\n"]
        for r in results:
            status = "✓" if r["ok"] else "✗"
            lines.append(f"\n## {r['role']} {status} ({r['turns']} turns)")
            lines.append(r["output"])
        return "\n".join(lines)

    # ── Sub-agent runner ──

    async def _run_subagent(self, task: str, profile: dict,
                            cwd: Path | None = None) -> dict:
        from core.agent_loop import AgentLoop
        from core.state import TextDelta, DoneEvent

        # Resolve tools
        if profile["tools_filter"]:
            tools = [
                t for t in self._tool_registry.values()
                if t.name in profile["tools_filter"]
            ]
        else:
            tools = list(self._tool_registry.values())

        if self._agent_factory is None:
            return {"output": "Agent factory not configured", "turns": 0, "tokens": 0}

        sub = self._agent_factory(
            tools=tools,
            system_prompt=profile["system_prompt"],
            max_turns=profile["max_turns"],
        )

        output_parts = []
        turns = 0
        tokens = 0
        async for event in sub.run(task):
            if isinstance(event, TextDelta):
                output_parts.append(event.text)
            elif isinstance(event, DoneEvent):
                turns = event.state.turn_count
                tokens = event.state.total_tokens

        return {
            "output": "".join(output_parts),
            "turns": turns,
            "tokens": tokens,
        }

    # ── Git helpers ──

    def _is_git_repo(self) -> bool:
        try:
            subprocess.run(
                ["git", "rev-parse", "--is-inside-work-tree"],
                cwd=str(self._project_root), capture_output=True,
                check=True, timeout=5,
            )
            return True
        except Exception:
            return False

    def _git(self, *args: str) -> str:
        result = subprocess.run(
            ["git", *args], cwd=str(self._project_root),
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or result.stdout.strip())
        return result.stdout.strip()
