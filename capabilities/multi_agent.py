"""Multi-agent collaboration — AgentTool unified routing

Reference: Claude Code's AgentTool (how-claude-code-works ch07)
Core insight: Sub-agents reuse the same turn engine with different params.

Collaboration modes:
- explore / general: single sub-agent with toolset filter
- team: coordinator decomposes task → parallel role sub-agents → merge
"""

import asyncio

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
            "tools_filter": ["Read", "Grep", "Glob", "WebSearch", "WebFetch"],
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
    }

    # Team mode roles — coordinator dispatches to these in parallel
    TEAM_ROLES = ["research", "coding", "testing"]

    _semaphore = asyncio.Semaphore(5)  # Max concurrent sub-agents

    def __init__(self, agent_loop_factory=None, tool_registry=None):
        super().__init__()
        self._agent_factory = agent_loop_factory
        self._tool_registry = tool_registry or {}

    async def execute(self, task: str, agent_type: str = "general") -> str:
        if agent_type == "team":
            return await self._run_team(task)

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

    async def _run_subagent(self, task: str, profile: dict) -> dict:
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
