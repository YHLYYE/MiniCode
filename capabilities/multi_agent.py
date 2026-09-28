"""Multi-agent collaboration — AgentTool unified routing

Reference: Claude Code's AgentTool (how-claude-code-works ch07)
Core insight: Sub-agents reuse the same turn engine with different params.

Collaboration modes:
- explore / general: single sub-agent with toolset filter
- team: research → coding → testing pipeline, each step fed the previous output
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
            "description": "explore (read-only search) | general (full toolset) "
                           "| team (research → coding → testing pipeline)",
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

    # Team mode roles — 串行流水线：研究 → 编码 → 验证
    # 角色之间是有依赖的，并行没有意义：testing 在 coding 写完之前开测，
    # 测的是改之前的代码。
    TEAM_ROLES = ["research", "coding", "testing"]

    # 每个角色只拿该拿的工具。此前三个角色都拿全量工具，
    # 于是 research 和 testing 也能写文件，而它们和 coding 是同一批文件。
    TEAM_ROLE_TOOLS: dict[str, list[str] | None] = {
        "research": ["Read", "Grep", "Glob", "WebSearch", "WebFetch"],
        "coding": None,  # None = 全部工具（唯一能写文件的角色）
        "testing": ["Read", "Grep", "Glob", "Bash"],
    }

    _semaphore = asyncio.Semaphore(5)  # Max concurrent sub-agents

    def __init__(self, agent_loop_factory=None, tool_registry=None):
        super().__init__()
        self._agent_factory = agent_loop_factory
        self._tool_registry = tool_registry or {}
        # 子 Agent 用完的 token / 成本攒在这里，等父级 drain_usage() 取走
        self._pending_tokens = 0
        self._pending_cost = 0.0

    # ── 用量回流 ──

    def drain_usage(self) -> tuple[int, float] | None:
        """交出自上次调用以来子 Agent 的 (tokens, cost_usd)，并清零。

        Agent Loop 每执行完一个工具就会调这个（如果工具定义了它），把用量
        累加进父级账本。这是 `--max-cost` 能覆盖子 Agent 的唯一通路。
        """
        if self._pending_tokens == 0 and self._pending_cost == 0.0:
            return None
        usage = (self._pending_tokens, self._pending_cost)
        self._pending_tokens = 0
        self._pending_cost = 0.0
        return usage

    def _record_subagent_usage(self, result: dict) -> None:
        self._pending_tokens += result.get("tokens", 0)
        self._pending_cost += result.get("cost_usd", 0.0)

    async def execute(self, task: str, agent_type: str = "general") -> str:
        if agent_type == "team":
            return await self._run_team(task)

        profile = self.AGENT_PROFILES[agent_type]
        async with self._semaphore:
            try:
                result = await self._run_subagent(task, profile)
            except Exception as e:
                return f"Sub-agent ({agent_type}) failed: {e}"
        self._record_subagent_usage(result)

        return (
            f"[Sub-agent: {agent_type}, {result['turns']} turns, "
            f"{result['tokens']} tokens, ${result['cost_usd']:.4f}]\n\n"
            f"{result['output']}"
        )

    # ── Team mode: research → coding → testing pipeline ──

    async def _run_team(self, task: str) -> str:
        """Three-role pipeline: research → coding → testing.

        Each role is a sub-agent with an isolated context window that returns
        only a summary. Each step receives the previous step's output, so the
        testing step verifies the code the coding step just wrote. Roles also
        get role-scoped tools — research/testing cannot write files.
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

        results: list[dict] = []
        upstream_role = ""
        upstream_output = ""

        for role in self.TEAM_ROLES:
            step_task = task
            if upstream_output:
                step_task = (
                    f"{task}\n\n"
                    f"--- Output from the {upstream_role} step ---\n"
                    f"{upstream_output}"
                )
            profile = {
                "tools_filter": self.TEAM_ROLE_TOOLS[role],
                "max_turns": 10,
                "system_prompt": role_prompts[role],
            }
            async with self._semaphore:
                try:
                    result = await self._run_subagent(step_task, profile)
                    entry = {"role": role, "ok": True, **result}
                except Exception as e:
                    entry = {"role": role, "ok": False, "output": str(e),
                             "turns": 0, "tokens": 0}
            results.append(entry)
            self._record_subagent_usage(entry)
            if not entry["ok"]:
                break  # 上一步失败 → 后面的步骤没有输入可接
            upstream_role, upstream_output = role, entry["output"]

        # Merge into a sequential summary (每一步都标出是否成功)
        lines = [
            f"[Agent Team: {len(results)}/{len(self.TEAM_ROLES)} steps, "
            f"task: {task[:80]}]\n"
        ]
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
        cost = 0.0
        async for event in sub.run(task):
            if isinstance(event, TextDelta):
                output_parts.append(event.text)
            elif isinstance(event, DoneEvent):
                turns = event.state.turn_count
                tokens = event.state.total_tokens
                cost = event.state.total_cost_usd

        return {
            "output": "".join(output_parts),
            "turns": turns,
            "tokens": tokens,
            "cost_usd": cost,
        }
