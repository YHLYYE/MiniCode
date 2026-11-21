"""Multi-agent collaboration — AgentTool unified routing

Reference: Claude Code's AgentTool (how-claude-code-works ch07)
Core insight: Sub-agents reuse the same turn engine with different params.

Collaboration modes:
- explore / general: single sub-agent with toolset filter
- team: research → coding → testing pipeline, each step fed the previous output
"""

import asyncio

from core.tools.base import Tool


MAX_PROJECT_RULES_CHARS = 2000

_RULES_HEADER = "--- 项目约定（CLAUDE.md）---"


def compose_subagent_prompt(role_prompt: str,
                            project_rules: str | None = None) -> str:
    """子 Agent 的系统提示词 = 角色提示词 + 项目约定（若有）。

    为什么需要拼这一步：子 Agent 的上下文只有「角色提示词 + 任务描述」，父级的
    系统提示词（含 CLAUDE.md、会话摘要、技能索引）**不会传下去**（新建 AgentLoop
    时只给 profile["system_prompt"]）。这样一来，研究阶段定下的规范约束不到编码
    阶段——两个角色都不知道项目约定。

    为什么不用「给 research 配 Remember 工具、让 coding 去查」那条路：约定是**规则**，
    每个角色都该遵守，应该主动注入；靠记忆库传递等于要求下游角色"碰巧想起来去查"，
    而且会把项目规则散进三类记忆里，和 CLAUDE.md 形成两套真相来源。

    截断是为了防止一份超长 CLAUDE.md 把每个子 Agent 的上下文都吃掉（上限定成
    和会话摘要同一个量级）。
    """
    rules = (project_rules or "").strip()
    if not rules:
        return role_prompt
    return f"{role_prompt}\n\n{_RULES_HEADER}\n{rules[:MAX_PROJECT_RULES_CHARS]}"


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
    # 并发安全**不能**用类属性一概声明：是否安全取决于运行时的 agent_type ——
    # explore 只读，可以和其他工具并行；general / team 能写文件（team 的 coding
    # 角色拿全量工具），同一回合里并行跑两个就可能同时改同一批文件。
    # 基类的 `check_concurrency_safe(input)` 钩子正是为「按参数变化」准备的，
    # 所以这里声明成 fail-closed 的 False，再由下面的钩子按 agent_type 放行。
    is_concurrency_safe = False

    def check_concurrency_safe(self, input: dict) -> bool:
        """只有只读的 explore 允许与其他工具并行执行。

        拿不准的一律按不安全处理（fail-closed）：没写 agent_type 时 default
        是 general，同样返回 False。
        """
        return (input or {}).get("agent_type") == "explore"

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

    def __init__(
        self,
        agent_loop_factory=None,
        tool_registry=None,
        worktree_manager=None,
        workspace_tool_factory=None,
        verifier_factory=None,
        isolate_subagents: bool = False,
    ):
        super().__init__()
        self._agent_factory = agent_loop_factory
        self._tool_registry = tool_registry or {}
        self._worktree_manager = worktree_manager
        self._workspace_tool_factory = workspace_tool_factory
        self._verifier_factory = verifier_factory
        self._isolate_subagents = isolate_subagents
        # 子 Agent 用完的 token / 成本攒在这里，等父级 drain_usage() 取走
        self._pending_tokens = 0
        self._pending_cost = 0.0
        self._completion_error: str | None = None

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

    def completion_blocker(self) -> str | None:
        return self._completion_error

    async def execute(self, task: str, agent_type: str = "general") -> str:
        if agent_type == "team":
            return await self._run_team(task)

        profile = self.AGENT_PROFILES.get(agent_type)
        if profile is None:
            # 模型幻觉出非法类型时，走「子 Agent 失败」同一条路：返回一段文本让
            # 父级看见并纠正，而不是抛 KeyError 冒到工具边界之外。
            return (f"Unknown agent_type: {agent_type!r}. "
                    f"Use one of {sorted(self.AGENT_PROFILES)} or 'team'.")
        async with self._semaphore:
            handle = None
            try:
                if (
                    agent_type == "general"
                    and self._isolate_subagents
                    and self._worktree_manager is not None
                    and self._workspace_tool_factory is not None
                ):
                    handle = self._worktree_manager.create(task)
                    workspace_tools = self._workspace_tool_factory(handle.path)
                    result = await self._run_subagent(
                        task, profile, tool_registry=workspace_tools
                    )
                    verification = None
                    if self._verifier_factory is not None:
                        verification = await self._verifier_factory(
                            handle.path
                        ).verify()
                    if verification is not None and not verification.passed:
                        self._worktree_manager.mark_verification(
                            handle.branch, "failed", verification.output
                        )
                        commit = None
                        diff = self._worktree_manager.working_tree_status(handle)
                        changed = self._worktree_manager.working_tree_changed_files(
                            handle
                        )
                        verify_line = (
                            "verification: failed\n"
                            f"verify_exit_code: {verification.exit_code}\n"
                            f"verify_evidence:\n{verification.output}"
                        )
                        self._completion_error = (
                            f"isolated subagent verification failed: "
                            f"exit={verification.exit_code}, "
                            f"branch={handle.branch}, path={handle.path}"
                        )
                    else:
                        verification_status = (
                            "passed" if verification is not None
                            else "not_configured"
                        )
                        commit = self._worktree_manager.commit(
                            handle,
                            f"minicode subagent: {task[:60]}",
                            verification_status=verification_status,
                        )
                        diff = self._worktree_manager.diff_stat(handle)
                        changed = self._worktree_manager.changed_files(handle)
                        verify_line = (
                            "verification: passed"
                            if verification is not None
                            else "verification: not_configured"
                        )
                        self._completion_error = None
                    result["worktree_header"] = (
                        f"branch: {handle.branch}\n"
                        f"commit: {commit or 'none'}\n"
                        f"path: {handle.path}\n"
                        f"uncommitted: {'true' if commit is None else 'false'}\n"
                        f"{verify_line}\n"
                        f"changed_files: {changed}\n"
                        f"diff:\n{diff}"
                    )
                else:
                    result = await self._run_subagent(task, profile)
            except Exception as e:
                if handle is not None:
                    self._worktree_manager.mark_failed(handle.branch, str(e))
                    self._completion_error = (
                        f"isolated subagent failed: {e}; "
                        f"branch={handle.branch}, path={handle.path}"
                    )
                    return (
                        f"Sub-agent ({agent_type}) failed: {e}\n"
                        f"branch: {handle.branch}\n"
                        f"path: {handle.path}\n"
                        "status: failed (worktree preserved)"
                    )
                return f"Sub-agent ({agent_type}) failed: {e}"
        self._record_subagent_usage(result)

        # 用 .get 读：任何一个提前返回都可能在某个键上缺项，而这里只是拼一行
        # 展示用的抬头，不该让缺键把整个工具调用打成异常。
        turns = result.get("turns", 0)
        tokens = result.get("tokens", 0)
        cost = result.get("cost_usd", 0.0)
        return (
            f"[Sub-agent: {agent_type}, {turns} turns, "
            f"{tokens} tokens, ${cost:.4f}]\n"
            f"{result.get('worktree_header', '')}\n\n"
            f"{result.get('output', '')}"
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

        if (
            self._isolate_subagents
            and self._worktree_manager is not None
            and self._workspace_tool_factory is not None
        ):
            return await self._run_team_isolated(task, role_prompts)

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

    async def _run_team_isolated(self, task: str, role_prompts: dict) -> str:
        try:
            handle = self._worktree_manager.create(task)
        except Exception as exc:
            return f"Agent Team failed: {exc}"

        workspace_tools = self._workspace_tool_factory(handle.path)
        results: list[dict] = []
        upstream_role = ""
        upstream_output = ""

        for role in self.TEAM_ROLES:
            step_task = task
            if upstream_output:
                step_task = (
                    f"{task}\n\n--- Output from the {upstream_role} step ---\n"
                    f"{upstream_output}"
                )
            profile = {
                "tools_filter": self.TEAM_ROLE_TOOLS[role],
                "max_turns": 10,
                "system_prompt": role_prompts[role],
            }
            registry = self._tool_registry if role == "research" else workspace_tools
            async with self._semaphore:
                try:
                    result = await self._run_subagent(
                        step_task, profile, tool_registry=registry
                    )
                    entry = {"role": role, "ok": True, **result}
                except Exception as exc:
                    entry = {
                        "role": role,
                        "ok": False,
                        "output": str(exc),
                        "turns": 0,
                        "tokens": 0,
                        "cost_usd": 0.0,
                    }
            results.append(entry)
            self._record_subagent_usage(entry)
            if not entry["ok"]:
                break
            upstream_role, upstream_output = role, entry["output"]

        passed = len(results) == len(self.TEAM_ROLES) and all(
            entry["ok"] for entry in results
        )
        commit = None
        verification = None
        if not passed:
            self._worktree_manager.mark_failed(
                handle.branch, "isolated team workflow failed"
            )
            self._completion_error = (
                "isolated team workflow failed; "
                f"branch={handle.branch}, path={handle.path}"
            )
        if passed:
            if self._verifier_factory is not None:
                try:
                    verification = await self._verifier_factory(
                        handle.path
                    ).verify()
                except Exception as exc:
                    verification = None
                    self._completion_error = (
                        f"isolated team verification failed to run: {exc}; "
                        f"branch={handle.branch}, path={handle.path}"
                    )
                    return (
                        f"Agent Team verification failed to run: {exc}\n"
                        f"branch: {handle.branch}\n"
                        f"path: {handle.path}\n"
                    )
            if verification is not None and not verification.passed:
                self._worktree_manager.mark_verification(
                    handle.branch, "failed", verification.output
                )
                self._completion_error = (
                    "isolated team verification failed: "
                    f"exit={verification.exit_code}, "
                    f"branch={handle.branch}, path={handle.path}"
                )
                lines = [
                    f"[Agent Team: {len(results)}/{len(self.TEAM_ROLES)} steps, "
                    "isolated]\n"
                    "status: verification_failed\n"
                    f"branch: {handle.branch}\n"
                    "commit: none\n"
                    f"path: {handle.path}\n"
                    "uncommitted: true\n"
                    f"changed_files: {self._worktree_manager.working_tree_changed_files(handle)}\n"
                    f"working_tree:\n{self._worktree_manager.working_tree_status(handle)}\n"
                    "verification: failed\n"
                    f"verify_exit_code: {verification.exit_code}\n"
                    f"verify_evidence:\n{verification.output}\n"
                ]
                for entry in results:
                    lines.append(f"\n## {entry['role']} ✓")
                    lines.append(entry["output"])
                return "\n".join(lines)
            try:
                verification_status = (
                    "passed" if verification is not None else "not_configured"
                )
                commit = self._worktree_manager.commit(
                    handle,
                    f"minicode team: {task[:60]}",
                    verification_status=verification_status,
                )
            except Exception as exc:
                self._completion_error = (
                    f"isolated team commit failed: {exc}; "
                    f"branch={handle.branch}, path={handle.path}"
                )
                return f"Agent Team commit failed: {exc}\npath: {handle.path}"

        if passed and verification is not None and verification.passed:
            self._completion_error = None
        elif passed and verification is None:
            self._completion_error = None

        lines = [
            f"[Agent Team: {len(results)}/{len(self.TEAM_ROLES)} steps, isolated]\n"
            f"branch: {handle.branch}\n"
            f"commit: {commit or 'none'}\n"
            f"path: {handle.path}\n"
            f"verification: {'passed' if verification is not None else 'not_configured'}\n"
        ]
        if passed:
            lines.append(
                f"changed_files: {self._worktree_manager.changed_files(handle)}\n"
                f"diff:\n{self._worktree_manager.diff_stat(handle)}\n"
            )
        for entry in results:
            status = "✓" if entry["ok"] else "✗"
            lines.append(f"\n## {entry['role']} {status}")
            lines.append(entry["output"])
        return "\n".join(lines)

    # ── Sub-agent runner ──

    async def _run_subagent(
        self, task: str, profile: dict, tool_registry: dict | None = None
    ) -> dict:
        from core.state import TextDelta, DoneEvent

        # Resolve tools
        registry = self._tool_registry if tool_registry is None else tool_registry
        if profile["tools_filter"]:
            tools = [
                t for t in registry.values()
                if t.name in profile["tools_filter"]
            ]
        else:
            tools = list(registry.values())

        if self._agent_factory is None:
            # 四个键必须齐全：execute 会把它们格式化进返回抬头。
            # 早先这里漏了 cost_usd，于是「优雅提示」这条分支自己在格式化时崩溃。
            return {"output": "Agent factory not configured",
                    "turns": 0, "tokens": 0, "cost_usd": 0.0}

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
