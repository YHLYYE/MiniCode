"""Security review — 4-layer defense-in-depth + Plan/Normal dual mode

Reference: Claude Code's permission system (how-claude-code-works ch11)
Core principle: fail-closed defaults. Every tool defaults to requiring
permission; security is layered so no single check is the sole gate.
"""

import asyncio
import json
import re
from enum import Enum
from pathlib import Path

from core.tools.base import ToolCall


class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SecurityBlock(Exception):
    """Raised when an operation is blocked by security rules.

    Caught by the Agent Loop and injected as a tool error message,
    triggering the error recovery path (ContinueReason.NEXT_TURN).
    """
    pass


# ── L1: Rule Filter (< 1ms, synchronous) ──

# Single source of truth for dangerous shell patterns.
# BashTool imports this list instead of keeping its own copy (two copies
# drifted apart: both had 10 patterns, and neither covered Windows deletion).
DANGEROUS_COMMAND_PATTERNS: list[tuple[str, str]] = [
    # ── Unix: recursive / forced deletion ──
    (r"\brm\s+(-[a-z]*[rf][a-z]*|--recursive|--force)", "recursive/forced deletion"),
    (r"\b(chmod|chown)\s+-R\b", "recursive permission change"),
    # ── Windows equivalents (previously unguarded) ──
    (r"\b(rd|rmdir)\b[^\n]*/\s?s\b", "recursive directory deletion"),
    (r"\bdel\b[^\n]*/\s?[sf]\b", "forced deletion"),
    (r"\b(remove-item|ri)\b[^\n]*-recurse", "recursive deletion"),
    (r"\bformat\s+[a-z]:", "filesystem format"),
    (r"\b(takeown|icacls)\b", "ownership / ACL takeover"),
    (r"\breg\s+delete\b", "registry deletion"),
    (r"\bdiskpart\b", "disk partitioning"),
    # ── Other destructive operations ──
    (r"\bgit\s+clean\b[^\n]*-[a-z]*f", "untracked file deletion"),
    # Discarding uncommitted work. `git checkout <branch>` / `-b <new>` are
    # daily work and must stay allowed, so these patterns require the explicit
    # discard forms (`--`, a bare `.` path, `-D`, `--hard`).
    (r"\bgit\s+reset\b[^\n]*--hard", "hard reset discards uncommitted work"),
    (r"\bgit\s+checkout\b[^\n]*(\s--(\s|$)|\s\.(\s|$))", "discards uncommitted changes"),
    (r"\bgit\s+restore\b", "discards working-tree changes"),
    # `-D` 必须区分大小写：`git branch -d` 只删已合并分支（安全），
    # 全局 IGNORECASE 会把两者一起拦掉。
    (r"\bgit\s+branch\b[^\n]*\s(?-i:-D)\b", "force-deletes a branch"),
    (r"\bgit\s+stash\s+(drop|clear)\b", "drops stashed work"),
    (r"\bsudo\b", "privilege escalation"),
    (r"\bchmod\s+777", "overly permissive permissions"),
    (r"(curl|wget)\b[^\n]*\|[^\n]*\b(sh|bash|zsh|python3?)\b", "remote script piped execution"),
    (r">\s*/dev/(?!null\b)[a-z]+", "system device overwrite"),
    (r"\bgit\s+push\b[^\n]*(--force\b|\s-f\b)", "force push"),
    (r"\b(DROP|TRUNCATE)\s+(TABLE|DATABASE)", "database destruction"),
    (r"\bmkfs\.", "filesystem format"),
    (r"\btruncate\s+-s\b", "truncates a file to zero"),
    (r"\bdd\s+if=", "direct disk I/O"),
    (r"/proc/|/sys/", "system filesystem access"),
    (r"\b(shutdown|reboot|halt|poweroff)\b", "system shutdown"),
    (r":\(\)\s*\{.*\}\s*;\s*:", "fork bomb"),
]

# Which tools take a filesystem path, and under which argument name.
# Grep/Glob were previously unconfined — `Grep(path="C:/")` could walk a whole drive.
_PATH_PARAM = {
    "Read": "file_path",
    "Write": "file_path",
    "Edit": "file_path",
    "Grep": "path",
    "Glob": "path",
}

class RuleFilter:
    """First line of defense — regex-based dangerous pattern detection.

    Checks commands against DANGEROUS_COMMAND_PATTERNS and validates
    every path-taking tool stays within the project directory.
    """

    DANGEROUS_PATTERNS = DANGEROUS_COMMAND_PATTERNS

    def __init__(self):
        self.path_allowlist = [Path.cwd()]

    def check(self, tool_call: ToolCall):
        """Check a tool call against security rules.
        Raises SecurityBlock if the operation is dangerous.
        """
        # Check shell commands
        if tool_call.name == "Bash":
            command = tool_call.input.get("command", "")
            for pattern, reason in self.DANGEROUS_PATTERNS:
                if re.search(pattern, command, re.IGNORECASE):
                    raise SecurityBlock(
                        f"Blocked ({reason}): {command[:100]}"
                    )

        # Check file paths: resolve symlinks/.. and confine to project
        if tool_call.name in _PATH_PARAM:
            raw = tool_call.input.get(_PATH_PARAM[tool_call.name]) or "."
            try:
                resolved = Path(raw).resolve()
            except (OSError, ValueError):
                raise SecurityBlock(f"Invalid path: {raw}")
            if not any(
                resolved.is_relative_to(Path(p).resolve())
                for p in self.path_allowlist
            ):
                raise SecurityBlock(f"Path outside project: {raw}")


# ── L2: Tool Self-Check (< 5ms, synchronous) ──

class ToolSelfCheck:
    """Second line — tool-specific parameter validation.

    Bash gets its own finer-grained rule (whitelisted read-only commands pass,
    everything else goes up). Every other tool that declares
    `is_destructive = True` (Write / Edit) is pushed to at least MEDIUM so it
    reaches L3 — previously they scored LOW here and were auto-approved, which
    meant the four-layer chain only ever applied to Bash.
    """

    COMMAND_WHITELIST: set[str] = {
        "ls", "cat", "head", "tail", "wc", "sort", "uniq",
        "grep", "find", "which", "pwd", "echo", "date",
    }

    def check(self, tool_call: ToolCall,
              is_destructive: bool = False) -> RiskLevel:
        if tool_call.name == "Bash":
            command = tool_call.input.get("command", "")

            # Output redirection is medium risk
            if ">" in command or ">>" in command:
                return RiskLevel.MEDIUM

            # Commands not in whitelist need deeper review
            base = command.strip().split()[0] if command.strip() else ""
            if base and base not in self.COMMAND_WHITELIST:
                return RiskLevel.MEDIUM

            return RiskLevel.LOW

        # 非 Bash 的破坏性工具（Write / Edit）：必须过 L3 风险分类，
        # 不能在这里拿 LOW 直接放行。
        if is_destructive:
            return RiskLevel.MEDIUM

        return RiskLevel.LOW


# ── L3: AI Risk Classifier (~500ms, async, optional) ──

class AIRiskClassifier:
    """Third line — LLM-based intent analysis.

    Detects: prompt injection patterns, scope violations,
    data exfiltration attempts, and irreversible operations.

    Every failure path returns HIGH (fail-closed): no model, classifier
    exception, unparseable output, or an out-of-vocabulary risk value.
    HIGH reaches L4, so the default answer to "the classifier did not
    answer" is "ask the human", never "let it through".
    """

    def __init__(self, model=None):
        self._model = model

    async def classify(self, tool_call: ToolCall) -> RiskLevel:
        if self._model is None:
            return RiskLevel.HIGH  # fail-closed：无分类器 → 视为高风险，交给人工

        prompt = f"""Analyze security risk. Output JSON only:
{{"risk": "low|medium|high|critical", "reason": "..."}}

Tool: {tool_call.name}
Input: {json.dumps(tool_call.input, indent=2)[:500]}"""

        try:
            result = await self._model.chat(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
            )
            # Extract JSON robustly (model may wrap in markdown fences)
            data = self._extract_json(result)
            risk = str(data.get("risk", "high")).lower()
            try:
                return RiskLevel(risk)
            except ValueError:
                return RiskLevel.HIGH  # 非法风险值 → fail-closed
        except Exception:
            return RiskLevel.HIGH  # 分类器异常 → fail-closed

    @staticmethod
    def _extract_json(text: str) -> dict:
        """Extract the first JSON object from model output.

        Handles markdown fences and prose, and scans for a *balanced* brace
        group. The earlier greedy `\\{.*\\}` swallowed two objects
        ("{...} {...}") into one unparseable string, which then fell through
        to the default — so the more the model said, the less we checked.
        """
        # Try direct parse first
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass
        # Walk every "{", take the slice up to its matching "}"
        for start in (m.start() for m in re.finditer(r"\{", text)):
            depth = 0
            for idx in range(start, len(text)):
                if text[idx] == "{":
                    depth += 1
                elif text[idx] == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            data = json.loads(text[start:idx + 1])
                            if isinstance(data, dict):
                                return data
                        except json.JSONDecodeError:
                            pass
                        break
        # Nothing parseable → fail-closed. Returning MEDIUM here meant a model
        # that answered with prose ("I cannot help with that") was *approved*
        # downstream, because authorize() lets LOW and MEDIUM through.
        return {"risk": "high"}


# ── Permission Manager (orchestrates all 4 layers) ──

class PermissionManager:
    """Orchestrates the 4-layer security check pipeline.

    Flow: L1 (rules) → L2 (self-check) → L3 (AI classify) → L4 (human)
    Each layer can short-circuit: L1/L2 reject immediately,
    L3 escalates to L4 for high/critical risks.
    """

    def __init__(self, model=None):
        self.rule_filter = RuleFilter()
        self.self_check = ToolSelfCheck()
        self.ai_classifier = AIRiskClassifier(model)

    async def authorize(self, tool_call: ToolCall,
                        is_destructive: bool) -> bool:
        """Run through all security layers. Returns True if allowed.

        `is_destructive` comes from the Tool object (the loop has it, the
        ToolCall doesn't). It is intentionally NOT defaulted: the old
        `= False` default meant a caller who forgot it silently downgraded
        Write/Edit to L2-LOW, auto-approving them and skipping L3+L4
        entirely — a four-layer chain quietly becoming two layers.
        """
        # L1: Rule filter — raises SecurityBlock on danger
        self.rule_filter.check(tool_call)

        # L2: Tool self-check
        risk = self.self_check.check(tool_call, is_destructive=is_destructive)
        if risk == RiskLevel.LOW:
            return True

        # L3: AI risk classifier
        risk = await self.ai_classifier.classify(tool_call)
        if risk in (RiskLevel.LOW, RiskLevel.MEDIUM):
            return True

        # L4: Human approval (high/critical only)
        return await self._request_approval(tool_call, risk)

    async def _request_approval(
        self, tool_call: ToolCall, risk: RiskLevel
    ) -> bool:
        """Block and wait for user confirmation."""
        print(f"""
{'!' * 60}
⚠  HIGH RISK OPERATION ({risk.value})
Tool: {tool_call.name}
Input: {json.dumps(tool_call.input, indent=2)[:300]}
{'!' * 60}
""")
        try:
            # 同步 input() 会阻塞事件循环 → 丢线程池
            response = await asyncio.to_thread(input, "Execute? [y/N]: ")
            return response.strip().lower() == "y"
        except (EOFError, KeyboardInterrupt):
            return False
        except Exception:
            # stdin 被重定向或关闭时（非交互运行、管道输入）input() 抛 OSError。
            # L4 必须 fail-closed：问不到人 = 不允许，不能让异常把任务打崩。
            print("[无法获取人工确认（stdin 不可读）→ 按 fail-closed 拒绝执行]")
            return False


# ── Plan / Normal Dual Mode ──

class ExecutionMode(Enum):
    PLAN = "plan"       # Read-only tools only
    NORMAL = "normal"   # Full toolset


# Plan 模式承诺的是「不碰用户的代码」，而这个工具只写 agent 自己的记账文件
# （.minicode/todos.md）。所以它可以进 Plan 模式 —— 否则「只读规划模式」里
# 连待办清单都写不了。
PLAN_MODE_EXTRA_TOOLS = {"TodoWrite"}


def resolve_tools_for_mode(tools: list, mode: ExecutionMode) -> list:
    """Filter tools based on execution mode.
    Plan mode: API-level isolation — write tools are physically removed.
    Even if the model hallucinates a write tool call, it cannot execute.

    TodoWrite is the one exception: Plan mode promises "don't touch the user's
    code", and the task list only writes to the agent's own bookkeeping file
    (.minicode/todos.md). Without it you cannot even write a plan in plan mode.
    """
    if mode == ExecutionMode.PLAN:
        return [
            t for t in tools
            if getattr(t, "is_readonly", False)
            or t.name in PLAN_MODE_EXTRA_TOOLS
        ]
    return tools
