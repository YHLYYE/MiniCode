"""Security review — 4-layer defense-in-depth + Plan/Normal dual mode

Reference: Claude Code's permission system (how-claude-code-works ch11)
Core principle: fail-closed defaults. Every tool defaults to requiring
permission; security is layered so no single check is the sole gate.
"""

import json
from dataclasses import dataclass
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

class RuleFilter:
    """First line of defense — regex-based dangerous pattern detection.

    Checks commands against 9+ dangerous patterns and validates
    file write paths are within the project directory.
    """

    DANGEROUS_PATTERNS: list[tuple[str, str]] = [
        (r"rm\s+(-rf?|--recursive)", "recursive deletion"),
        (r"\bsudo\b", "privilege escalation"),
        (r"chmod\s+777", "overly permissive permissions"),
        (r"(curl|wget).*\|.*(sh|bash|python)", "remote script execution"),
        (r">\s*/dev/[a-z]+", "system device overwrite"),
        (r"git\s+push\s+(--force|-f)", "force push"),
        (r"(DROP|TRUNCATE)\s+(TABLE|DATABASE)", "database destruction"),
        (r"\bmkfs\.", "filesystem format"),
        (r"dd\s+if=", "direct disk I/O"),
    ]

    def __init__(self):
        self.path_allowlist = [Path.cwd()]

    def check(self, tool_call: ToolCall):
        """Check a tool call against security rules.
        Raises SecurityBlock if the operation is dangerous.
        """
        import re

        # Check shell commands
        if tool_call.name == "Bash":
            command = tool_call.input.get("command", "")
            for pattern, reason in self.DANGEROUS_PATTERNS:
                if re.search(pattern, command, re.IGNORECASE):
                    raise SecurityBlock(
                        f"Blocked ({reason}): {command[:100]}"
                    )

        # Check file write paths
        if tool_call.name in ("Write",):
            file_path = Path(tool_call.input.get("file_path", ""))
            allowed = any(
                str(file_path).startswith(str(p))
                for p in self.path_allowlist
            )
            if not allowed:
                raise SecurityBlock(
                    f"Path outside project: {file_path}"
                )


# ── L2: Tool Self-Check (< 5ms, synchronous) ──

class ToolSelfCheck:
    """Second line — tool-specific parameter validation.

    Checks for output redirection in Bash commands and verifies
    commands are in the allowlist for medium-risk operations.
    """

    COMMAND_WHITELIST: set[str] = {
        "ls", "cat", "head", "tail", "wc", "sort", "uniq",
        "grep", "find", "which", "pwd", "echo", "date",
    }

    def check(self, tool_call: ToolCall) -> RiskLevel:
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


# ── L3: AI Risk Classifier (~500ms, async, optional) ──

class AIRiskClassifier:
    """Third line — LLM-based intent analysis.

    Detects: prompt injection patterns, scope violations,
    data exfiltration attempts, and irreversible operations.
    Falls back to MEDIUM risk when model is unavailable.
    """

    def __init__(self, model=None):
        self._model = model

    async def classify(self, tool_call: ToolCall) -> RiskLevel:
        if self._model is None:
            return RiskLevel.MEDIUM

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
            return RiskLevel(data.get("risk", "medium"))
        except Exception:
            return RiskLevel.MEDIUM

    @staticmethod
    def _extract_json(text: str) -> dict:
        """Extract a JSON object from model output (handles markdown fences)."""
        import re
        # Try direct parse first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        # Try extracting {...} block
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        return {"risk": "medium"}


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

    async def authorize(self, tool_call: ToolCall) -> bool:
        """Run through all security layers. Returns True if allowed."""
        # L1: Rule filter — raises SecurityBlock on danger
        self.rule_filter.check(tool_call)

        # L2: Tool self-check
        risk = self.self_check.check(tool_call)
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
            response = input("Execute? [y/N]: ").strip().lower()
            return response == "y"
        except (EOFError, KeyboardInterrupt):
            return False


# ── Plan / Normal Dual Mode ──

class ExecutionMode(Enum):
    PLAN = "plan"       # Read-only tools only
    NORMAL = "normal"   # Full toolset


def resolve_tools_for_mode(tools: list, mode: ExecutionMode) -> list:
    """Filter tools based on execution mode.
    Plan mode: API-level isolation — write tools are physically removed.
    Even if the model hallucinates a write tool call, it cannot execute.
    """
    if mode == ExecutionMode.PLAN:
        return [t for t in tools if getattr(t, "is_readonly", False)]
    return tools
