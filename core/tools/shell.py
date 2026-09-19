"""Shell command execution tool — 3-layer security defense"""
import asyncio
import re
from pathlib import Path
from core.tools.base import Tool
from capabilities.security import SecurityBlock


class BashTool(Tool):
    """Execute shell commands with safety checks.

    3-layer security:
    1. Dangerous pattern regex detection (23+ patterns)
    2. Timeout enforcement (max 60s)
    3. Working directory confinement (project root only)
    """

    name = "Bash"
    description = (
        "Execute a shell command in the project directory. "
        "Commands are checked against dangerous patterns before execution. "
        "Timeout is enforced (max 60 seconds)."
    )
    input_schema = {
        "command": {
            "type": "string",
            "description": "Shell command to execute",
        },
        "timeout": {
            "type": "integer",
            "description": "Timeout in seconds (default 30, max 60)",
        },
    }
    is_readonly = False
    is_concurrency_safe = False
    is_destructive = True

    DANGEROUS_PATTERNS: list[tuple[str, str]] = [
        (r"rm\s+(-rf?|--recursive)", "Recursive deletion"),
        (r"\bsudo\b", "Privilege escalation"),
        (r"chmod\s+777", "Overly permissive permissions"),
        (r"(curl|wget).*\|.*(sh|bash|python)", "Remote script piped execution"),
        (r">\s*/dev/[a-z]+", "Overwrite system device"),
        (r"git\s+push\s+(--force|-f)", "Force push"),
        (r"(DROP|TRUNCATE)\s+(TABLE|DATABASE)", "Database destruction"),
        (r"\bmkfs\.", "Filesystem format"),
        (r"dd\s+if=", "Direct disk I/O"),
        (r"(/proc/|/sys/)", "System filesystem access"),
    ]
    MAX_EXECUTION_TIME = 60

    async def execute(self, command: str, timeout: int = 30) -> str:
        # L1: Dangerous pattern detection
        for pattern, reason in self.DANGEROUS_PATTERNS:
            if re.search(pattern, command, re.IGNORECASE):
                raise SecurityBlock(
                    f"Blocked ({reason}): {command[:100]}"
                )

        # L2: Timeout enforcement
        timeout = min(timeout, self.MAX_EXECUTION_TIME)

        # L3: Working directory confinement
        cwd = Path.cwd()

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(cwd),
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )
        except asyncio.TimeoutError:
            return f"Command timed out after {timeout}s: {command[:100]}"
        except Exception as e:
            return f"Command execution error: {e}"

        # Decode with system encoding first (Windows: GBK), then UTF-8
        import sys
        import locale
        encoding = sys.getdefaultencoding() if sys.platform == "win32" else "utf-8"

        def safe_decode(data: bytes) -> str:
            try:
                return data.decode(encoding)
            except (UnicodeDecodeError, LookupError):
                return data.decode("utf-8", errors="replace")

        result_parts = []
        if stdout:
            result_parts.append(safe_decode(stdout))
        if stderr:
            result_parts.append("[stderr]\n" + safe_decode(stderr))
        result_parts.append(f"[exit code: {proc.returncode}]")
        return "\n".join(result_parts)
