"""Shell command execution tool — 3-layer security defense"""
import asyncio
import re
from pathlib import Path
from core.tools.base import Tool
from capabilities.security import SecurityBlock, DANGEROUS_COMMAND_PATTERNS


class BashTool(Tool):
    """Execute shell commands with safety checks.

    3-layer security:
    1. Dangerous pattern regex detection (DANGEROUS_COMMAND_PATTERNS — shared
       with the permission pipeline's L1 rule filter)
    2. Timeout enforcement (max 60s)
    3. Working directory: the subprocess starts in the process's cwd.
       Absolute / `..` paths inside the command are NOT confined — command-level
       confinement is what the L1 pattern list is for.
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

    DANGEROUS_PATTERNS = DANGEROUS_COMMAND_PATTERNS
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

        proc = None
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
            # 取消 communicate 不会终止子进程 → 必须显式 kill，否则泄漏
            if proc is not None:
                proc.kill()
                await proc.wait()
            return f"Command timed out after {timeout}s: {command[:100]}"
        except Exception as e:
            if proc is not None:
                proc.kill()
                await proc.wait()
            return f"Command execution error: {e}"

        # Decode with system locale encoding (Windows: GBK/cp936), then UTF-8
        import sys
        import locale
        encoding = locale.getpreferredencoding(False) if sys.platform == "win32" else "utf-8"

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
