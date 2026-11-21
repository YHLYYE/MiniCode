"""External completion verification for coding tasks."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class VerificationResult:
    command: str
    passed: bool
    exit_code: int | None
    output: str
    timed_out: bool = False


class CommandVerifier:
    """Run a user-supplied command and convert its result into evidence."""

    def __init__(
        self,
        command: str,
        *,
        cwd: Path | None = None,
        timeout_seconds: float = 120,
        max_output_chars: int = 8_000,
    ):
        if not command.strip():
            raise ValueError("verification command cannot be empty")
        if timeout_seconds <= 0:
            raise ValueError("verification timeout must be positive")
        if max_output_chars < 1:
            raise ValueError("max_output_chars must be positive")
        self.command = command
        self.cwd = cwd or Path.cwd()
        self.timeout_seconds = timeout_seconds
        self.max_output_chars = max_output_chars

    async def verify(self) -> VerificationResult:
        process = None
        try:
            process = await asyncio.create_subprocess_shell(
                self.command,
                cwd=str(self.cwd),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=self.timeout_seconds
            )
        except asyncio.TimeoutError:
            if process is not None:
                process.kill()
                await process.wait()
            return VerificationResult(
                command=self.command,
                passed=False,
                exit_code=None,
                output=f"verification timed out after {self.timeout_seconds}s",
                timed_out=True,
            )
        except Exception as exc:
            if process is not None and process.returncode is None:
                process.kill()
                await process.wait()
            return VerificationResult(
                command=self.command,
                passed=False,
                exit_code=None,
                output=f"{type(exc).__name__}: {exc}",
            )

        combined = b"\n".join(part for part in (stdout, stderr) if part)
        decoded = combined.decode("utf-8", errors="replace")
        return VerificationResult(
            command=self.command,
            passed=process.returncode == 0,
            exit_code=process.returncode,
            output=decoded[-self.max_output_chars :],
        )
