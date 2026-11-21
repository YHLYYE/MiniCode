"""External completion verification for coding tasks."""
from __future__ import annotations

import asyncio
import locale
import os
import signal
import subprocess
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
            popen_kwargs = (
                {"creationflags": getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)}
                if os.name == "nt"
                else {"start_new_session": True}
            )
            process = await asyncio.create_subprocess_shell(
                self.command,
                cwd=str(self.cwd),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                **popen_kwargs,
            )
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=self.timeout_seconds
            )
        except asyncio.TimeoutError:
            if process is not None:
                await _terminate_process_tree(process)
            return VerificationResult(
                command=self.command,
                passed=False,
                exit_code=None,
                output=f"verification timed out after {self.timeout_seconds}s",
                timed_out=True,
            )
        except Exception as exc:
            if process is not None and process.returncode is None:
                await _terminate_process_tree(process)
            return VerificationResult(
                command=self.command,
                passed=False,
                exit_code=None,
                output=f"{type(exc).__name__}: {exc}",
            )

        combined = b"\n".join(part for part in (stdout, stderr) if part)
        decoded = _decode_output(combined)
        return VerificationResult(
            command=self.command,
            passed=process.returncode == 0,
            exit_code=process.returncode,
            output=decoded[-self.max_output_chars :],
        )


async def _terminate_process_tree(process) -> None:
    """Terminate the shell and its descendants on Windows and POSIX."""
    if process.returncode is not None:
        return
    if os.name == "nt":
        # A console interrupt reaches descendants that inherited the new
        # process group, while taskkill /T is the authoritative fallback.
        try:
            process.send_signal(signal.CTRL_BREAK_EVENT)
        except (OSError, ValueError):
            pass
        for _ in range(2):
            completed = await asyncio.to_thread(
                subprocess.run,
                ["taskkill", "/T", "/F", "/PID", str(process.pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if completed.returncode == 0:
                break
            await asyncio.sleep(0.05)
    else:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    try:
        await asyncio.wait_for(process.wait(), timeout=1)
    except asyncio.TimeoutError:
        process.kill()
        await process.wait()


def _decode_output(data: bytes) -> str:
    encoding = locale.getpreferredencoding(False)
    try:
        return data.decode(encoding)
    except (UnicodeDecodeError, LookupError):
        return data.decode("utf-8", errors="replace")
