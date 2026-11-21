"""Optional Docker-backed shell sandbox."""
from __future__ import annotations

import asyncio
import math
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path

from core.tools.base import Tool


@dataclass(frozen=True)
class DockerSandboxResult:
    command: str
    exit_code: int | None
    output: str
    timed_out: bool = False


class DockerSandbox:
    MAX_EXECUTION_TIME = 60
    MAX_OUTPUT_BYTES = 1_000_000

    def __init__(
        self,
        workspace_root: Path,
        *,
        image: str = "alpine:3.20",
        network: str = "none",
        memory: str = "512m",
        cpus: str = "1",
        pids_limit: int = 128,
        timeout_seconds: float = 60,
    ):
        self.workspace_root = Path(workspace_root).resolve()
        self.image = image
        self.network = network
        self.memory = memory
        self.cpus = cpus
        self.pids_limit = pids_limit
        self.timeout_seconds = timeout_seconds
        self.last_container_name = ""

    @classmethod
    def _bounded_timeout(cls, value: float) -> float:
        """Reject invalid values and keep caller-controlled timeouts bounded."""
        if isinstance(value, bool):
            raise ValueError("timeout must be a positive number")
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("timeout must be a positive number") from exc
        if not math.isfinite(numeric) or numeric <= 0:
            raise ValueError("timeout must be a positive number")
        return min(numeric, cls.MAX_EXECUTION_TIME)

    @staticmethod
    async def _stop_process(process: asyncio.subprocess.Process | None) -> None:
        if process is None or process.returncode is not None:
            return
        try:
            process.kill()
        except ProcessLookupError:
            pass
        try:
            await asyncio.wait_for(process.wait(), timeout=5)
        except asyncio.TimeoutError:
            pass

    @staticmethod
    async def _remove_container(name: str) -> None:
        try:
            await asyncio.to_thread(
                subprocess.run,
                ["docker", "rm", "-f", name],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5,
            )
        except (OSError, subprocess.SubprocessError):
            pass

    @classmethod
    async def _read_stream(
        cls, stream: asyncio.StreamReader
    ) -> tuple[bytes, bool]:
        chunks: list[bytes] = []
        total = 0
        while chunk := await stream.read(64 * 1024):
            remaining = cls.MAX_OUTPUT_BYTES - total
            if remaining > 0:
                chunks.append(chunk[:remaining])
            total += len(chunk)
            if total > cls.MAX_OUTPUT_BYTES:
                return b"".join(chunks), True
        return b"".join(chunks), False

    @classmethod
    async def _collect_output(
        cls, process: asyncio.subprocess.Process
    ) -> tuple[bytes, bytes, bool, bool]:
        """Drain both pipes concurrently and stop a runaway output producer."""
        assert process.stdout is not None
        assert process.stderr is not None
        stdout_task = asyncio.create_task(cls._read_stream(process.stdout))
        stderr_task = asyncio.create_task(cls._read_stream(process.stderr))
        tasks = {stdout_task, stderr_task}
        try:
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            if any(result.cancelled() is False and result.result()[1] for result in done):
                await cls._stop_process(process)
            stdout_result, stderr_result = await asyncio.gather(
                stdout_task, stderr_task
            )
            await process.wait()
            return (
                stdout_result[0],
                stderr_result[0],
                stdout_result[1],
                stderr_result[1],
            )
        finally:
            pending = [task for task in tasks if not task.done()]
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)

    async def run(self, command: str, timeout_seconds: float | None = None) -> DockerSandboxResult:
        name = f"minicode-sandbox-{uuid.uuid4().hex[:12]}"
        self.last_container_name = name
        timeout = self._bounded_timeout(
            self.timeout_seconds if timeout_seconds is None else timeout_seconds
        )
        args = [
            "docker", "run", "--rm", "--name", name,
            "--network", self.network,
            "--read-only",
            "--tmpfs", "/tmp:rw,size=64m",
            "--mount",
            f"type=bind,source={self.workspace_root.as_posix()},target=/workspace",
            "--workdir", "/workspace",
            "--memory", self.memory,
            "--cpus", self.cpus,
            "--pids-limit", str(self.pids_limit),
            "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges",
            self.image,
            "/bin/sh", "-lc", command,
        ]
        process = None
        try:
            process = await asyncio.create_subprocess_exec(
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr, stdout_truncated, stderr_truncated = await asyncio.wait_for(
                self._collect_output(process), timeout=timeout
            )
        except asyncio.TimeoutError:
            await self._stop_process(process)
            await self._remove_container(name)
            return DockerSandboxResult(
                command=command,
                exit_code=None,
                output=f"Sandbox command timed out after {timeout}s",
                timed_out=True,
            )
        except asyncio.CancelledError:
            await self._stop_process(process)
            await self._remove_container(name)
            raise
        except Exception as exc:
            await self._stop_process(process)
            await self._remove_container(name)
            return DockerSandboxResult(
                command=command,
                exit_code=None,
                output=f"{type(exc).__name__}: {exc}",
            )

        output = (stdout + stderr).decode("utf-8", errors="replace").strip()
        if stdout_truncated or stderr_truncated:
            output = (
                f"{output}\n[output truncated at {self.MAX_OUTPUT_BYTES} bytes "
                "per stream]"
            ).strip()
        return DockerSandboxResult(
            command=command,
            exit_code=process.returncode,
            output=output,
        )


class DockerBashTool(Tool):
    name = "Bash"
    description = (
        "Execute a shell command inside a constrained Docker container. "
        "The workspace is bind-mounted at /workspace; root is read-only and "
        "network defaults to none."
    )
    input_schema = {
        "command": {"type": "string", "description": "Shell command"},
        "timeout": {"type": "integer", "description": "Timeout seconds"},
    }
    is_readonly = False
    is_concurrency_safe = False
    is_destructive = True

    def __init__(
        self,
        workspace_root: Path,
        *,
        image: str = "alpine:3.20",
        network: str = "none",
        timeout_seconds: float = 60,
    ):
        super().__init__()
        self._sandbox = DockerSandbox(
            workspace_root,
            image=image,
            network=network,
            timeout_seconds=timeout_seconds,
        )

    async def execute(self, command: str, timeout: int = 30) -> str:
        result = await self._sandbox.run(command, timeout_seconds=timeout)
        parts = [result.output] if result.output else []
        parts.append(f"[exit code: {result.exit_code}]")
        return "\n".join(parts)
