"""Optional Docker-backed shell sandbox."""
from __future__ import annotations

import asyncio
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
    def __init__(
        self,
        workspace_root: Path,
        *,
        image: str = "alpine:3.20",
        network: str = "none",
        memory: str = "512m",
        cpus: str = "1",
        pids_limit: int = 128,
        timeout_seconds: float = 120,
    ):
        self.workspace_root = Path(workspace_root).resolve()
        self.image = image
        self.network = network
        self.memory = memory
        self.cpus = cpus
        self.pids_limit = pids_limit
        self.timeout_seconds = timeout_seconds
        self.last_container_name = ""

    async def run(self, command: str, timeout_seconds: float | None = None) -> DockerSandboxResult:
        name = f"minicode-sandbox-{uuid.uuid4().hex[:12]}"
        self.last_container_name = name
        timeout = timeout_seconds or self.timeout_seconds
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
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=timeout
            )
        except asyncio.TimeoutError:
            if process is not None:
                process.kill()
                await process.wait()
            await asyncio.to_thread(
                subprocess.run,
                ["docker", "rm", "-f", name],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return DockerSandboxResult(
                command=command,
                exit_code=None,
                output=f"Sandbox command timed out after {timeout}s",
                timed_out=True,
            )
        except Exception as exc:
            if process is not None and process.returncode is None:
                process.kill()
                await process.wait()
            return DockerSandboxResult(
                command=command,
                exit_code=None,
                output=f"{type(exc).__name__}: {exc}",
            )

        output = (stdout + stderr).decode("utf-8", errors="replace").strip()
        return DockerSandboxResult(
            command=command,
            exit_code=process.returncode,
            output=output,
        )


class DockerBashTool(Tool):
    name = "Bash"
    description = (
        "Execute a shell command inside a constrained Docker container. "
        "Only the workspace is mounted writable; network defaults to none."
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
        timeout_seconds: float = 120,
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
