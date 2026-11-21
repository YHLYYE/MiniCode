import shutil
import subprocess

import pytest

from capabilities.docker_sandbox import DockerBashTool, DockerSandbox


def docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    return subprocess.run(
        ["docker", "info"], stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode == 0


pytestmark = pytest.mark.skipif(
    not docker_available(), reason="Docker daemon is not available"
)


@pytest.mark.asyncio
async def test_docker_sandbox_mounts_only_workspace(tmp_path):
    sandbox = DockerSandbox(tmp_path, image="alpine:3.20", timeout_seconds=10)

    result = await sandbox.run("printf sandboxed > inside.txt; pwd")

    assert result.exit_code == 0
    assert "/workspace" in result.output
    assert (tmp_path / "inside.txt").read_text() == "sandboxed"


@pytest.mark.asyncio
async def test_docker_sandbox_root_is_read_only(tmp_path):
    sandbox = DockerSandbox(tmp_path, image="alpine:3.20", timeout_seconds=10)

    result = await sandbox.run("printf bad > /outside.txt")

    assert result.exit_code != 0
    assert not (tmp_path.parent / "outside.txt").exists()


@pytest.mark.asyncio
async def test_docker_sandbox_disables_network(tmp_path):
    sandbox = DockerSandbox(tmp_path, image="alpine:3.20", timeout_seconds=10)

    result = await sandbox.run("wget -T 1 -qO- https://example.com")

    assert result.exit_code != 0


@pytest.mark.asyncio
async def test_docker_sandbox_timeout_removes_container(tmp_path):
    sandbox = DockerSandbox(tmp_path, image="alpine:3.20", timeout_seconds=0.3)

    result = await sandbox.run("sleep 5")

    assert result.timed_out is True
    names = subprocess.check_output(
        ["docker", "ps", "-a", "--filter", f"name={sandbox.last_container_name}",
         "--format", "{{.Names}}"],
        text=True,
    ).strip()
    assert names == ""


@pytest.mark.asyncio
async def test_docker_bash_tool_uses_workspace(tmp_path):
    tool = DockerBashTool(
        workspace_root=tmp_path,
        image="alpine:3.20",
        timeout_seconds=10,
    )

    result = await tool.execute(command="printf tool > tool.txt; pwd")

    assert "[exit code: 0]" in result
    assert "/workspace" in result
    assert (tmp_path / "tool.txt").read_text() == "tool"
