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


requires_docker = pytest.mark.skipif(
    not docker_available(), reason="Docker daemon is not available"
)


def test_docker_timeout_is_bounded_and_rejects_invalid_values(tmp_path):
    sandbox = DockerSandbox(tmp_path)

    assert sandbox._bounded_timeout(3600) == DockerSandbox.MAX_EXECUTION_TIME
    with pytest.raises(ValueError, match="positive number"):
        sandbox._bounded_timeout(0)
    with pytest.raises(ValueError, match="positive number"):
        sandbox._bounded_timeout(float("inf"))


@pytest.mark.asyncio
async def test_docker_unexpected_error_removes_container(monkeypatch, tmp_path):
    class FakeProcess:
        returncode = None

        def kill(self):
            self.returncode = -9

        async def wait(self):
            return self.returncode

    async def fake_exec(*args, **kwargs):
        return FakeProcess()

    async def fail_collect(self, process):
        raise RuntimeError("pipe failure")

    removed = []

    async def fake_remove(name):
        removed.append(name)

    monkeypatch.setattr("asyncio.create_subprocess_exec", fake_exec)
    monkeypatch.setattr(DockerSandbox, "_collect_output", fail_collect)
    monkeypatch.setattr(
        DockerSandbox, "_remove_container", staticmethod(fake_remove)
    )

    sandbox = DockerSandbox(tmp_path)
    result = await sandbox.run("echo should-not-run")

    assert "RuntimeError: pipe failure" in result.output
    assert removed == [sandbox.last_container_name]


@requires_docker
@pytest.mark.asyncio
async def test_docker_sandbox_mounts_only_workspace(tmp_path):
    sandbox = DockerSandbox(tmp_path, image="alpine:3.20", timeout_seconds=10)

    result = await sandbox.run("printf sandboxed > inside.txt; pwd")

    assert result.exit_code == 0
    assert "/workspace" in result.output
    assert (tmp_path / "inside.txt").read_text() == "sandboxed"


@requires_docker
@pytest.mark.asyncio
async def test_docker_sandbox_root_is_read_only(tmp_path):
    sandbox = DockerSandbox(tmp_path, image="alpine:3.20", timeout_seconds=10)

    result = await sandbox.run("printf bad > /outside.txt")

    assert result.exit_code != 0
    assert not (tmp_path.parent / "outside.txt").exists()


@requires_docker
@pytest.mark.asyncio
async def test_docker_sandbox_disables_network(tmp_path):
    sandbox = DockerSandbox(tmp_path, image="alpine:3.20", timeout_seconds=10)

    result = await sandbox.run("wget -T 1 -qO- https://example.com")

    assert result.exit_code != 0


@requires_docker
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


@requires_docker
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


@requires_docker
@pytest.mark.asyncio
async def test_docker_sandbox_truncates_runaway_output(tmp_path):
    sandbox = DockerSandbox(tmp_path, image="alpine:3.20", timeout_seconds=10)

    result = await sandbox.run(
        f"head -c {DockerSandbox.MAX_OUTPUT_BYTES + 1024} /dev/zero | tr '\\0' x"
    )

    assert "output truncated" in result.output
    assert len(result.output.encode("utf-8")) < DockerSandbox.MAX_OUTPUT_BYTES + 1024
