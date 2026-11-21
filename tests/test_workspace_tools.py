import sys
from pathlib import Path

import pytest

from core.tools.files import ReadTool, WriteTool
from core.tools.edit import EditTool
from core.tools.shell import BashTool
from capabilities.security import SecurityBlock


@pytest.mark.asyncio
async def test_workspace_write_maps_relative_path(tmp_path):
    tool = WriteTool(workspace_root=tmp_path)

    result = await tool.execute(file_path="src/a.txt", content="ok")

    assert result.startswith("Created")
    assert (tmp_path / "src/a.txt").read_text() == "ok"


@pytest.mark.asyncio
async def test_workspace_write_rejects_absolute_escape(tmp_path):
    tool = WriteTool(workspace_root=tmp_path)
    outside = tmp_path.parent / "escape.txt"

    result = await tool.execute(file_path=str(outside), content="bad")

    assert "outside workspace" in result
    assert not outside.exists()


@pytest.mark.asyncio
async def test_workspace_read_rejects_absolute_escape(tmp_path):
    outside = tmp_path.parent / "outside-read.txt"
    outside.write_text("secret")
    tool = ReadTool(workspace_root=tmp_path)

    result = await tool.execute(file_path=str(outside))

    assert "outside workspace" in result


@pytest.mark.asyncio
async def test_workspace_edit_rejects_absolute_escape(tmp_path):
    outside = tmp_path.parent / "outside-edit.txt"
    outside.write_text("old")
    tool = EditTool(workspace_root=tmp_path)

    result = await tool.execute(
        file_path=str(outside), old_string="old", new_string="new"
    )

    assert "outside workspace" in result
    assert outside.read_text() == "old"


@pytest.mark.asyncio
async def test_workspace_shell_uses_explicit_cwd(tmp_path):
    tool = BashTool(workspace_root=tmp_path)

    result = await tool.execute(
        command='python -c "import os; print(os.getcwd())"'
    )

    assert str(Path(tmp_path).resolve()).lower() in result.lower()


@pytest.mark.asyncio
async def test_workspace_shell_blocks_parent_directory_escape(tmp_path):
    tool = BashTool(workspace_root=tmp_path)

    with pytest.raises(SecurityBlock, match="workspace"):
        await tool.execute(command="cd .. && echo bad > escaped.txt")


@pytest.mark.asyncio
async def test_default_shell_uses_execution_cwd(tmp_path, monkeypatch):
    tool = BashTool()
    monkeypatch.chdir(tmp_path)

    result = await tool.execute(
        command=f'"{sys.executable}" -c "import os; print(os.getcwd())"'
    )

    assert str(tmp_path.resolve()).lower() in result.lower()
