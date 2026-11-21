import sys
from pathlib import Path

import pytest

from core.tools.files import WriteTool
from core.tools.shell import BashTool


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
async def test_workspace_shell_uses_explicit_cwd(tmp_path):
    tool = BashTool(workspace_root=tmp_path)

    result = await tool.execute(
        command=f'"{sys.executable}" -c "import os; print(os.getcwd())"'
    )

    assert str(Path(tmp_path).resolve()).lower() in result.lower()
