"""Edit 工具测试 — 精确替换"""
import pytest

from core.tools.edit import EditTool


@pytest.mark.asyncio
async def test_edit_replaces_unique_string(tmp_path):
    f = tmp_path / "a.py"
    f.write_text("def foo():\n    return 1\n", encoding="utf-8")
    result = await EditTool().execute(str(f), "return 1", "return 42")
    assert "Edited" in result
    assert f.read_text(encoding="utf-8") == "def foo():\n    return 42\n"


@pytest.mark.asyncio
async def test_edit_not_found(tmp_path):
    f = tmp_path / "a.py"
    f.write_text("hello\n", encoding="utf-8")
    result = await EditTool().execute(str(f), "nonexistent", "x")
    assert "not found" in result


@pytest.mark.asyncio
async def test_edit_not_unique(tmp_path):
    f = tmp_path / "a.py"
    f.write_text("foo\nfoo\n", encoding="utf-8")
    result = await EditTool().execute(str(f), "foo", "bar")
    assert "2 times" in result


@pytest.mark.asyncio
async def test_edit_empty_old_string(tmp_path):
    f = tmp_path / "a.py"
    f.write_text("hello\n", encoding="utf-8")
    result = await EditTool().execute(str(f), "", "x")
    assert "must not be empty" in result


@pytest.mark.asyncio
async def test_edit_missing_file(tmp_path):
    result = await EditTool().execute(str(tmp_path / "nope.py"), "a", "b")
    assert "File not found" in result
