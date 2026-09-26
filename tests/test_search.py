"""代码搜索工具测试 — Grep / Glob"""
import pytest

from core.tools.search import GrepTool, GlobTool


@pytest.mark.asyncio
async def test_grep_finds_matches(tmp_path):
    (tmp_path / "a.py").write_text("def foo():\n    return 42\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("def bar():\n    return 43\n", encoding="utf-8")

    result = await GrepTool().execute("def foo", path=str(tmp_path))
    assert "a.py:1: def foo()" in result
    assert "b.py" not in result


@pytest.mark.asyncio
async def test_grep_no_match(tmp_path):
    (tmp_path / "a.py").write_text("hello\n", encoding="utf-8")
    result = await GrepTool().execute("nonexistent_xyz", path=str(tmp_path))
    assert "No matches" in result


@pytest.mark.asyncio
async def test_grep_invalid_regex(tmp_path):
    result = await GrepTool().execute("(", path=str(tmp_path))
    assert "invalid regex" in result


@pytest.mark.asyncio
async def test_grep_skips_binary_and_skipped_dirs(tmp_path):
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "c.pyc").write_bytes(b"\x00\x01\x02")
    (tmp_path / "real.py").write_text("needle here\n", encoding="utf-8")

    result = await GrepTool().execute("needle", path=str(tmp_path))
    assert "real.py:1: needle here" in result
    assert "pyc" not in result


@pytest.mark.asyncio
async def test_glob_finds_files(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "src" / "util.py").write_text("y = 2\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("docs\n", encoding="utf-8")

    result = await GlobTool().execute("**/*.py", path=str(tmp_path))
    assert "main.py" in result
    assert "util.py" in result
    assert "README.md" not in result


@pytest.mark.asyncio
async def test_glob_no_match(tmp_path):
    result = await GlobTool().execute("*.xyz", path=str(tmp_path))
    assert "No files match" in result
