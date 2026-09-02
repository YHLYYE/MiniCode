"""可插拔 Filesystem Backend 测试"""
import pytest
from core.tools.fs_backend import FilesystemBackend, LocalFilesystemBackend
from core.tools.files import ReadTool, WriteTool


class InMemoryBackend(FilesystemBackend):
    """模拟沙箱后端 — 用内存 dict 替代文件系统，验证可插拔性。"""

    def __init__(self):
        self._files: dict[str, str] = {}
        self._dirs: set[str] = set()

    def read(self, path: str) -> str:
        return self._files[path]

    def write(self, path: str, content: str) -> None:
        self._files[path] = content

    def exists(self, path: str) -> bool:
        return path in self._files

    def is_dir(self, path: str) -> bool:
        return path in self._dirs


@pytest.mark.asyncio
async def test_local_backend(tmp_path):
    """本地后端读写真实文件"""
    backend = LocalFilesystemBackend()
    f = tmp_path / "test.txt"
    backend.write(str(f), "hello\nworld")
    assert backend.read(str(f)) == "hello\nworld"
    assert backend.exists(str(f))
    assert not backend.is_dir(str(f))


@pytest.mark.asyncio
async def test_write_tool_with_in_memory_backend():
    """WriteTool 通过内存 backend 工作，不碰本地文件系统"""
    backend = InMemoryBackend()
    tool = WriteTool(backend=backend)

    result = await tool.execute("/sandbox/app.py", "print('hi')")

    assert "Created" in result
    assert backend.exists("/sandbox/app.py")
    assert backend.read("/sandbox/app.py") == "print('hi')"


@pytest.mark.asyncio
async def test_read_tool_with_in_memory_backend():
    """ReadTool 通过内存 backend 读取，返回带行号内容"""
    backend = InMemoryBackend()
    backend.write("/sandbox/app.py", "line1\nline2\nline3")

    tool = ReadTool(backend=backend)
    result = await tool.execute("/sandbox/app.py")

    assert "line1" in result
    assert "1\t" in result or "   1" in result  # line number prefix


@pytest.mark.asyncio
async def test_read_tool_missing_file():
    """读不存在的文件返回错误"""
    backend = InMemoryBackend()
    tool = ReadTool(backend=backend)
    result = await tool.execute("/sandbox/missing.py")
    assert "not found" in result


@pytest.mark.asyncio
async def test_tools_default_to_local_backend(tmp_path):
    """不传 backend 时默认本地文件系统"""
    tool = WriteTool()  # 默认 LocalFilesystemBackend
    f = tmp_path / "default.txt"
    result = await tool.execute(str(f), "content")
    assert "Created" in result
    assert f.exists()
