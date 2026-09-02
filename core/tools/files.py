"""File read/write tools — operate through a pluggable FilesystemBackend."""
from pathlib import Path

from core.tools.base import Tool
from core.tools.fs_backend import FilesystemBackend, LocalFilesystemBackend


class ReadTool(Tool):
    """Read a file and return its content with line numbers"""

    name = "Read"
    description = (
        "Read a file from the filesystem. "
        "Returns content with line numbers (format: 'LINE_NUM\\tCONTENT')."
    )
    input_schema = {
        "file_path": {
            "type": "string",
            "description": "Absolute path to the file to read",
        },
    }
    is_readonly = True
    is_concurrency_safe = True

    def __init__(self, backend: FilesystemBackend | None = None):
        super().__init__()
        self._backend = backend or LocalFilesystemBackend()

    async def execute(self, file_path: str) -> str:
        if not self._backend.exists(file_path):
            return f"Error: File not found: {file_path}"
        if self._backend.is_dir(file_path):
            return f"Error: Path is a directory: {file_path}"
        try:
            content = self._backend.read(file_path)
        except Exception as e:
            return f"Error reading {file_path}: {e}"

        if not content:
            return f"(empty file: {file_path})"

        lines = content.split("\n")
        return "\n".join(f"{i+1:4}\t{line}" for i, line in enumerate(lines))


class WriteTool(Tool):
    """Write content to a file (creates or overwrites)"""

    name = "Write"
    description = (
        "Write a file to the filesystem. "
        "Creates parent directories if needed. Overwrites existing files."
    )
    input_schema = {
        "file_path": {
            "type": "string",
            "description": "Absolute path to the file to write",
        },
        "content": {
            "type": "string",
            "description": "Content to write to the file",
        },
    }
    is_readonly = False
    is_concurrency_safe = False
    is_destructive = True

    def __init__(self, backend: FilesystemBackend | None = None):
        super().__init__()
        self._backend = backend or LocalFilesystemBackend()

    async def execute(self, file_path: str, content: str) -> str:
        existed = self._backend.exists(file_path)
        try:
            self._backend.write(file_path, content)
        except Exception as e:
            return f"Error writing {file_path}: {e}"

        action = "Updated" if existed else "Created"
        return f"{action} {file_path} ({len(content)} chars)"
