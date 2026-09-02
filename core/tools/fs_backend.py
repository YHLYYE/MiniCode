"""Pluggable filesystem backend — borrowed from DeepAgents' design.

File tools (Read/Write) operate through this interface so the storage
medium can be swapped: local disk (default), Docker sandbox, remote
filesystem, etc. — without changing the tool's public behavior.
"""

from abc import ABC, abstractmethod
from pathlib import Path


class FilesystemBackend(ABC):
    """Pluggable filesystem backend for file tools."""

    @abstractmethod
    def read(self, path: str) -> str:
        """Read file content as text."""

    @abstractmethod
    def write(self, path: str, content: str) -> None:
        """Write text content to a file (creating parent dirs)."""

    @abstractmethod
    def exists(self, path: str) -> bool:
        """Return True if the path exists."""

    @abstractmethod
    def is_dir(self, path: str) -> bool:
        """Return True if the path is a directory."""


class LocalFilesystemBackend(FilesystemBackend):
    """Default backend — the local filesystem (zero dependencies)."""

    def read(self, path: str) -> str:
        return Path(path).read_text(encoding="utf-8", errors="replace")

    def write(self, path: str, content: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")

    def exists(self, path: str) -> bool:
        return Path(path).exists()

    def is_dir(self, path: str) -> bool:
        return Path(path).is_dir()
