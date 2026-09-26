"""Code search tools — Grep (content) and Glob (filename).

These fill a core coding-agent gap: without them the model can only Read
whole files and has no way to search the codebase.
"""
import glob
import re
from pathlib import Path

from core.tools.base import Tool

# 搜索时跳过的目录和二进制/无关文件
_SKIP_DIRS = {
    ".git", ".venv", "venv", "node_modules", "__pycache__",
    ".minicode", ".idea", ".vscode", ".pytest_cache",
}
_SKIP_SUFFIXES = {
    ".png", ".jpg", ".jpeg", ".gif", ".ico", ".pyc", ".so", ".dll",
    ".exe", ".zip", ".tar", ".gz", ".pdf", ".db", ".sqlite",
}
_MAX_RESULTS = 200


class GrepTool(Tool):
    """Search file contents for a regex pattern (code search)."""

    name = "Grep"
    description = (
        "Search file contents for a regex pattern and return matching lines "
        "as 'file:line: content'. Use to find symbols, strings, or code "
        "patterns across the project."
    )
    input_schema = {
        "pattern": {
            "type": "string",
            "description": "Regex pattern to search for",
        },
        "path": {
            "type": "string",
            "description": "Directory to search (default: project root)",
        },
    }
    is_readonly = True
    is_concurrency_safe = True

    async def execute(self, pattern: str, path: str = ".") -> str:
        root = Path(path)
        if not root.exists():
            return f"Error: path not found: {path}"
        try:
            regex = re.compile(pattern)
        except re.error as e:
            return f"Error: invalid regex: {e}"

        matches: list[str] = []
        for p in root.rglob("*"):
            if not p.is_file():
                continue
            if any(part in _SKIP_DIRS for part in p.parts):
                continue
            if p.suffix.lower() in _SKIP_SUFFIXES:
                continue
            try:
                text = p.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for i, line in enumerate(text.split("\n"), 1):
                if regex.search(line):
                    matches.append(f"{p}:{i}: {line.strip()}")
                    if len(matches) >= _MAX_RESULTS:
                        break
            if len(matches) >= _MAX_RESULTS:
                break

        if not matches:
            return f"No matches for: {pattern}"
        return "\n".join(matches)


class GlobTool(Tool):
    """Find files matching a glob pattern."""

    name = "Glob"
    description = (
        "Find files matching a glob pattern (e.g. '**/*.py', 'src/*.ts'). "
        "Returns matching file paths."
    )
    input_schema = {
        "pattern": {
            "type": "string",
            "description": "Glob pattern (supports ** for recursive)",
        },
        "path": {
            "type": "string",
            "description": "Directory to search (default: project root)",
        },
    }
    is_readonly = True
    is_concurrency_safe = True

    async def execute(self, pattern: str, path: str = ".") -> str:
        root = Path(path)
        if not root.exists():
            return f"Error: path not found: {path}"

        matches = []
        for rel in glob.glob(pattern, root_dir=str(root), recursive=True):
            full = root / rel
            if full.is_file() and not any(
                part in _SKIP_DIRS for part in full.parts
            ):
                matches.append(str(full))

        matches.sort()
        if not matches:
            return f"No files match: {pattern}"
        return "\n".join(matches[:_MAX_RESULTS])
