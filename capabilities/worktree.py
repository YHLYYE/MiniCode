"""Safe Git Worktree lifecycle for isolated sub-agent execution."""
from __future__ import annotations

import hashlib
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class WorktreeHandle:
    slug: str
    task_id: str
    branch: str
    path: Path
    base_commit: str


class WorktreeManager:
    def __init__(self, repo_root: Path):
        self.repo_root = repo_root.resolve()
        self.worktrees_root = self.repo_root / ".minicode" / "worktrees"

    def _git(self, *args: str, cwd: Path | None = None) -> str:
        try:
            return subprocess.check_output(
                ["git", *args],
                cwd=cwd or self.repo_root,
                text=True,
                errors="replace",
                stderr=subprocess.STDOUT,
            ).strip()
        except subprocess.CalledProcessError as exc:
            output = (exc.stdout or "").strip()
            raise RuntimeError(output or f"git {' '.join(args)} failed") from exc

    def _ensure_ready(self) -> None:
        self._git("rev-parse", "--show-toplevel")
        if self._git("status", "--porcelain"):
            raise RuntimeError(
                "working tree must be clean before creating a worktree"
            )

    def create(self, task: str) -> WorktreeHandle:
        if not task.strip():
            raise ValueError("task name cannot be empty")

        self._ensure_ready()
        task_id = hashlib.sha256(task.encode("utf-8")).hexdigest()[:8]
        slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", task).strip("-").lower()
        slug = (slug or "task")[:32]
        branch = f"minicode/{slug}-{task_id}"
        path = self.worktrees_root / f"{slug}-{task_id}"
        base_commit = self._git("rev-parse", "HEAD")

        path.parent.mkdir(parents=True, exist_ok=True)
        self._git("worktree", "add", "-b", branch, str(path), base_commit)
        return WorktreeHandle(
            slug=slug,
            task_id=task_id,
            branch=branch,
            path=path.resolve(),
            base_commit=base_commit,
        )

    def commit(self, handle: WorktreeHandle, message: str) -> str | None:
        self._git("add", "-A", cwd=handle.path)
        diff = subprocess.run(
            ["git", "diff", "--cached", "--quiet"],
            cwd=handle.path,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if diff.returncode == 0:
            return None
        self._git("commit", "-m", message, cwd=handle.path)
        return self._git("rev-parse", "HEAD", cwd=handle.path)

    def diff_stat(self, handle: WorktreeHandle) -> str:
        return self._git(
            "diff", "--stat", f"{handle.base_commit}..HEAD", cwd=handle.path
        )

    def changed_files(self, handle: WorktreeHandle) -> list[str]:
        raw = self._git(
            "diff", "--name-only", f"{handle.base_commit}..HEAD", cwd=handle.path
        )
        return [line for line in raw.splitlines() if line.strip()]

    def working_tree_status(self, handle: WorktreeHandle) -> str:
        return self._git("status", "--short", cwd=handle.path)

    def working_tree_changed_files(self, handle: WorktreeHandle) -> list[str]:
        raw = self.working_tree_status(handle)
        files = []
        for line in raw.splitlines():
            path = line[3:].strip()
            if " -> " in path:
                path = path.split(" -> ", 1)[1]
            if path:
                files.append(path)
        return files

    def remove(self, handle: WorktreeHandle) -> None:
        self._git("worktree", "remove", "--force", str(handle.path))
