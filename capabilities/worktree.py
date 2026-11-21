"""Safe Git Worktree lifecycle for isolated sub-agent execution."""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from core.tools.base import Tool


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
        self.index_path = self.worktrees_root / "index.json"

    def _load_index(self) -> dict:
        if not self.index_path.exists():
            return {"worktrees": []}
        try:
            data = json.loads(self.index_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {"worktrees": []}
        if not isinstance(data, dict) or not isinstance(data.get("worktrees"), list):
            return {"worktrees": []}
        return data

    def _save_index(self, data: dict) -> None:
        self.worktrees_root.mkdir(parents=True, exist_ok=True)
        temporary = self.index_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
            newline="\n",
        )
        os.replace(temporary, self.index_path)

    def find(self, branch_or_id: str) -> dict | None:
        for entry in self._load_index()["worktrees"]:
            if branch_or_id in {entry.get("branch"), entry.get("task_id")}:
                return entry
        return None

    def list_entries(self) -> list[dict]:
        return list(self._load_index()["worktrees"])

    def _update_entry(self, branch: str, **changes) -> dict:
        data = self._load_index()
        for entry in data["worktrees"]:
            if entry.get("branch") == branch:
                entry.update(changes)
                entry["updated_at"] = datetime.now(timezone.utc).isoformat()
                self._save_index(data)
                return entry
        raise RuntimeError(f"unknown worktree branch: {branch}")

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

    def _ensure_local_ignore(self) -> None:
        git_dir = Path(self._git("rev-parse", "--git-common-dir"))
        if not git_dir.is_absolute():
            git_dir = (self.repo_root / git_dir).resolve()
        exclude = git_dir / "info" / "exclude"
        exclude.parent.mkdir(parents=True, exist_ok=True)
        text = exclude.read_text(encoding="utf-8") if exclude.exists() else ""
        entries = {line.strip() for line in text.splitlines()}
        if ".minicode/" not in entries:
            suffix = "" if text.endswith("\n") or not text else "\n"
            exclude.write_text(
                text + suffix + ".minicode/\n",
                encoding="utf-8",
                newline="\n",
            )

    def create(self, task: str) -> WorktreeHandle:
        if not task.strip():
            raise ValueError("task name cannot be empty")

        self._ensure_ready()
        self._ensure_local_ignore()
        task_id = hashlib.sha256(task.encode("utf-8")).hexdigest()[:8]
        slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", task).strip("-").lower()
        slug = (slug or "task")[:32]
        branch = f"minicode/{slug}-{task_id}"
        path = self.worktrees_root / f"{slug}-{task_id}"
        base_commit = self._git("rev-parse", "HEAD")

        path.parent.mkdir(parents=True, exist_ok=True)
        self._git("worktree", "add", "-b", branch, str(path), base_commit)
        handle = WorktreeHandle(
            slug=slug,
            task_id=task_id,
            branch=branch,
            path=path.resolve(),
            base_commit=base_commit,
        )
        data = self._load_index()
        data["worktrees"].append(
            {
                "task_id": task_id,
                "branch": branch,
                "path": str(handle.path),
                "base_commit": base_commit,
                "status": "active",
                "verification": "pending",
                "verification_evidence": "",
                "commit": None,
                "changed_files": [],
                "created_at": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        self._save_index(data)
        return handle

    def commit(
        self,
        handle: WorktreeHandle,
        message: str,
        verification_status: str = "not_configured",
    ) -> str | None:
        self._git("add", "-A", cwd=handle.path)
        diff = subprocess.run(
            ["git", "diff", "--cached", "--quiet"],
            cwd=handle.path,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if diff.returncode == 0:
            self._update_entry(
                handle.branch,
                status="ready_unverified",
                verification=verification_status,
                commit=None,
                changed_files=[],
            )
            return None
        self._git("commit", "-m", message, cwd=handle.path)
        commit = self._git("rev-parse", "HEAD", cwd=handle.path)
        changed = self.changed_files(handle)
        self._update_entry(
            handle.branch,
            status="ready" if verification_status == "passed" else "ready_unverified",
            verification=verification_status,
            commit=commit,
            changed_files=changed,
        )
        return commit

    def mark_verification(self, branch: str, status: str, evidence: str = "") -> dict:
        if status not in {"passed", "failed", "not_configured"}:
            raise ValueError(f"invalid verification status: {status}")
        current = self.find(branch)
        if current is None:
            raise RuntimeError(f"unknown worktree branch: {branch}")
        if status == "failed":
            entry_status = "verify_failed"
        elif current.get("status") == "ready":
            entry_status = "ready"
        else:
            entry_status = "active"
        return self._update_entry(
            branch,
            status=entry_status,
            verification=status,
            verification_evidence=evidence[-8000:],
        )

    def mark_failed(self, branch: str, reason: str) -> dict:
        return self._update_entry(
            branch,
            status="failed",
            verification_evidence=reason[-8000:],
        )

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

    def apply(self, branch_or_id: str) -> str:
        entry = self.find(branch_or_id)
        if entry is None:
            raise RuntimeError(f"unknown worktree: {branch_or_id}")
        if entry.get("status") != "ready" or entry.get("verification") != "passed":
            raise RuntimeError("worktree must be verified and ready before apply")
        if self._git("status", "--porcelain"):
            raise RuntimeError("main working tree must be clean before apply")
        commit = entry.get("commit")
        if not commit:
            raise RuntimeError("worktree has no commit to apply")
        ancestor = subprocess.run(
            ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
            cwd=self.repo_root,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if ancestor.returncode == 0:
            self._update_entry(entry["branch"], status="applied")
            return self._git("rev-parse", "HEAD")
        try:
            self._git("cherry-pick", commit)
        except RuntimeError:
            subprocess.run(
                ["git", "cherry-pick", "--abort"],
                cwd=self.repo_root,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            raise
        applied = self._git("rev-parse", "HEAD")
        self._update_entry(entry["branch"], status="applied", commit=applied)
        return applied

    def remove(self, branch_or_id: str) -> None:
        entry = self.find(branch_or_id)
        if entry is None:
            raise RuntimeError(f"unknown worktree: {branch_or_id}")
        if entry.get("status") == "active":
            raise RuntimeError("active worktree cannot be removed")
        path = Path(entry["path"])
        if path.exists():
            self._git("worktree", "remove", "--force", str(path))
        self._update_entry(entry["branch"], status="removed")


class WorktreeListTool(Tool):
    name = "WorktreeList"
    description = "List isolated task worktrees and their lifecycle state."
    input_schema = {}
    is_readonly = True
    is_concurrency_safe = True

    def __init__(self, manager: WorktreeManager):
        super().__init__()
        self._manager = manager

    async def execute(self) -> str:
        entries = self._manager.list_entries()
        if not entries:
            return "No isolated worktrees."
        lines = []
        for entry in entries:
            lines.append(
                f"{entry['branch']} | status={entry['status']} | "
                f"verification={entry['verification']} | "
                f"commit={entry.get('commit') or 'none'} | "
                f"files={entry.get('changed_files', [])} | path={entry['path']}"
            )
        return "\n".join(lines)


class WorktreeApplyTool(Tool):
    name = "WorktreeApply"
    description = "Apply a verified ready worktree commit after user approval."
    input_schema = {"branch": {"type": "string"}}
    is_readonly = False
    is_destructive = True
    is_concurrency_safe = False

    def __init__(self, manager: WorktreeManager):
        super().__init__()
        self._manager = manager

    async def execute(self, branch: str) -> str:
        try:
            commit = self._manager.apply(branch)
            return f"Applied {branch} as {commit}."
        except Exception as exc:
            return f"Error applying {branch}: {exc}"


class WorktreeRemoveTool(Tool):
    name = "WorktreeRemove"
    description = "Remove a completed or failed worktree while retaining its branch."
    input_schema = {"branch": {"type": "string"}}
    is_readonly = False
    is_destructive = True
    is_concurrency_safe = False

    def __init__(self, manager: WorktreeManager):
        super().__init__()
        self._manager = manager

    async def execute(self, branch: str) -> str:
        try:
            self._manager.remove(branch)
            return f"Removed worktree {branch}; branch retained."
        except Exception as exc:
            return f"Error removing {branch}: {exc}"
