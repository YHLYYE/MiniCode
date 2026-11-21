"""Persistent task graph primitives for MiniCode."""
from __future__ import annotations

import json
import os
import secrets
import threading
from datetime import datetime, timezone
from pathlib import Path


class TaskGraphError(RuntimeError):
    """An invalid operation or unreadable task graph."""


class TaskGraphStore:
    """Project-scoped JSON task graph with atomic writes."""

    def __init__(self, project_root: Path):
        self.project_root = Path(project_root).resolve()
        self.path = self.project_root / ".minicode" / "tasks.json"
        self._lock = threading.RLock()

    def tasks(self) -> list[dict]:
        with self._lock:
            return list(self._load_unlocked()["tasks"])

    def get(self, task_id: str) -> dict:
        with self._lock:
            for task in self._load_unlocked()["tasks"]:
                if task["id"] == task_id:
                    return dict(task)
        raise TaskGraphError(f"unknown task: {task_id}")

    def create(
        self,
        subject: str,
        description: str = "",
        blocked_by: list[str] | None = None,
    ) -> dict:
        with self._lock:
            data = self._load_unlocked()
            existing = {task["id"] for task in data["tasks"]}
            dependencies = list(dict.fromkeys(blocked_by or []))
            unknown = [task_id for task_id in dependencies if task_id not in existing]
            if unknown:
                raise TaskGraphError(
                    f"unknown dependency: {', '.join(unknown)}"
                )
            task_id = self._new_id(existing)
            now = _utc_now()
            task = {
                "id": task_id,
                "subject": _validated_text(subject, 200, "subject"),
                "description": _validated_text(
                    description, 4000, "description", allow_empty=True
                ),
                "status": "pending",
                "blocked_by": dependencies,
                "result": "",
                "created_at": now,
                "started_at": None,
                "completed_at": None,
            }
            data["tasks"].append(task)
            self._save_unlocked(data)
            return dict(task)

    def update(self, task_id: str, status: str, result: str = "") -> dict:
        if status not in {"in_progress", "completed"}:
            raise TaskGraphError(f"invalid target status: {status}")
        with self._lock:
            data = self._load_unlocked()
            task = _find_task(data, task_id)
            current = task["status"]
            if status == current:
                if result:
                    task["result"] = _validated_text(result, 8000, "result")
                    self._save_unlocked(data)
                return dict(task)
            if status == "in_progress" and current == "pending":
                blockers = _unfinished_blockers(data, task)
                if blockers:
                    raise TaskGraphError(
                        f"task is blocked by: {', '.join(blockers)}"
                    )
                task["status"] = status
                task["started_at"] = _utc_now()
            elif status == "completed" and current == "in_progress":
                task["status"] = status
                task["completed_at"] = _utc_now()
                if result:
                    task["result"] = _validated_text(result, 8000, "result")
            else:
                raise TaskGraphError(
                    f"invalid transition: {current} -> {status}"
                )
            self._save_unlocked(data)
            return dict(task)

    def _load_unlocked(self) -> dict:
        if not self.path.exists():
            return {"version": 1, "tasks": []}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise TaskGraphError(f"task graph is corrupt: {exc}") from exc
        _validate_graph(data)
        return data

    def _save_unlocked(self, data: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(f".json.{secrets.token_hex(4)}.tmp")
        try:
            temporary.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding="utf-8",
                newline="\n",
            )
            os.replace(temporary, self.path)
        finally:
            if temporary.exists():
                temporary.unlink()

    @staticmethod
    def _new_id(existing: set[str]) -> str:
        while True:
            task_id = f"task_{secrets.token_hex(4)}"
            if task_id not in existing:
                return task_id


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _validated_text(
    value: object,
    limit: int,
    field: str,
    *,
    allow_empty: bool = False,
) -> str:
    if not isinstance(value, str):
        raise TaskGraphError(f"{field} must be a string")
    text = value.strip()
    if not text and not allow_empty:
        raise TaskGraphError(f"{field} cannot be empty")
    if len(text) > limit:
        raise TaskGraphError(f"{field} exceeds {limit} characters")
    return text


def _validate_graph(data: object) -> None:
    if not isinstance(data, dict) or data.get("version") != 1:
        raise TaskGraphError("task graph is corrupt: unsupported structure or version")
    tasks = data.get("tasks")
    if not isinstance(tasks, list):
        raise TaskGraphError("task graph is corrupt: tasks must be a list")
    seen: set[str] = set()
    for task in tasks:
        if not isinstance(task, dict):
            raise TaskGraphError("task graph is corrupt: task must be an object")
        task_id = task.get("id")
        if not isinstance(task_id, str) or not task_id.startswith("task_"):
            raise TaskGraphError("task graph is corrupt: invalid task id")
        if task_id in seen:
            raise TaskGraphError("task graph is corrupt: duplicate task id")
        seen.add(task_id)


def _find_task(data: dict, task_id: str) -> dict:
    for task in data["tasks"]:
        if task.get("id") == task_id:
            return task
    raise TaskGraphError(f"unknown task: {task_id}")


def _unfinished_blockers(data: dict, task: dict) -> list[str]:
    by_id = {entry["id"]: entry for entry in data["tasks"]}
    return [
        task_id
        for task_id in task.get("blocked_by", [])
        if by_id.get(task_id, {}).get("status") != "completed"
    ]

