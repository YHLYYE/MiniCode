"""Persistent task graph primitives for MiniCode."""
from __future__ import annotations

import json
import os
import re
import secrets
import threading
from datetime import datetime, timezone
from pathlib import Path

from core.tools.base import Tool


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

    def render_snapshot(self, include_completed: bool = False) -> str:
        tasks = self.tasks()
        by_id = {task["id"]: task for task in tasks}
        visible = [
            task for task in tasks
            if include_completed or task["status"] != "completed"
        ]
        ready = sum(_readiness(task, by_id) == "ready" for task in tasks)
        blocked = sum(_readiness(task, by_id) == "blocked" for task in tasks)
        in_progress = sum(task["status"] == "in_progress" for task in tasks)
        completed = sum(task["status"] == "completed" for task in tasks)

        lines = ["## Task Graph"]
        if not visible:
            lines.append("- none")
        for task in visible:
            state = task["status"]
            if state == "pending":
                state = _readiness(task, by_id)
            line = f"- {task['id']} [{state}] {task['subject']}"
            if task.get("blocked_by"):
                line += f" (blocked_by: {', '.join(task['blocked_by'])})"
            lines.append(line)
        lines.append(
            f"counts: ready={ready} in_progress={in_progress} "
            f"blocked={blocked} completed={completed}"
        )
        return "\n".join(lines)

    def prompt_hint(self, max_chars: int = 4000) -> str:
        tasks = self.tasks()
        by_id = {task["id"]: task for task in tasks}
        unfinished = [task for task in tasks if task["status"] != "completed"]
        if not unfinished:
            return ""

        progressing = [task for task in unfinished if task["status"] == "in_progress"]
        pending = [task for task in unfinished if task["status"] == "pending"]
        selected = (progressing + pending)[:12]
        omitted = max(0, len(unfinished) - len(selected))

        lines = ["## Task Graph", "Current persistent plan:"]
        for task in selected:
            state = task["status"]
            if state == "pending":
                state = _readiness(task, by_id)
            subject = " ".join(task["subject"].split())
            line = f"- {task['id']} [{state}] {subject}"
            if task.get("blocked_by"):
                line += f" (blocked_by: {', '.join(task['blocked_by'])})"
            if len("\n".join([*lines, line])) > max_chars:
                omitted += 1
                continue
            lines.append(line)
        if omitted:
            lines.append(f"- ... {omitted} more unfinished task(s)")
        return "\n".join(lines)[:max_chars]

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
    valid_statuses = {"pending", "in_progress", "completed"}
    for task in tasks:
        if not isinstance(task, dict):
            raise TaskGraphError("task graph is corrupt: task must be an object")
        task_id = task.get("id")
        if not isinstance(task_id, str) or not re.fullmatch(
            r"task_[0-9a-f]{8}", task_id
        ):
            raise TaskGraphError("task graph is corrupt: invalid task id")
        if task_id in seen:
            raise TaskGraphError("task graph is corrupt: duplicate task id")
        seen.add(task_id)
        if task.get("status") not in valid_statuses:
            raise TaskGraphError("task graph is corrupt: invalid task status")
        if not isinstance(task.get("subject"), str) or not task["subject"].strip():
            raise TaskGraphError("task graph is corrupt: invalid task subject")
        for field in ("description", "result"):
            if not isinstance(task.get(field), str):
                raise TaskGraphError(
                    f"task graph is corrupt: invalid task {field}"
                )
        dependencies = task.get("blocked_by")
        if not isinstance(dependencies, list) or any(
            not isinstance(item, str) for item in dependencies
        ):
            raise TaskGraphError("task graph is corrupt: invalid task dependencies")

    for task in tasks:
        unknown = [item for item in task["blocked_by"] if item not in seen]
        if unknown:
            raise TaskGraphError(
                "task graph is corrupt: unknown task dependency"
            )


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


def _readiness(task: dict, by_id: dict[str, dict]) -> str:
    if task.get("status") != "pending":
        return task.get("status", "pending")
    blockers = [
        task_id
        for task_id in task.get("blocked_by", [])
        if by_id.get(task_id, {}).get("status") != "completed"
    ]
    return "blocked" if blockers else "ready"


class TaskCreateTool(Tool):
    name = "TaskCreate"
    description = (
        "Create a persistent task. Dependencies must reference existing task IDs."
    )
    input_schema = {
        "subject": {"type": "string", "description": "Short task title"},
        "description": {"type": "string", "description": "Optional task details"},
        "blocked_by": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Existing task IDs that must complete first",
        },
    }
    is_readonly = False
    is_concurrency_safe = False

    def __init__(self, store: TaskGraphStore):
        self.store = store

    async def execute(
        self,
        subject: str,
        description: str = "",
        blocked_by: list[str] | None = None,
    ) -> str:
        task = self.store.create(subject, description, blocked_by)
        return f"created: {task['id']}\n{self.store.render_snapshot()}"


class TaskUpdateTool(Tool):
    name = "TaskUpdate"
    description = (
        "Start or complete an existing task after validating dependencies and state."
    )
    input_schema = {
        "task_id": {"type": "string", "description": "Task ID to update"},
        "status": {
            "type": "string",
            "enum": ["in_progress", "completed"],
            "description": "Target state",
        },
        "result": {"type": "string", "description": "Optional completion evidence"},
    }
    is_readonly = False
    is_concurrency_safe = False

    def __init__(self, store: TaskGraphStore):
        self.store = store

    async def execute(self, task_id: str, status: str, result: str = "") -> str:
        self.store.update(task_id, status, result)
        return self.store.render_snapshot()


class TaskListTool(Tool):
    name = "TaskList"
    description = (
        "List persistent tasks, dependency blockers, and status counts."
    )
    input_schema = {
        "include_completed": {
            "type": "boolean",
            "description": "Include completed tasks in the output",
        },
    }
    is_readonly = True
    is_concurrency_safe = True

    def __init__(self, store: TaskGraphStore):
        self.store = store

    async def execute(self, include_completed: bool = False) -> str:
        return self.store.render_snapshot(include_completed=include_completed)

