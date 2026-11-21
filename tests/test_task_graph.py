import json

import pytest

from config import Config
from capabilities.task_graph import (
    TaskCreateTool,
    TaskGraphError,
    TaskGraphStore,
    TaskListTool,
    TaskUpdateTool,
)
from main import _build_tools
from prompt.system_prompt import build_system_prompt


def test_create_task_persists_and_reloads(tmp_path):
    store = TaskGraphStore(tmp_path)
    task = store.create("实现存储", "写入 JSON")

    assert task["status"] == "pending"
    assert task["id"].startswith("task_")
    assert (tmp_path / ".minicode" / "tasks.json").exists()

    reloaded = TaskGraphStore(tmp_path).get(task["id"])
    assert reloaded == task


def test_unknown_dependency_is_rejected(tmp_path):
    store = TaskGraphStore(tmp_path)

    with pytest.raises(TaskGraphError, match="unknown dependency"):
        store.create("后续任务", blocked_by=["task_ffffffff"])


def test_blocked_task_cannot_start_until_dependency_completes(tmp_path):
    store = TaskGraphStore(tmp_path)
    first = store.create("第一步")
    second = store.create("第二步", blocked_by=[first["id"]])

    with pytest.raises(TaskGraphError, match="blocked by"):
        store.update(second["id"], "in_progress")

    store.update(first["id"], "in_progress")
    store.update(first["id"], "completed", result="done")
    started = store.update(second["id"], "in_progress")
    assert started["status"] == "in_progress"


def test_task_cannot_skip_in_progress(tmp_path):
    store = TaskGraphStore(tmp_path)
    task = store.create("不能跳级")

    with pytest.raises(TaskGraphError, match="pending -> completed"):
        store.update(task["id"], "completed")


def test_repeated_status_update_is_idempotent(tmp_path):
    store = TaskGraphStore(tmp_path)
    task = store.create("重复更新")
    started = store.update(task["id"], "in_progress")
    repeated = store.update(task["id"], "in_progress")

    assert repeated["started_at"] == started["started_at"]


def test_corrupt_task_file_fails_closed(tmp_path):
    path = tmp_path / ".minicode" / "tasks.json"
    path.parent.mkdir(parents=True)
    path.write_text("{broken", encoding="utf-8")

    store = TaskGraphStore(tmp_path)
    with pytest.raises(TaskGraphError, match="corrupt"):
        store.tasks()
    with pytest.raises(TaskGraphError, match="corrupt"):
        store.create("不能覆盖")
    assert path.read_text(encoding="utf-8") == "{broken"


@pytest.mark.asyncio
async def test_tools_emit_canonical_snapshot(tmp_path):
    store = TaskGraphStore(tmp_path)
    created = await TaskCreateTool(store).execute(subject="实现工具")
    task_id = created.splitlines()[0].split(":", 1)[1].strip()

    listed = await TaskListTool(store).execute()
    assert "## Task Graph" in listed
    assert task_id in listed
    assert "ready=1" in listed

    updated = await TaskUpdateTool(store).execute(
        task_id=task_id, status="in_progress"
    )
    assert "in_progress=1" in updated


@pytest.mark.asyncio
async def test_blocked_count_is_derived_from_dependencies(tmp_path):
    store = TaskGraphStore(tmp_path)
    first = store.create("第一步")
    second = store.create("第二步", blocked_by=[first["id"]])

    output = await TaskListTool(store).execute()

    assert second["id"] in output
    assert "blocked=1" in output


@pytest.mark.asyncio
async def test_task_list_can_include_completed(tmp_path):
    store = TaskGraphStore(tmp_path)
    task = store.create("已完成任务")
    store.update(task["id"], "in_progress")
    store.update(task["id"], "completed")

    without = await TaskListTool(store).execute()
    with_completed = await TaskListTool(store).execute(include_completed=True)

    assert task["id"] not in without
    assert task["id"] in with_completed


@pytest.mark.parametrize(
    "task_patch",
    [
        {"status": None},
        {"subject": None},
        {"blocked_by": "not-a-list"},
        {"blocked_by": ["task_ffffffff"]},
    ],
)
def test_corrupt_task_fields_fail_closed(tmp_path, task_patch):
    path = tmp_path / ".minicode" / "tasks.json"
    path.parent.mkdir(parents=True)
    task = {
        "id": "task_12345678",
        "subject": "broken",
        "description": "",
        "status": "pending",
        "blocked_by": [],
        "result": "",
        "created_at": "2026-11-21T00:00:00+00:00",
        "started_at": None,
        "completed_at": None,
    }
    task.update(task_patch)
    path.write_text(
        json.dumps({"version": 1, "tasks": [task]}), encoding="utf-8"
    )

    with pytest.raises(TaskGraphError, match="corrupt"):
        TaskGraphStore(tmp_path).tasks()


def test_prompt_injects_open_tasks_only(tmp_path):
    store = TaskGraphStore(tmp_path)
    open_task = store.create("未完成")
    done = store.create("已完成")
    store.update(done["id"], "in_progress")
    store.update(done["id"], "completed")

    hint = store.prompt_hint()

    assert open_task["id"] in hint
    assert done["id"] not in hint


def test_system_prompt_renders_task_graph_hint(tmp_path):
    store = TaskGraphStore(tmp_path)
    task = store.create("提示词任务")

    prompt = build_system_prompt(task_graph_hint=store.prompt_hint())

    assert "## Task Graph" in prompt
    assert task["id"] in prompt


def test_prompt_reports_corrupt_graph_without_crashing(
    tmp_path, monkeypatch, capsys
):
    (tmp_path / ".minicode").mkdir()
    (tmp_path / ".minicode" / "tasks.json").write_text(
        "{broken", encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)

    _, prompt, _, memory_manager, factory = _build_tools("normal", Config())
    try:
        assert "Task Graph Unavailable" in prompt
        assert "Task Graph Unavailable" in factory("new task")
        assert "Task graph unavailable" in capsys.readouterr().out
    finally:
        memory_manager.close()
