import pytest

from capabilities.task_graph import TaskGraphError, TaskGraphStore


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
