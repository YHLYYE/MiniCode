import sqlite3

import pytest

from capabilities.memory import (
    MemoryEntry,
    MemoryManager,
    MemoryType,
    SQLiteMemoryStore,
)


def test_old_sqlite_rows_migrate_with_safe_defaults(tmp_path):
    db = tmp_path / "old.sqlite"
    conn = sqlite3.connect(db)
    conn.execute("""CREATE TABLE memories (
        id TEXT PRIMARY KEY, memory_type TEXT, content TEXT, context TEXT,
        file_paths TEXT, tags TEXT, timestamp REAL, success INTEGER,
        fingerprint TEXT, hit_count INTEGER)""")
    conn.execute(
        "INSERT INTO memories VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("old", "procedural", "旧经验", "", "[]", "[]", 1.0, 1, "fp", 1),
    )
    conn.commit()
    conn.close()

    entry = SQLiteMemoryStore(db).search(MemoryType.PROCEDURAL, "旧经验", 3)[0]
    assert entry.scope == "persistent"
    assert entry.status == "active"
    assert entry.success_count == 0
    assert entry.failure_count == 0


def test_memory_entry_metadata_round_trip():
    entry = MemoryEntry.make(
        MemoryType.PROCEDURAL,
        "先跑 pytest",
        scope="current_task",
        source="task_log",
        evidence_level="observed",
        subject_key="pre_commit_test",
        status="conflicted",
        conflict_group="cg_1",
        supersedes_id="old",
        success_count=2,
        failure_count=1,
    )

    restored = MemoryEntry.from_dict(entry.to_dict())

    assert restored.scope == "current_task"
    assert restored.subject_key == "pre_commit_test"
    assert restored.status == "conflicted"
    assert restored.conflict_group == "cg_1"
    assert restored.success_count == 2
    assert restored.failure_count == 1


@pytest.mark.asyncio
async def test_current_task_memory_is_not_automatically_injected(tmp_path):
    mm = MemoryManager(project_root=tmp_path)
    await mm.record_procedural(
        "本次任务的临时路径 /tmp/build",
        scope="current_task",
        source="task_log",
    )

    assert mm.search_sync("临时路径 build", scope="persistent") == []
    assert any(
        entry.scope == "current_task"
        for entry in mm.search_sync("临时路径 build")
    )


@pytest.mark.asyncio
async def test_new_user_correction_supersedes_old_rule(tmp_path):
    store = SQLiteMemoryStore(tmp_path / "mem.sqlite")
    mm = MemoryManager(project_root=tmp_path, store=store)
    old = await mm.record_procedural(
        "用户纠正：使用 Tab 缩进",
        source="user_correction",
        evidence_level="user",
        subject_key="indent_style",
    )
    new = await mm.record_procedural(
        "用户纠正：使用 4 个空格缩进",
        source="user_correction",
        evidence_level="user",
        subject_key="indent_style",
    )

    assert store.get(old.id).status == "superseded"
    assert store.get(new.id).status == "active"
    assert store.get(new.id).supersedes_id == old.id


@pytest.mark.asyncio
async def test_uncertain_same_subject_entries_are_marked_conflicted(tmp_path):
    store = SQLiteMemoryStore(tmp_path / "mem.sqlite")
    mm = MemoryManager(project_root=tmp_path, store=store)
    first = await mm.record_procedural(
        "提交前先跑 pytest",
        subject_key="pre_commit_test",
        source="verified_fix",
        evidence_level="verified",
    )
    second = await mm.record_procedural(
        "提交前不用跑 pytest，CI 会跑",
        subject_key="pre_commit_test",
        source="observed",
        evidence_level="observed",
    )

    assert store.get(first.id).status == "conflicted"
    assert store.get(second.id).status == "conflicted"
    assert store.get(first.id).conflict_group == store.get(second.id).conflict_group
