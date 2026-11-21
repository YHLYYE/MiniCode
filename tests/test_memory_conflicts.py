import sqlite3

from capabilities.memory import MemoryEntry, MemoryType, SQLiteMemoryStore


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
