# MiniCode Memory Correctness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Make persistent memory admission-safe, conflict-aware and driven by verified task outcomes without adding request-time model calls.

**Architecture:** Extend `MemoryEntry` and both stores with scope, provenance, conflict and outcome metadata. Writes apply deterministic duplicate/replacement/conflict rules. Prompt injection reads only persistent active/conflicted entries, while `AgentLoop` reports terminal success to update outcome counters.

**Tech Stack:** Python 3.11+, dataclasses, SQLite schema migration, BM25, pytest, pytest-asyncio

---

## File Structure

- Modify `capabilities/memory.py`: entry fields, schema migration, scope filtering, conflict state, ranking, outcome counters.
- Modify `core/agent_loop.py`: assign temporary scopes and report explicit terminal success.
- Modify `main.py`: automatic memory injection filters persistent entries.
- Create `tests/test_memory_conflicts.py`: conflicts, read-time annotation, migration and scope behavior.
- Modify `tests/test_memory.py`: compatible fields and outcome/task behavior.
- Modify `README.md`, `DESIGN.md`, `ROADMAP.md`: document scoped/conflict-aware memory.

### Task 1: Extend MemoryEntry and SQLite Schema

**Files:**
- Modify: `capabilities/memory.py`
- Test: `tests/test_memory_conflicts.py`

- [x] **Step 1: Write migration and round-trip tests**

```python
import sqlite3

from capabilities.memory import MemoryEntry, MemoryType, SQLiteMemoryStore


def test_old_sqlite_rows_migrate_with_safe_defaults(tmp_path):
    db = tmp_path / "old.sqlite"
    conn = sqlite3.connect(db)
    conn.execute("""CREATE TABLE memories (
        id TEXT PRIMARY KEY, memory_type TEXT, content TEXT, context TEXT,
        file_paths TEXT, tags TEXT, timestamp REAL, success INTEGER,
        fingerprint TEXT, hit_count INTEGER)""")
    conn.execute("INSERT INTO memories VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                 ("old", "procedural", "旧经验", "", "[]", "[]", 1.0, 1, "fp", 1))
    conn.commit()
    conn.close()

    entry = SQLiteMemoryStore(db).search(MemoryType.PROCEDURAL, "旧经验", 3)[0]
    assert entry.scope == "persistent"
    assert entry.status == "active"
    assert entry.success_count == 0
    assert entry.failure_count == 0


def test_memory_entry_metadata_round_trip():
    entry = MemoryEntry.make(
        MemoryType.PROCEDURAL, "先跑 pytest",
        scope="current_task", source="task_log", evidence_level="observed",
        subject_key="pre_commit_test", status="conflicted",
        conflict_group="cg_1", supersedes_id="old", success_count=2,
        failure_count=1,
    )
    restored = MemoryEntry.from_dict(entry.to_dict())
    assert restored.scope == "current_task"
    assert restored.subject_key == "pre_commit_test"
    assert restored.status == "conflicted"
    assert restored.conflict_group == "cg_1"
    assert restored.success_count == 2
    assert restored.failure_count == 1
```

- [x] **Step 2: Run tests and verify they fail**

Run: `python -m pytest tests/test_memory_conflicts.py -q`  
Expected: FAIL because the new fields and migration columns are absent.

- [x] **Step 3: Extend `MemoryEntry`**

```python
@dataclass
class MemoryEntry:
    ...
    scope: str = "persistent"
    source: str = "observed"
    evidence_level: str = "observed"
    subject_key: str = ""
    status: str = "active"
    conflict_group: str = ""
    supersedes_id: str = ""
    success_count: int = 0
    failure_count: int = 0
```

Also update `to_dict()`, `from_dict()`, and `MemoryEntry.make()` with safe defaults. Unknown persisted scope/source/evidence/status values normalize to `persistent / observed / observed / active`.

- [x] **Step 4: Make SQLite migration idempotent**

Extend `SQLiteMemoryStore._migrate()` with:

```python
columns = (
    ("scope", "TEXT DEFAULT 'persistent'"),
    ("source", "TEXT DEFAULT 'observed'"),
    ("evidence_level", "TEXT DEFAULT 'observed'"),
    ("subject_key", "TEXT DEFAULT ''"),
    ("status", "TEXT DEFAULT 'active'"),
    ("conflict_group", "TEXT DEFAULT ''"),
    ("supersedes_id", "TEXT DEFAULT ''"),
    ("success_count", "INTEGER DEFAULT 0"),
    ("failure_count", "INTEGER DEFAULT 0"),
)
```

Update inserts/selects/load helpers so every operation includes the columns.

- [x] **Step 5: Run focused tests and commit**

Run: `python -m pytest tests/test_memory_conflicts.py tests/test_memory.py -q`  
Expected: PASS.

```bash
git add capabilities/memory.py tests/test_memory_conflicts.py tests/test_memory.py
git commit -m "feat: add memory provenance and conflict schema"
```

### Task 2: Enforce Scope and Conflict Rules on Write

**Files:**
- Modify: `capabilities/memory.py`
- Test: `tests/test_memory_conflicts.py`

- [x] **Step 1: Write scope and conflict tests**

```python
@pytest.mark.asyncio
async def test_current_task_memory_is_not_automatically_injected(tmp_path):
    mm = MemoryManager(project_root=tmp_path)
    await mm.record_procedural("本次任务的临时路径 /tmp/build", scope="current_task")
    assert mm.search_sync("临时路径 build", scope="persistent") == []
    assert any(e.scope == "current_task" for e in mm.search_sync("临时路径 build"))


@pytest.mark.asyncio
async def test_new_user_correction_supersedes_old_rule(tmp_path):
    store = SQLiteMemoryStore(tmp_path / "mem.sqlite")
    mm = MemoryManager(project_root=tmp_path, store=store)
    old = await mm.record_procedural(
        "用户纠正：使用 Tab 缩进", scope="persistent",
        source="user_correction", evidence_level="user",
        subject_key="indent_style",
    )
    new = await mm.record_procedural(
        "用户纠正：使用 4 个空格缩进", scope="persistent",
        source="user_correction", evidence_level="user",
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
        "提交前先跑 pytest", subject_key="pre_commit_test",
        source="verified_fix", evidence_level="verified",
    )
    second = await mm.record_procedural(
        "提交前不用跑 pytest，CI 会跑", subject_key="pre_commit_test",
        source="observed", evidence_level="observed",
    )
    assert store.get(first.id).status == "conflicted"
    assert store.get(second.id).status == "conflicted"
    assert store.get(first.id).conflict_group == store.get(second.id).conflict_group
```

- [x] **Step 2: Run tests and verify they fail**

Run: `python -m pytest tests/test_memory_conflicts.py -q`  
Expected: FAIL on absent scope filtering and conflict metadata.

- [x] **Step 3: Add validated write admission and relation detection**

In `MemoryManager._add()` validate:

```python
VALID_SCOPES = {"persistent", "current_task"}
VALID_SOURCES = {"user_correction", "verified_fix", "explicit", "task_log", "observed"}
VALID_EVIDENCE = {"user", "verified", "observed", "inferred"}
VALID_STATUSES = {"active", "conflicted", "superseded"}
```

Before insertion, inspect existing entries of the same `memory_type` and persistent/current-task scope:

```python
if normalized_fingerprint is identical:
    bump existing timestamp/success and return existing

if same subject_key and new source == "user_correction":
    mark old rows superseded
    new.supersedes_id = newest_old.id

elif same subject_key and evidence/version does not prove replacement:
    group = new or existing conflict_group
    mark all rows conflicted

elif keyword overlap >= CONFLICT_SIMILARITY_THRESHOLD and subject_key is missing:
    group and mark conflicted

else:
    insert active
```

Use `keyword_tokens` for deterministic overlap. Never call a model.

- [x] **Step 4: Add scope to search**

Update abstract and concrete store signatures:

```python
def search(self, memory_type, query, top_k, min_similarity=None, scope=None):
    ...
```

`scope=None` keeps all entries for manual retrieval; `scope="persistent"` filters before BM25.

- [x] **Step 5: Run focused tests and commit**

Run: `python -m pytest tests/test_memory_conflicts.py tests/test_memory.py tests/test_memory_eval.py -q`  
Expected: PASS.

```bash
git add capabilities/memory.py tests/test_memory_conflicts.py
git commit -m "feat: enforce memory scope and write-time conflicts"
```

### Task 3: Persistent-Only Automatic Injection and Conflict Labels

**Files:**
- Modify: `main.py`
- Modify: `capabilities/memory.py`
- Test: `tests/test_memory_conflicts.py`

- [x] **Step 1: Write injection and formatting tests**

```python
@pytest.mark.asyncio
async def test_current_task_entry_is_not_injected_by_default(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    async def seed():
        mm = MemoryManager(project_root=tmp_path)
        await mm.record_procedural(
            "本次任务临时使用 /tmp/build，清理后再提交",
            scope="current_task", source="task_log",
        )
    asyncio.run(seed())
    _, prompt, _, mm, _ = _build_tools("normal", Config())
    try:
        assert "/tmp/build" not in prompt
    finally:
        mm.close()


def test_conflicted_memory_hint_is_labeled(tmp_path):
    mm = MemoryManager(project_root=tmp_path)
    a = asyncio.run(mm.record_procedural("提交前先跑 pytest", status="active"))
    b = asyncio.run(mm.record_procedural("提交前不用跑 pytest", status="active"))
    # Test helper directly formats conflict metadata without hiding it.
    b.conflict_group = "cg_1"
    b.status = "conflicted"
    a.conflict_group = "cg_1"
    a.status = "conflicted"
    hint = format_memory_hint([a, b])
    assert "[存在冲突]" in hint
    assert "历史记录，仅供参考" in hint
```

- [x] **Step 2: Run tests and verify they fail**

Run: `python -m pytest tests/test_memory_conflicts.py -q`  
Expected: FAIL because automatic search is unfiltered and formatting has no conflict label.

- [x] **Step 3: Filter automatic injection**

Change `main.py`:

```python
hits = memory_manager.search_sync(
    task,
    top_k=MEMORY_HINT_TOP_K,
    scope="persistent",
    min_similarity=MEMORY_INJECT_MIN_SIMILARITY,
)
```

The `RecallMemory` tool keeps `scope=None` as its default.

- [x] **Step 4: Label active/conflicted entries in `format_memory_hint`**

```python
conflict = " [存在冲突]" if entry.status == "conflicted" else ""
line = f"- [{label}]{conflict} {entry.content}"
...
if any(e.status == "conflicted" for e in entries):
    lines.append("历史记录，仅供参考；多条记录冲突时请以当前代码和用户请求为准。")
```

Keep the 600-character cap; append the note only if it fits.

- [x] **Step 5: Run focused tests and commit**

Run: `python -m pytest tests/test_memory_conflicts.py tests/test_wiring.py tests/test_memory_eval.py -q`  
Expected: PASS.

```bash
git add main.py capabilities/memory.py tests/test_memory_conflicts.py
git commit -m "feat: inject only persistent memory and label conflicts"
```

### Task 4: Record Task Outcomes from Explicit Terminal State

**Files:**
- Modify: `core/agent_loop.py`
- Modify: `capabilities/memory.py`
- Test: `tests/test_memory.py`
- Test: `tests/test_memory_conflicts.py`

- [x] **Step 1: Write outcome tests**

```python
@pytest.mark.asyncio
async def test_verified_success_increments_success_count(tmp_path):
    mm = MemoryManager(project_root=tmp_path)
    entry = await mm.record_procedural("修复 NPE", subject_key="npe")
    mm.record_injection([entry])
    mm.record_task_outcome(success=True)
    assert mm.search_sync("修复 NPE", top_k=1)[0].success_count == 1


@pytest.mark.asyncio
async def test_terminal_failure_increments_failure_count(tmp_path):
    mm = MemoryManager(project_root=tmp_path)
    entry = await mm.record_procedural("修复 NPE", subject_key="npe")
    mm.record_injection([entry])
    mm.record_task_outcome(success=False)
    assert mm.search_sync("修复 NPE", top_k=1)[0].failure_count == 1
```

- [x] **Step 2: Run tests and verify they fail**

Run: `python -m pytest tests/test_memory_conflicts.py -q`  
Expected: FAIL because only `adoption_count` is updated.

- [x] **Step 3: Update outcome settlement**

`MemoryManager.record_task_outcome()` should bump one of:

```python
field = "success_count" if success else "failure_count"
changed = self._store.bump(self._last_injected_ids, field, 1)
self._last_injected_ids = []
return changed
```

Catch store errors in the caller so outcome accounting never changes task termination.

- [x] **Step 4: Pass explicit success from AgentLoop**

Change `_record_task_done`:

```python
async def _record_task_done(self, note="", *, success: bool):
    ...
    outcome = getattr(self._memory, "record_task_outcome", None)
    if callable(outcome):
        try:
            outcome(success=success)
        except Exception:
            pass
```

Call sites use `success=False` for max tokens, empty response, verification failure and max turns; use `success=True` only for `COMPLETED`.

- [x] **Step 5: Ensure temporary event scopes**

Use `scope="current_task"` for raw tool-error episodes, file-operation episodes and task-completion episodes. Keep user corrections and distilled error-to-fix procedural memories persistent.

- [x] **Step 6: Run tests and commit**

Run: `python -m pytest tests/test_memory.py tests/test_memory_conflicts.py tests/test_agent_loop.py -q`  
Expected: PASS.

```bash
git add core/agent_loop.py capabilities/memory.py tests/test_memory.py tests/test_memory_conflicts.py
git commit -m "feat: settle memory feedback from terminal outcomes"
```

### Task 5: Ranking, Documentation and Full Verification

**Files:**
- Modify: `capabilities/memory.py`
- Modify: `README.md`
- Modify: `DESIGN.md`
- Modify: `ROADMAP.md`
- Test: `tests/test_memory_conflicts.py`

- [x] **Step 1: Write ranking test**

```python
def test_successful_memory_outranks_failed_peer(tmp_path):
    mm = MemoryManager(project_root=tmp_path)
    good = asyncio.run(mm.record_procedural("pytest pre commit check"))
    bad = asyncio.run(mm.record_procedural("pytest pre commit skip"))
    good.success_count = 3
    bad.failure_count = 3
    hits = _rank_procedural([good.to_dict(), bad.to_dict()], "pytest pre commit", 2)
    assert hits[0]["content"] == "pytest pre commit check"
```

- [x] **Step 2: Run test and verify it fails**

Run: `python -m pytest tests/test_memory_conflicts.py::test_successful_memory_outranks_failed_peer -q`  
Expected: FAIL because ranking still uses `adoption_count`.

- [x] **Step 3: Update ranking bonus**

Use:

```python
success = min(int(entry.get("success_count", 0) or 0), 5)
failure = min(int(entry.get("failure_count", 0) or 0), 5)
bonus *= max(0.5, 1.0 + 0.05 * success - 0.03 * failure)
```

Keep all BM25 score and strict-evidence behavior unchanged.

- [x] **Step 4: Run memory tests**

Run: `python -m pytest tests/test_memory.py tests/test_memory_eval.py tests/test_memory_conflicts.py -q`  
Expected: PASS with existing recall floors unchanged.

- [x] **Step 5: Update documentation**

Document:

- `persistent / current_task` scope;
- write-time conflict states and replacement rules;
- read-time conflict label;
- success/failure outcome counters;
- no model extraction, conditional rerank or offline merge in this release.

Update test counts only after the full suite passes.

- [x] **Step 6: Run all gates**

```bash
python -m pytest -q
python benchmarks/verify_claims.py
python -m compileall -q .
python -m pyflakes .
git diff --check
```

Expected: every command exits 0 and documentation counts match pytest.

- [x] **Step 7: Commit**

```bash
git add capabilities/memory.py README.md DESIGN.md ROADMAP.md tests/test_memory_conflicts.py
git commit -m "docs: document scoped conflict-aware memory"
```

## Final Acceptance Checklist

- [x] Old SQLite data migrates without loss.
- [x] Automatic injection uses only persistent entries.
- [x] Temporary entries remain manually recallable.
- [x] User corrections supersede older rules.
- [x] Uncertain same-topic entries are marked conflicted.
- [x] Conflict labels appear in memory hints.
- [x] Success/failure counters use explicit terminal outcomes.
- [x] Outcome accounting failure cannot crash a task.
- [x] BM25 and recall quality floors remain green.
- [x] Full tests and documentation gates pass.
