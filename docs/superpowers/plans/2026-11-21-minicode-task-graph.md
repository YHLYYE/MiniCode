# MiniCode Persistent Task Graph Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Add a model-operated persistent task graph with dependency validation, deterministic state transitions, session recovery, and compression anchoring.

**Architecture:** A project-scoped JSON store owns persistence and invariants. Three tools expose create/update/list operations to the model. Main wiring shares one store instance with sub-agents, while the prompt and compressor consume the same store snapshot.

**Tech Stack:** Python 3.11+, dataclasses, `threading.RLock`, JSON, `os.replace`, pytest, pytest-asyncio

---

## File Structure

- Create `capabilities/task_graph.py`: store, validation, formatting, and three tool classes.
- Create `tests/test_task_graph.py`: storage, state-machine, tool, prompt, and recovery tests.
- Modify `main.py`: instantiate the shared store, register tools, and inject prompt hints.
- Modify `prompt/system_prompt.py`: render the task graph dynamic section.
- Modify `capabilities/security.py`: allow only create/list in Plan mode.
- Modify `core/agent_loop.py`: register successful task snapshots as compression anchors.
- Modify `tests/test_wiring.py`: assert normal/plan tool sets and shared sub-agent store.
- Modify `tests/test_task_anchor.py`: prove the graph survives all compression tiers.
- Modify `README.md`, `DESIGN.md`, `ROADMAP.md`: document the verified feature.

### Task 1: Persistent Task Store

**Files:**
- Create: `capabilities/task_graph.py`
- Test: `tests/test_task_graph.py`

- [x] **Step 1: Write storage and state tests**

```python
import json
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
```

- [x] **Step 2: Run tests and verify they fail**

Run: `python -m pytest tests/test_task_graph.py -q`
Expected: collection fails because `capabilities.task_graph` does not exist.

- [x] **Step 3: Implement the store**

```python
class TaskGraphError(RuntimeError):
    pass


class TaskGraphStore:
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
            deps = list(dict.fromkeys(blocked_by or []))
            unknown = [task_id for task_id in deps if task_id not in existing]
            if unknown:
                raise TaskGraphError(f"unknown dependency: {', '.join(unknown)}")
            task_id = self._new_id(existing)
            now = _utc_now()
            task = {
                "id": task_id,
                "subject": _validated_text(subject, 200, "subject"),
                "description": _validated_text(description, 4000, "description", allow_empty=True),
                "status": "pending",
                "blocked_by": deps,
                "result": "",
                "created_at": now,
                "started_at": None,
                "completed_at": None,
            }
            data["tasks"].append(task)
            self._save_unlocked(data)
            return dict(task)

    def update(self, task_id: str, status: str, result: str = "") -> dict:
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
                    raise TaskGraphError(f"task is blocked by: {', '.join(blockers)}")
                task["status"] = status
                task["started_at"] = _utc_now()
            elif status == "completed" and current == "in_progress":
                task["status"] = status
                task["completed_at"] = _utc_now()
                if result:
                    task["result"] = _validated_text(result, 8000, "result")
            else:
                raise TaskGraphError(f"invalid transition: {current} -> {status}")
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
        temp = self.path.with_suffix(f".json.{secrets.token_hex(4)}.tmp")
        try:
            temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")
            os.replace(temp, self.path)
        finally:
            if temp.exists():
                temp.unlink()
```

Also implement `_utc_now`, `_new_id`, `_validated_text`, `_find_task`, `_unfinished_blockers`, and `_validate_graph`. `_validate_graph` rejects unknown versions, non-list tasks, duplicate IDs, invalid statuses, non-string subjects, and unknown dependency IDs.

- [x] **Step 4: Run tests**

Run: `python -m pytest tests/test_task_graph.py -q`
Expected: PASS for storage and state-machine tests.

- [x] **Step 5: Commit**

```bash
git add capabilities/task_graph.py tests/test_task_graph.py
git commit -m "feat: add persistent task graph store"
```

### Task 2: Model-Facing Tools and Snapshot Format

**Files:**
- Modify: `capabilities/task_graph.py`
- Test: `tests/test_task_graph.py`

- [x] **Step 1: Write tool-output tests**

```python
import pytest

from capabilities.task_graph import TaskCreateTool, TaskListTool, TaskUpdateTool, TaskGraphStore


@pytest.mark.asyncio
async def test_tools_emit_canonical_snapshot(tmp_path):
    store = TaskGraphStore(tmp_path)
    created = await TaskCreateTool(store).execute(subject="实现工具")
    task_id = created.splitlines()[0].split(":", 1)[1].strip()

    listed = await TaskListTool(store).execute()
    assert "## Task Graph" in listed
    assert task_id in listed
    assert "ready=1" in listed

    updated = await TaskUpdateTool(store).execute(task_id=task_id, status="in_progress")
    assert "in_progress=1" in updated


@pytest.mark.asyncio
async def test_blocked_count_is_derived_from_dependencies(tmp_path):
    store = TaskGraphStore(tmp_path)
    first = store.create("第一步")
    second = store.create("第二步", blocked_by=[first["id"]])
    output = await TaskListTool(store).execute()
    assert second["id"] in output
    assert "blocked=1" in output
```

- [x] **Step 2: Run tests and verify they fail**

Run: `python -m pytest tests/test_task_graph.py -q`
Expected: import fails because the three tool classes are missing.

- [x] **Step 3: Implement tools and rendering**

```python
class TaskCreateTool(Tool):
    name = "TaskCreate"
    description = "Create a persistent task. Dependencies must reference existing tasks."
    input_schema = {
        "subject": {"type": "string"},
        "description": {"type": "string"},
        "blocked_by": {"type": "array", "items": {"type": "string"}},
    }
    is_readonly = False
    is_concurrency_safe = False

    def __init__(self, store: TaskGraphStore):
        self.store = store

    async def execute(self, subject: str, description: str = "", blocked_by: list[str] | None = None) -> str:
        task = self.store.create(subject, description, blocked_by)
        return f"created: {task['id']}\n{self.store.render_snapshot()}"


class TaskUpdateTool(Tool):
    name = "TaskUpdate"
    description = "Start or complete an existing task after validating dependencies and state."
    input_schema = {
        "task_id": {"type": "string"},
        "status": {"type": "string", "enum": ["in_progress", "completed"]},
        "result": {"type": "string"},
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
    description = "List persistent tasks, dependency blockers, and status counts."
    input_schema = {"include_completed": {"type": "boolean"}}
    is_readonly = True
    is_concurrency_safe = True

    def __init__(self, store: TaskGraphStore):
        self.store = store

    async def execute(self, include_completed: bool = False) -> str:
        return self.store.render_snapshot(include_completed=include_completed)
```

`render_snapshot()` headings, status rows, blockers, and counts. It starts with `## Task Graph` so the Loop can identify valid snapshots.

- [x] **Step 4: Run tests**

Run: `python -m pytest tests/test_task_graph.py -q`
Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add capabilities/task_graph.py tests/test_task_graph.py
git commit -m "feat: expose task graph tools"
```

### Task 3: Main Wiring, Plan Mode, and Shared Sub-Agent Store

**Files:**
- Modify: `main.py`
- Modify: `capabilities/security.py`
- Modify: `tests/test_wiring.py`

- [x] **Step 1: Write wiring tests**

```python
def test_normal_mode_registers_task_graph_tools(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tools, _, _, memory_manager, _ = _unpack("normal")
    try:
        names = {tool.name for tool in tools}
        assert {"TaskCreate", "TaskUpdate", "TaskList"} <= names
    finally:
        memory_manager.close()


def test_plan_mode_allows_create_and_list_but_not_update(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tools, _, _, memory_manager, _ = _unpack("plan")
    try:
        names = {tool.name for tool in tools}
        assert {"TaskCreate", "TaskList"} <= names
        assert "TaskUpdate" not in names
    finally:
        memory_manager.close()


def test_subagent_factory_reuses_parent_task_store(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tools, _, _, memory_manager, _ = _unpack("normal")
    try:
        parent = next(tool for tool in tools if tool.name == "TaskCreate")
        agent = next(tool for tool in tools if tool.name == "Agent")
        child_tools = agent._workspace_tool_factory(tmp_path / "child")
        assert child_tools["TaskCreate"].store is parent.store
    finally:
        memory_manager.close()
```

- [x] **Step 2: Run tests and verify they fail**

Run: `python -m pytest tests/test_wiring.py -q`
Expected: task tools are absent; Plan-mode assertions fail.

- [x] **Step 3: Wire the store into main**

```python
from capabilities.task_graph import (
    TaskCreateTool, TaskGraphStore, TaskListTool, TaskUpdateTool,
)

task_store = TaskGraphStore(Path.cwd())
task_tools = [
    TaskCreateTool(task_store),
    TaskUpdateTool(task_store),
    TaskListTool(task_store),
]

base_tools = [
    ReadTool(), WriteTool(), EditTool(), main_bash, GrepTool(), GlobTool(),
    WebSearchTool(), WebFetchTool(), TodoWriteTool(),
    *task_tools,
    SkillTool(skill_system), RecallMemoryTool(memory_manager), RememberTool(memory_manager),
]

def workspace_tool_factory(root):
    return {
        tool.name: tool
        for tool in [
            ReadTool(workspace_root=root), WriteTool(workspace_root=root), EditTool(workspace_root=root),
            DockerBashTool(root, image=sandbox_image, network=sandbox_network)
            if sandbox_bash else BashTool(workspace_root=root),
            GrepTool(workspace_root=root), GlobTool(workspace_root=root),
            WebSearchTool(), WebFetchTool(), TodoWriteTool(),
            TaskCreateTool(task_store), TaskUpdateTool(task_store), TaskListTool(task_store),
            SkillTool(skill_system), RecallMemoryTool(memory_manager), RememberTool(memory_manager),
        ]
    }
```

- [x] **Step 4: Update Plan-mode whitelist**

In `capabilities/security.py`:

```python
PLAN_MODE_EXTRA_TOOLS = {"TodoWrite", "TaskCreate", "TaskList"}
```

Do not add `TaskUpdate`; plan mode creates and views plans but cannot claim execution.

- [x] **Step 5: Run wiring tests**

Run: `python -m pytest tests/test_wiring.py -q`
Expected: PASS.

- [x] **Step 6: Commit**

```bash
git add main.py capabilities/security.py tests/test_wiring.py
git commit -m "feat: wire task graph into main and subagents"
```

### Task 4: Prompt Injection and Corrupt-Graph Recovery

**Files:**
- Modify: `prompt/system_prompt.py`
- Modify: `main.py`
- Test: `tests/test_task_graph.py`

- [x] **Step 1: Write prompt tests**

```python
def test_prompt_injects_open_tasks_only(tmp_path):
    store = TaskGraphStore(tmp_path)
    open_task = store.create("未完成")
    done = store.create("已完成")
    store.update(done["id"], "in_progress")
    store.update(done["id"], "completed")

    hint = store.prompt_hint()
    assert open_task["id"] in hint
    assert done["id"] not in hint


def test_prompt_reports_corrupt_graph_without_crashing(tmp_path, monkeypatch):
    (tmp_path / ".minicode").mkdir()
    (tmp_path / ".minicode" / "tasks.json").write_text("{broken", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    tools, prompt, _, memory_manager, factory = _unpack("normal")
    try:
        assert "Task Graph Unavailable" in prompt
        assert "Task Graph Unavailable" in factory("new task")
    finally:
        memory_manager.close()
```

- [x] **Step 2: Run tests and verify they fail**

Run: `python -m pytest tests/test_task_graph.py -q`
Expected: `prompt_hint` is absent or prompt does not include the dynamic section.

- [x] **Step 3: Add prompt parameter and renderer**

```python
def build_system_prompt(..., memory_hint: str = "", task_graph_hint: str = "") -> str:
    ...
    if task_graph_hint:
        dynamic_parts.append(task_graph_hint)
    if memory_hint:
        dynamic_parts.append(memory_hint)
    if routing_hint:
        dynamic_parts.append(routing_hint)
```

Place the task graph before memory and routing so the most task-specific routing block remains last.

- [x] **Step 4: Build a safe hint in main**

```python
task_graph_warning_emitted = False

def task_graph_hint() -> str:
    nonlocal task_graph_warning_emitted
    try:
        return task_store.prompt_hint()
    except TaskGraphError as exc:
        if not task_graph_warning_emitted:
            print(f"[Task graph unavailable: {exc}]")
            task_graph_warning_emitted = True
        return f"## Task Graph Unavailable\n{exc}"
```

Pass `task_graph_hint=task_graph_hint()` into `build_system_prompt()`.

- [x] **Step 5: Run prompt tests**

Run: `python -m pytest tests/test_task_graph.py tests/test_wiring.py -q`
Expected: PASS.

- [x] **Step 6: Commit**

```bash
git add main.py prompt/system_prompt.py tests/test_task_graph.py
git commit -m "feat: inject and recover task graph context"
```

### Task 5: Compression Anchor

**Files:**
- Modify: `core/agent_loop.py`
- Modify: `tests/test_task_anchor.py`

- [x] **Step 1: Write the anchor test**

```python
@pytest.mark.asyncio
async def test_task_graph_tool_result_becomes_compression_anchor(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from capabilities.task_graph import TaskCreateTool, TaskGraphStore

    model = _MockModel([[
        _Chunk(type="tool_use_start", name="TaskCreate", input={"subject": "实现压缩锚点"}, tool_call_id="t1"),
        _Chunk(type="message_stop", stop_reason="tool_use", usage=_Usage()),
    ], [
        _Chunk(type="text_delta", text="done"),
        _Chunk(type="message_stop", stop_reason="end_turn", usage=_Usage()),
    ]])
    loop = AgentLoop(tools=[TaskCreateTool(TaskGraphStore(tmp_path))], model_adapter=model, system_prompt="test")
    _ = [event async for event in loop.run("go")]
    assert "实现压缩锚点" in loop._compressor._task_list
```

- [x] **Step 2: Run test and verify it fails**

Run: `python -m pytest tests/test_task_anchor.py::test_task_graph_tool_result_becomes_compression_anchor -q`
Expected: anchor remains empty.

- [x] **Step 3: Register graph snapshots**

```python
TASK_GRAPH_TOOLS = {"TaskCreate", "TaskUpdate", "TaskList"}

if tool_name == "TodoWrite" and result.startswith("## Task List"):
    self._compressor.record_task_list(result)
elif tool_name in TASK_GRAPH_TOOLS and "## Task Graph" in result:
    self._compressor.record_task_list(result)
```

- [x] **Step 4: Run all compression-anchor tests**

Run: `python -m pytest tests/test_task_anchor.py -q`
Expected: PASS; existing TodoWrite behavior remains green.

- [x] **Step 5: Commit**

```bash
git add core/agent_loop.py tests/test_task_anchor.py
git commit -m "feat: preserve task graph across compression"
```

### Task 6: Documentation and Full Verification

**Files:**
- Modify: `README.md`
- Modify: `DESIGN.md`
- Modify: `ROADMAP.md`

- [x] **Step 1: Run the full suite before documentation numbers**

Run: `python -m pytest -q`
Expected: all tests pass; record the exact passed/skipped count for documentation.

- [x] **Step 2: Document verified behavior**

Update README:

- tool count and the three task tools;
- `.minicode/tasks.json` location;
- dependency and state rules;
- Plan mode contains create/list but not update.

Update DESIGN with the data model, state machine, compressed anchor, corrupt-file behavior, and single-process limitation. Update ROADMAP from missing persistence to implemented persistence while retaining “no claims/leases/multi-process scheduler” as future work.

- [x] **Step 3: Run all verification**

Run:

```bash
python -m pytest -q
python benchmarks/verify_claims.py
python -m compileall -q .
python -m pyflakes .
git diff --check
```

Expected: every command exits 0 and the documented test count matches pytest.

- [x] **Step 4: Commit documentation**

```bash
git add README.md DESIGN.md ROADMAP.md
git commit -m "docs: document persistent task graph"
```

## Final Acceptance Checklist

- [x] `TaskGraphStore` persists tasks under `.minicode/tasks.json`.
- [x] Dependencies and status transitions are enforced before writes.
- [x] Corrupt files produce a warning and are never overwritten.
- [x] `TaskCreate / TaskUpdate / TaskList` are registered in Normal mode.
- [x] Plan mode registers only `TaskCreate / TaskList` from the new tools.
- [x] Main and child agents share one store instance.
- [x] Prompt injection contains open tasks only.
- [x] All three graph tools register successful snapshots as compression anchors.
- [x] Existing TodoWrite behavior is unchanged.
- [x] Full tests, claim verification, compileall, pyflakes, and diff checks pass.
