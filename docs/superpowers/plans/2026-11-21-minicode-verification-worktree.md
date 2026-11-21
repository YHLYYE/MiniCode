# MiniCode Verification and Worktree Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an optional external completion verifier and Git Worktree isolation for write-capable sub-agents without changing MiniCode's default behavior.

**Architecture:** A standalone `CommandVerifier` gates natural completion inside `AgentLoop`; a `WorktreeManager` creates safe task branches and `AgentTool` runs write-capable descendants with tools bound to that workspace. Existing CLI, memory, Skill, model adapter, and main workspace flow remain intact unless the new flags are enabled.

**Tech Stack:** Python 3.10+, asyncio, Git worktree, pytest, existing MiniCode Tool/AgentLoop abstractions.

---

## File Map

- Create: `capabilities/verification.py` - command execution and structured verification result.
- Create: `tests/test_verification.py` - verifier behavior.
- Modify: `core/state.py` - verification retry and explicit termination reasons.
- Modify: `core/agent_loop.py` - gate natural completion and retry with evidence.
- Modify: `main.py` - CLI flags and termination display.
- Modify: `core/tools/fs_backend.py` - optional workspace root and safe path resolution.
- Modify: `core/tools/files.py` - backend root injection.
- Modify: `core/tools/edit.py` - backend root injection.
- Modify: `core/tools/search.py` - workspace root defaults and escape checks.
- Modify: `core/tools/shell.py` - explicit working directory.
- Create: `capabilities/worktree.py` - Git repository checks and Worktree lifecycle.
- Create: `tests/test_worktree.py` - Worktree safety and lifecycle.
- Modify: `capabilities/multi_agent.py` - optional Worktree execution for `general` and `team`.
- Modify: `tests/test_multi_agent.py` - isolation integration tests.
- Modify: `.gitignore` only if `.minicode/` is not already ignored.
- Modify after validation: `README.md`, `DESIGN.md`.

---

### Task 1: Command Verifier

**Files:**
- Create: `capabilities/verification.py`
- Create: `tests/test_verification.py`

- [ ] **Step 1: Write failing verifier tests**

```python
import sys

import pytest

from capabilities.verification import CommandVerifier


@pytest.mark.asyncio
async def test_verifier_passes_on_zero_exit():
    verifier = CommandVerifier(
        f'"{sys.executable}" -c "print(\'ok\')"',
        timeout_seconds=5,
    )
    result = await verifier.verify()
    assert result.passed is True
    assert result.exit_code == 0
    assert "ok" in result.output


@pytest.mark.asyncio
async def test_verifier_returns_tail_of_failed_output():
    verifier = CommandVerifier(
        f'"{sys.executable}" -c "print(123); raise SystemExit(2)"',
        timeout_seconds=5,
        max_output_chars=20,
    )
    result = await verifier.verify()
    assert result.passed is False
    assert result.exit_code == 2
    assert len(result.output) <= 20


@pytest.mark.asyncio
async def test_verifier_reports_timeout():
    verifier = CommandVerifier(
        f'"{sys.executable}" -c "import time; time.sleep(2)"',
        timeout_seconds=0.1,
    )
    result = await verifier.verify()
    assert result.passed is False
    assert result.timed_out is True
```

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m pytest tests/test_verification.py -q`

Expected: collection/import failure because `CommandVerifier` does not exist.

- [ ] **Step 3: Implement verifier**

```python
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class VerificationResult:
    command: str
    passed: bool
    exit_code: int | None
    output: str
    timed_out: bool = False


class CommandVerifier:
    def __init__(
        self,
        command: str,
        *,
        cwd: Path | None = None,
        timeout_seconds: float = 120,
        max_output_chars: int = 8_000,
    ):
        if not command.strip():
            raise ValueError("verification command cannot be empty")
        self.command = command
        self.cwd = cwd or Path.cwd()
        self.timeout_seconds = timeout_seconds
        self.max_output_chars = max_output_chars

    async def verify(self) -> VerificationResult:
        process = None
        try:
            process = await asyncio.create_subprocess_shell(
                self.command,
                cwd=str(self.cwd),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=self.timeout_seconds
            )
        except asyncio.TimeoutError:
            if process is not None:
                process.kill()
                await process.wait()
            return VerificationResult(
                command=self.command,
                passed=False,
                exit_code=None,
                output=f"verification timed out after {self.timeout_seconds}s",
                timed_out=True,
            )
        except Exception as exc:
            if process is not None and process.returncode is None:
                process.kill()
                await process.wait()
            return VerificationResult(
                command=self.command,
                passed=False,
                exit_code=None,
                output=f"{type(exc).__name__}: {exc}",
            )

        combined = b"\n".join(part for part in (stdout, stderr) if part)
        decoded = combined.decode("utf-8", errors="replace")
        tail = decoded[-self.max_output_chars :]
        return VerificationResult(
            command=self.command,
            passed=process.returncode == 0,
            exit_code=process.returncode,
            output=tail,
        )
```

- [ ] **Step 4: Run tests and verify GREEN**

Run: `python -m pytest tests/test_verification.py -q`

Expected: `3 passed`.

- [ ] **Step 5: Commit**

```bash
git add capabilities/verification.py tests/test_verification.py
git commit -m "feat: add external command verifier"
```

---

### Task 2: Explicit Termination State and Agent Loop Gate

**Files:**
- Modify: `core/state.py`
- Modify: `core/agent_loop.py`
- Create: `tests/test_verification_loop.py`

- [ ] **Step 1: Write failing loop tests**

```python
import pytest

from core.agent_loop import AgentLoop
from core.state import DoneEvent, TerminationReason
from tests.test_agent_loop import MockModelAdapter, make_stop, make_text


class FailingThenPassingVerifier:
    def __init__(self):
        self.calls = 0

    async def verify(self):
        self.calls += 1
        from capabilities.verification import VerificationResult
        return VerificationResult(
            command="fake",
            passed=self.calls > 1,
            exit_code=1 if self.calls == 1 else 0,
            output="tests failed" if self.calls == 1 else "ok",
        )


@pytest.mark.asyncio
async def test_agent_loop_retries_when_verification_fails():
    verifier = FailingThenPassingVerifier()
    loop = AgentLoop(
        tools=[],
        model_adapter=MockModelAdapter([
            [make_text("done"), make_stop()],
            [make_text("fixed"), make_stop()],
        ]),
        system_prompt="Test.",
        verifier=verifier,
        verify_retries=2,
    )
    events = [event async for event in loop.run("fix tests")]
    done = next(event for event in events if isinstance(event, DoneEvent))
    assert verifier.calls == 2
    assert done.state.termination_reason == TerminationReason.COMPLETED


@pytest.mark.asyncio
async def test_agent_loop_exhausted_verification_is_not_success():
    class AlwaysFail:
        async def verify(self):
            from capabilities.verification import VerificationResult
            return VerificationResult("fake", False, 1, "still broken")

    loop = AgentLoop(
        tools=[],
        model_adapter=MockModelAdapter([
            [make_text("done"), make_stop()],
            [make_text("done again"), make_stop()],
        ]),
        system_prompt="Test.",
        verifier=AlwaysFail(),
        verify_retries=1,
    )
    events = [event async for event in loop.run("fix tests")]
    done = next(event for event in events if isinstance(event, DoneEvent))
    assert done.state.termination_reason == TerminationReason.VERIFY_FAILED
```

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m pytest tests/test_verification_loop.py -q`

Expected: import failure for `TerminationReason` and `AgentLoop` does not accept `verifier`.

- [ ] **Step 3: Add state types**

```python
class TerminationReason(Enum):
    COMPLETED = "completed"
    VERIFY_FAILED = "verify_failed"
    MAX_TURNS = "max_turns"
    EMPTY_RESPONSE = "empty_response"
    MAX_OUTPUT_TOKENS = "max_output_tokens"


class ContinueReason(Enum):
    # existing members...
    VERIFICATION_RETRY = "verification_retry"
```

Add to `LoopState`:

```python
termination_reason: TerminationReason | None = None
```

- [ ] **Step 4: Integrate verifier into AgentLoop**

Add constructor parameters:

```python
verifier=None,
verify_retries: int = 2,
```

Store:

```python
self._verifier = verifier
self._verify_retries = max(0, verify_retries)
self._verification_failures = 0
```

Reset `_verification_failures = 0` at the start of `run()`.

Replace the no-tool completion branch:

```python
if not tool_calls and assistant_text:
    if self._verifier is not None:
        verification = await self._verifier.verify()
        if not verification.passed:
            if self._verification_failures < self._verify_retries:
                self._verification_failures += 1
                evidence = (
                    "[External verification failed]\n"
                    f"command: {verification.command}\n"
                    f"exit_code: {verification.exit_code}\n"
                    f"timed_out: {verification.timed_out}\n"
                    f"evidence:\n{verification.output}\n"
                    "请修复未通过的验收，不要只重复声称完成。"
                )
                self._state = self._state.add_message(
                    Message(role="user", content=evidence)
                )
                self._state = self._state.with_transition(
                    ContinueReason.VERIFICATION_RETRY
                )
                yield RecoveryNotice(
                    ContinueReason.VERIFICATION_RETRY,
                    "外部验收失败，已将证据回灌模型",
                )
                continue
            self._state = self._state.with_field(
                termination_reason=TerminationReason.VERIFY_FAILED
            )
            await self._record_task_done("terminated: verification failed")
            yield DoneEvent(self._state)
            return
    self._state = self._state.with_field(
        transition=None,
        termination_reason=TerminationReason.COMPLETED,
    )
    await self._record_task_done(assistant_text[:300])
    yield DoneEvent(self._state)
    return
```

Set the other termination exits to their matching `TerminationReason`.

- [ ] **Step 5: Run focused tests**

Run: `python -m pytest tests/test_verification_loop.py tests/test_agent_loop.py -q`

Expected: all pass; existing tests still pass.

- [ ] **Step 6: Commit**

```bash
git add core/state.py core/agent_loop.py tests/test_verification_loop.py
git commit -m "feat: gate completion with external verification"
```

---

### Task 3: CLI Wiring and Termination Display

**Files:**
- Modify: `main.py`
- Create: `tests/test_cli_verification.py`

- [ ] **Step 1: Write failing wiring test**

```python
from click.testing import CliRunner

from main import main


def test_help_lists_verification_and_isolation_flags():
    result = CliRunner().invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "--verify-command" in result.output
    assert "--verify-timeout" in result.output
    assert "--verify-retries" in result.output
    assert "--isolate-subagents" in result.output
```

- [ ] **Step 2: Run test and verify RED**

Run: `python -m pytest tests/test_cli_verification.py -q`

Expected: FAIL because flags are absent.

- [ ] **Step 3: Add Click options and construct verifier**

```python
@click.option("--verify-command", default=None)
@click.option("--verify-timeout", default=120, type=int)
@click.option("--verify-retries", default=2, type=int)
@click.option("--isolate-subagents", is_flag=True)
```

Change `main` signature to receive those values. Before `_make_loop`:

```python
verifier = None
if verify_command:
    verifier = CommandVerifier(
        verify_command,
        cwd=Path.cwd(),
        timeout_seconds=verify_timeout,
    )
```

Pass `verifier=verifier` and `verify_retries=verify_retries` through `_make_loop` and `_interactive_repl`.

In `DoneEvent`, print termination reason:

```python
reason = getattr(s, "termination_reason", None)
reason_label = reason.value if reason else "completed"
print(f"termination: {reason_label}")
```

- [ ] **Step 4: Run CLI and loop tests**

Run: `python -m pytest tests/test_cli_verification.py tests/test_verification_loop.py -q`

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add main.py tests/test_cli_verification.py
git commit -m "feat: add verification CLI options"
```

---

### Task 4: Workspace-Scoped Tool Instances

**Files:**
- Modify: `core/tools/fs_backend.py`
- Modify: `core/tools/files.py`
- Modify: `core/tools/edit.py`
- Modify: `core/tools/search.py`
- Modify: `core/tools/shell.py`
- Create: `tests/test_workspace_tools.py`

- [ ] **Step 1: Write failing root-confinement tests**

```python
from pathlib import Path
import pytest

from core.tools.files import ReadTool, WriteTool
from core.tools.shell import BashTool


@pytest.mark.asyncio
async def test_workspace_write_maps_relative_path(tmp_path):
    tool = WriteTool(workspace_root=tmp_path)
    result = await tool.execute(file_path="src/a.txt", content="ok")
    assert result.startswith("Created")
    assert (tmp_path / "src/a.txt").read_text() == "ok"


@pytest.mark.asyncio
async def test_workspace_write_rejects_absolute_escape(tmp_path):
    tool = WriteTool(workspace_root=tmp_path)
    outside = tmp_path.parent / "escape.txt"
    result = await tool.execute(file_path=str(outside), content="bad")
    assert "outside workspace" in result
    assert not outside.exists()


@pytest.mark.asyncio
async def test_workspace_shell_uses_explicit_cwd(tmp_path):
    tool = BashTool(workspace_root=tmp_path)
    result = await tool.execute(command="pwd")
    assert str(tmp_path).lower() in result.lower()
```

- [ ] **Step 2: Run and verify RED**

Run: `python -m pytest tests/test_workspace_tools.py -q`

Expected: constructors reject `workspace_root`.

- [ ] **Step 3: Add root-aware backend**

```python
class LocalFilesystemBackend(FilesystemBackend):
    def __init__(self, root: Path | None = None):
        self.root = root.resolve() if root else None

    def resolve(self, path: str) -> Path:
        candidate = Path(path)
        if self.root and not candidate.is_absolute():
            candidate = self.root / candidate
        candidate = candidate.resolve()
        if self.root and not candidate.is_relative_to(self.root):
            raise PermissionError(f"path outside workspace: {path}")
        return candidate
```

Use `resolve()` in read/write/exists/is_dir.

- [ ] **Step 4: Wire tool constructors**

```python
class WriteTool(Tool):
    def __init__(self, backend=None, workspace_root: Path | None = None):
        super().__init__()
        self._backend = backend or LocalFilesystemBackend(workspace_root)
```

Apply the same pattern to Read and Edit.

For Grep/Glob:

```python
def __init__(self, workspace_root: Path | None = None):
    super().__init__()
    self._workspace_root = workspace_root.resolve() if workspace_root else None

def _root(self, path: str) -> Path:
    root = self._workspace_root or Path.cwd()
    candidate = (Path(path) if Path(path).is_absolute() else root / path).resolve()
    if self._workspace_root and not candidate.is_relative_to(self._workspace_root):
        raise PermissionError(f"path outside workspace: {path}")
    return candidate
```

For Bash:

```python
def __init__(self, workspace_root: Path | None = None):
    super().__init__()
    self._workspace_root = workspace_root.resolve() if workspace_root else Path.cwd().resolve()

async def execute(...):
    cwd = self._workspace_root
```

- [ ] **Step 5: Run tests**

Run: `python -m pytest tests/test_workspace_tools.py tests/test_fs_backend.py tests/test_security.py -q`

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add core/tools/fs_backend.py core/tools/files.py core/tools/edit.py core/tools/search.py core/tools/shell.py tests/test_workspace_tools.py
git commit -m "feat: add workspace-scoped tool instances"
```

---

### Task 5: Worktree Manager

**Files:**
- Create: `capabilities/worktree.py`
- Create: `tests/test_worktree.py`

- [ ] **Step 1: Write failing Worktree tests**

```python
import subprocess
from pathlib import Path

import pytest

from capabilities.worktree import WorktreeManager


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()


@pytest.fixture
def repo(tmp_path):
    git(tmp_path, "init")
    git(tmp_path, "config", "user.email", "test@example.com")
    git(tmp_path, "config", "user.name", "Test")
    (tmp_path / "a.txt").write_text("a")
    git(tmp_path, "add", ".")
    git(tmp_path, "commit", "-m", "init")
    return tmp_path


def test_rejects_dirty_repo(repo):
    (repo / "a.txt").write_text("dirty")
    manager = WorktreeManager(repo)
    with pytest.raises(RuntimeError, match="clean"):
        manager.create("task")


def test_creates_and_commits_isolated_branch(repo):
    manager = WorktreeManager(repo)
    handle = manager.create("实现登录修复")
    assert handle.path.is_dir()
    (handle.path / "b.txt").write_text("b")
    commit = manager.commit(handle, "subagent changes")
    assert commit
    assert handle.branch in git(repo, "branch", "--list", handle.branch)
    assert (repo / "a.txt").read_text() == "a"


def test_task_text_cannot_escape_worktree_root(repo):
    manager = WorktreeManager(repo)
    handle = manager.create("../../escape")
    assert handle.path.parent == manager.worktrees_root.resolve()


def test_rejects_empty_task(repo):
    manager = WorktreeManager(repo)
    with pytest.raises(ValueError):
        manager.create("   ")
```

- [ ] **Step 2: Run and verify RED**

Run: `python -m pytest tests/test_worktree.py -q`

Expected: import failure for `WorktreeManager`.

- [ ] **Step 3: Implement manager**

```python
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
        return subprocess.check_output(
            ["git", *args], cwd=cwd or self.repo_root,
            text=True, stderr=subprocess.STDOUT,
        ).strip()

    def _ensure_ready(self) -> None:
        self._git("rev-parse", "--show-toplevel")
        if self._git("status", "--porcelain"):
            raise RuntimeError("working tree must be clean before creating a worktree")

    def create(self, task: str) -> WorktreeHandle:
        if not task.strip():
            raise ValueError("task name cannot be empty")
        slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", task).strip("-").lower() or "task"
        if slug in {".", ".."} or "/" in slug or "\\" in slug:
            raise ValueError("unsafe worktree name")
        self._ensure_ready()
        task_id = hashlib.sha256(task.encode("utf-8")).hexdigest()[:8]
        slug = slug[:32]
        branch = f"minicode/{slug}-{task_id}"
        path = self.worktrees_root / f"{slug}-{task_id}"
        base = self._git("rev-parse", "HEAD")
        path.parent.mkdir(parents=True, exist_ok=True)
        self._git("worktree", "add", "-b", branch, str(path), base)
        return WorktreeHandle(slug, task_id, branch, path, base)

    def commit(self, handle: WorktreeHandle, message: str) -> str | None:
        self._git("add", "-A", cwd=handle.path)
        if subprocess.run(
            ["git", "diff", "--cached", "--quiet"], cwd=handle.path
        ).returncode == 0:
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

    def remove(self, handle: WorktreeHandle) -> None:
        self._git("worktree", "remove", "--force", str(handle.path))
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_worktree.py -q`

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add capabilities/worktree.py tests/test_worktree.py
git commit -m "feat: add safe task worktree manager"
```

---

### Task 6: AgentTool Worktree Integration

**Files:**
- Modify: `capabilities/multi_agent.py`
- Modify: `main.py`
- Modify: `tests/test_multi_agent.py`

- [ ] **Step 1: Write failing isolation tests**

```python
@pytest.mark.asyncio
async def test_general_agent_runs_in_worktree_when_enabled(repo):
    seen = {}

    def factory(tools, system_prompt, max_turns):
        seen["tools"] = {tool.name: tool for tool in tools}

        class FakeLoop:
            async def run(self, task):
                from core.state import TextDelta, DoneEvent, LoopState
                result = await seen["tools"]["Write"].execute(
                    file_path="generated.txt", content="isolated"
                )
                seen["write_result"] = result
                yield TextDelta("wrote generated.txt")
                yield DoneEvent(LoopState(turn_count=1))

        return FakeLoop()

    manager = WorktreeManager(repo)
    agent_tool = AgentTool(
        agent_loop_factory=factory,
        tool_registry={},
        worktree_manager=manager,
        workspace_tool_factory=lambda root: {
            "Write": WriteTool(workspace_root=root)
        },
        isolate_subagents=True,
    )
    result = await agent_tool.execute("write generated file", agent_type="general")
    assert "branch: minicode/" in result
    assert not (repo / "generated.txt").exists()


@pytest.mark.asyncio
async def test_explore_never_uses_worktree(repo):
    manager = WorktreeManager(repo)
    agent_tool = AgentTool(
        agent_loop_factory=_pipeline_factory([]),
        tool_registry={},
        worktree_manager=manager,
        isolate_subagents=True,
    )
    await agent_tool.execute("inspect repo", agent_type="explore")
    assert list((repo / ".minicode" / "worktrees").glob("*")) == []
```

- [ ] **Step 2: Run and verify RED**

Run: `python -m pytest tests/test_multi_agent.py -q`

Expected: `AgentTool` rejects new keyword arguments.

- [ ] **Step 3: Add optional manager/tool factory/config**

```python
def __init__(
    self,
    agent_loop_factory=None,
    tool_registry=None,
    worktree_manager=None,
    workspace_tool_factory=None,
    isolate_subagents: bool = False,
):
    # existing assignments...
    self._worktree_manager = worktree_manager
    self._workspace_tool_factory = workspace_tool_factory
    self._isolate_subagents = isolate_subagents
```

- [ ] **Step 4: Implement general isolation**

In `_run_subagent`, accept optional `workspace_tools`:

```python
async def _run_subagent(self, task: str, profile: dict, tools: list | None = None) -> dict:
    if tools is None:
        tools = self._resolve_tools(profile)
```

In `execute`, for `general` when enabled:

```python
handle = self._worktree_manager.create(task)
tools = list(self._workspace_tool_factory(handle.path).values())
result = await self._run_subagent(task, profile, tools=tools)
commit = self._worktree_manager.commit(handle, f"minicode subagent: {task[:60]}")
summary = self._worktree_manager.diff_stat(handle)
changed = self._worktree_manager.changed_files(handle)
return (
    f"[Sub-agent: general, worktree]\n"
    f"branch: {handle.branch}\n"
    f"commit: {commit or 'none'}\n"
    f"path: {handle.path}\n"
    f"changed_files: {changed}\n"
    f"diff:\n{summary}\n\n{result['output']}"
)
```

Do not delete the Worktree on failure.

- [ ] **Step 5: Implement team isolation**

`research` continues with parent registry. For `coding` and `testing`, create one Worktree before coding and keep the same tool registry for both roles:

```python
worktree = self._worktree_manager.create(task)
isolated_registry = self._workspace_tool_factory(worktree.path)
coding_profile = self._team_profile("coding")
testing_profile = self._team_profile("testing")
coding = await self._run_subagent(task, coding_profile, isolated_registry)
step_task = f"{task}\n\n--- coding output ---\n{coding['output']}"
testing = await self._run_subagent(step_task, testing_profile, isolated_registry)
```

Pass role-scoped subsets from `isolated_registry` into each call. If coding or testing fails, do not commit. If both pass, commit once to `worktree.branch`. Return branch, commit, changed files, diff stat and both role summaries.

- [ ] **Step 6: Wire CLI flag**

When `--isolate-subagents` is true in `main._build_tools`, construct `WorktreeManager(Path.cwd())`, supply a `workspace_tool_factory`, and set `isolate_subagents=True` on `AgentTool`.

- [ ] **Step 7: Run tests**

Run: `python -m pytest tests/test_multi_agent.py tests/test_worktree.py -q`

Expected: all pass.

- [ ] **Step 8: Commit**

```bash
git add capabilities/multi_agent.py main.py tests/test_multi_agent.py
git commit -m "feat: isolate write subagents with worktrees"
```

---

### Task 7: Full Regression and Documentation

**Files:**
- Modify: `README.md`
- Modify: `DESIGN.md`
- Test: all existing tests

- [ ] **Step 1: Run full suite**

Run: `python -m pytest -q`

Expected: at least `251 passed, 1 skipped`, plus all new tests; zero failures.

- [ ] **Step 2: Run compile check**

Run: `python -m compileall -q .`

Expected: exit code 0 and no output.

- [ ] **Step 3: Verify default behavior**

Run: `python -m pytest tests/test_agent_loop.py tests/test_multi_agent.py -q`

Expected: existing tests pass without new flags.

- [ ] **Step 4: Update README**

Document exact usage:

```shell
python main.py "修复测试失败" --verify-command "python -m pytest -q" --verify-retries 2
python main.py "实现小功能" --isolate-subagents
```

Document that:

- Worktree isolation requires a clean Git working tree;
- Worktree changes are not auto-merged;
- the verifier executes only user-supplied commands;
- default behavior is unchanged.

- [ ] **Step 5: Update DESIGN**

Add concise sections for:

- external verification gate;
- write-capable sub-agent Worktree isolation;
- non-goals: no automatic merge, no MCP/Cron/Workflow merge.

- [ ] **Step 6: Commit**

```bash
git add README.md DESIGN.md
git commit -m "docs: document verification and worktree isolation"
```

---

## Final Acceptance Matrix

| Requirement | Evidence |
|---|---|
| No flags means old behavior | existing AgentLoop and AgentTool tests pass |
| Failed verification cannot finish | `test_agent_loop_retries_when_verification_fails` |
| Exhausted verification returns failure | `test_agent_loop_exhausted_verification_is_not_success` |
| CLI exposes controls | `test_help_lists_verification_and_isolation_flags` |
| Tools cannot escape workspace | `test_workspace_write_rejects_absolute_escape` |
| Dirty/non-Git repos fail closed | `test_rejects_dirty_repo` and repository checks |
| General write subagent is isolated | `test_general_agent_runs_in_worktree_when_enabled` |
| Explore never creates Worktree | `test_explore_never_uses_worktree` |
| Team coding/testing share one Worktree | team isolation tests |
| No auto-merge | Worktree result contains branch/commit only |

## Self-Review

- Spec coverage: every requirement in the design maps to Tasks 1-7.
- Placeholder scan: no TODO, TBD, or unspecified implementation step remains.
- Type consistency: `VerificationResult`, `TerminationReason`, `WorktreeHandle`, `isolate_subagents`, and `workspace_tool_factory` use one spelling throughout.
- Scope check: verification and Worktree isolation are independently testable and share only the AgentTool composition point.
