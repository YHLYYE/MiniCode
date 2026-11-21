import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from capabilities.worktree import (
    WorktreeApplyTool,
    WorktreeListTool,
    WorktreeManager,
    WorktreeRemoveTool,
)


def git(repo, *args):
    return subprocess.check_output(
        ["git", *args], cwd=repo, text=True, stderr=subprocess.STDOUT
    ).strip()


@pytest.fixture
def repo(tmp_path):
    git(tmp_path, "init")
    git(tmp_path, "config", "user.email", "test@example.com")
    git(tmp_path, "config", "user.name", "Test")
    (tmp_path / "a.txt").write_text("a")
    git(tmp_path, "add", ".")
    git(tmp_path, "commit", "-m", "init")
    return tmp_path


def commit_change(manager, handle):
    (handle.path / "feature.txt").write_text("feature")
    commit = manager.commit(
        handle,
        "verified feature",
        verification_status="passed",
    )
    manager.mark_verification(handle.branch, "passed", "tests passed")
    return commit


def test_index_tracks_created_and_verified_worktree(repo):
    manager = WorktreeManager(repo)
    handle = manager.create("feature")

    active = manager.find(handle.branch)
    assert active["status"] == "active"
    assert active["verification"] == "pending"

    commit = commit_change(manager, handle)
    ready = manager.find(handle.branch)

    assert commit
    assert ready["status"] == "ready"
    assert ready["verification"] == "passed"
    assert ready["commit"] == commit
    assert ready["changed_files"] == ["feature.txt"]


def test_apply_requires_verified_ready_worktree(repo):
    manager = WorktreeManager(repo)
    handle = manager.create("feature")
    (handle.path / "feature.txt").write_text("feature")
    manager.commit(handle, "unverified feature")

    with pytest.raises(RuntimeError, match="verified"):
        manager.apply(handle.branch)


def test_apply_cherry_picks_ready_commit(repo):
    manager = WorktreeManager(repo)
    handle = manager.create("feature")
    commit_change(manager, handle)

    result = manager.apply(handle.branch)

    assert (repo / "feature.txt").read_text() == "feature"
    assert manager.find(handle.branch)["status"] == "applied"
    assert result == git(repo, "rev-parse", "HEAD")


def test_remove_rejects_active_but_keeps_branch(repo):
    manager = WorktreeManager(repo)
    handle = manager.create("feature")

    with pytest.raises(RuntimeError, match="active"):
        manager.remove(handle.branch)

    (handle.path / "feature.txt").write_text("feature")
    manager.commit(handle, "verified feature", verification_status="passed")
    manager.mark_verification(handle.branch, "passed", "ok")
    manager.remove(handle.branch)

    assert not handle.path.exists()
    assert handle.branch in git(repo, "branch", "--list", handle.branch)
    assert manager.find(handle.branch)["status"] == "removed"


@pytest.mark.asyncio
async def test_worktree_tools_wrap_manager(repo):
    manager = WorktreeManager(repo)
    handle = manager.create("feature")
    commit_change(manager, handle)

    listed = await WorktreeListTool(manager).execute()
    applied = await WorktreeApplyTool(manager).execute(branch=handle.branch)
    removed = await WorktreeRemoveTool(manager).execute(branch=handle.branch)

    assert handle.branch in listed
    assert "applied" in applied.lower()
    assert "removed" in removed.lower()


def test_apply_and_remove_require_permission():
    manager = WorktreeManager(Path.cwd())
    assert WorktreeApplyTool(manager).is_destructive is True
    assert WorktreeRemoveTool(manager).is_destructive is True
    assert WorktreeListTool(manager).is_readonly is True


def test_corrupt_index_fails_closed(repo):
    manager = WorktreeManager(repo)
    manager.worktrees_root.mkdir(parents=True, exist_ok=True)
    manager.index_path.write_text("{broken", encoding="utf-8")

    with pytest.raises(RuntimeError, match="corrupt"):
        manager.list_entries()

    assert manager.index_path.read_text(encoding="utf-8") == "{broken"


def test_concurrent_index_updates_do_not_lose_entries(repo):
    manager = WorktreeManager(repo)
    first = manager.create("first")
    second = manager.create("second")

    def fail(branch):
        manager.mark_failed(branch, "test failure")

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(fail, [first.branch, second.branch]))

    assert manager.find(first.branch)["status"] == "failed"
    assert manager.find(second.branch)["status"] == "failed"


def test_cherry_pick_conflict_aborts_and_preserves_ready_state(repo):
    manager = WorktreeManager(repo)
    handle = manager.create("conflict")
    (handle.path / "a.txt").write_text("feature")
    manager.commit(handle, "feature change", verification_status="passed")

    (repo / "a.txt").write_text("main")
    git(repo, "add", "a.txt")
    git(repo, "commit", "-m", "main change")

    with pytest.raises(RuntimeError):
        manager.apply(handle.branch)

    assert manager.find(handle.branch)["status"] == "ready"
    assert handle.path.exists()
    assert handle.branch in git(repo, "branch", "--list", handle.branch)
    assert git(repo, "status", "--porcelain") == ""
    assert not (repo / ".git" / "CHERRY_PICK_HEAD").exists()


def test_apply_rejects_existing_git_sequencer_state(repo):
    manager = WorktreeManager(repo)
    handle = manager.create("sequencer")
    (handle.path / "feature.txt").write_text("feature")
    manager.commit(handle, "feature", verification_status="passed")
    (repo / ".git" / "CHERRY_PICK_HEAD").write_text("dummy", encoding="utf-8")

    with pytest.raises(RuntimeError, match="Git operation"):
        manager.apply(handle.branch)
