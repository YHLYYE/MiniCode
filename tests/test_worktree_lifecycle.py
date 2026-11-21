import subprocess
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
