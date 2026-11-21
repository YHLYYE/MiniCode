import subprocess
from pathlib import Path

import pytest

from capabilities.worktree import WorktreeManager


def git(repo: Path, *args: str) -> str:
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


def test_rejects_dirty_repo(repo):
    (repo / "a.txt").write_text("dirty")
    manager = WorktreeManager(repo)

    with pytest.raises(RuntimeError, match="clean"):
        manager.create("task")


def test_creates_and_commits_isolated_branch(repo):
    manager = WorktreeManager(repo)

    handle = manager.create("实现登录修复")
    assert handle.path.is_dir()
    assert handle.path.parent == manager.worktrees_root.resolve()
    (handle.path / "b.txt").write_text("b")

    commit = manager.commit(handle, "subagent changes")

    assert commit
    assert handle.branch in git(repo, "branch", "--list", handle.branch)
    assert manager.changed_files(handle) == ["b.txt"]
    assert (repo / "a.txt").read_text() == "a"


def test_task_text_cannot_escape_worktree_root(repo):
    manager = WorktreeManager(repo)

    handle = manager.create("../../escape")

    assert handle.path.parent == manager.worktrees_root.resolve()


def test_rejects_empty_task(repo):
    manager = WorktreeManager(repo)

    with pytest.raises(ValueError):
        manager.create("   ")
