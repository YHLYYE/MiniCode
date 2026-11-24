from pathlib import Path

from benchmarks.coding_task_set import CODING_TASKS, materialize_task
from benchmarks.real_coding_eval import (
    _evaluation_workspace,
    aggregate_runs,
    missing_credentials,
    run_single_task,
    select_tasks,
)


def _run(task_id, category, passed, claimed=True, turns=2, tokens=100, cost=0.01, duration=1.5):
    return {
        "task_id": task_id,
        "category": category,
        "passed": passed,
        "claimed_completion": claimed,
        "false_completion": claimed and not passed,
        "turns": turns,
        "total_tokens": tokens,
        "cost_usd": cost,
        "duration_seconds": duration,
        "tool_calls": 3,
        "tool_errors": 0,
    }


def test_aggregate_runs_uses_raw_numerator_and_denominator():
    summary = aggregate_runs([
        _run("a", "parsing", True, turns=1, tokens=50, cost=0.01, duration=1.0),
        _run("b", "parsing", False, turns=3, tokens=150, cost=0.03, duration=2.0),
    ])

    assert summary["attempted"] == 2
    assert summary["passed"] == 1
    assert summary["pass_rate"] == 0.5
    assert summary["false_completions"] == 1
    assert summary["false_completion_rate"] == 0.5
    assert summary["mean_turns"] == 2.0
    assert summary["median_tokens"] == 100
    assert summary["by_category"]["parsing"] == {
        "attempted": 2,
        "passed": 1,
        "false_completions": 1,
    }


def test_aggregate_empty_runs_does_not_invent_metrics():
    summary = aggregate_runs([])

    assert summary["attempted"] == 0
    assert summary["pass_rate"] is None
    assert summary["false_completion_rate"] is None
    assert summary["mean_turns"] is None


def test_hidden_test_is_injected_only_after_agent_finishes(tmp_path):
    task = CODING_TASKS[0]

    def fake_agent(root):
        assert not (root / "tests" / "test_hidden_acceptance.py").exists()
        materialize_task(task, root, solution=True)
        (root / ".minicode").mkdir(exist_ok=True)
        (root / ".minicode" / "state.json").write_text("{}", encoding="utf-8")
        (root / "__pycache__").mkdir(exist_ok=True)
        (root / "__pycache__" / "math_utils.pyc").write_bytes(b"cache")
        return {
            "claimed_completion": True,
            "termination_reason": "completed",
            "turns": 1,
            "input_tokens": 10,
            "output_tokens": 5,
            "total_tokens": 15,
            "cost_usd": 0.0,
            "duration_seconds": 0.2,
            "model_calls": 1,
            "tool_calls": 1,
            "tool_errors": 0,
            "tool_names": ["Edit"],
            "answer": "fixed",
        }

    run = run_single_task(task, fake_agent, workspace_root=tmp_path)

    assert run["passed"] is True
    assert run["claimed_completion"] is True
    assert run["false_completion"] is False
    assert "math_utils.py" in run["patch"]
    assert run["changed_files"] == ["math_utils.py"]
    assert "2 passed" in run["hidden_test_output"]


def test_select_tasks_filters_by_id_category_and_limit():
    assert [task.id for task in select_tasks(task_id="clamp_bounds")] == ["clamp_bounds"]
    parsing = select_tasks(category="parsing")
    assert parsing and all(task.category == "parsing" for task in parsing)
    assert len(select_tasks(limit=3)) == 3


def test_missing_credentials_is_explicit_for_supported_provider():
    assert missing_credentials("deepseek-chat", env={}) == ["DEEPSEEK_API_KEY"]
    assert missing_credentials("claude-sonnet-4-6", env={"ANTHROPIC_API_KEY": "x"}) == []
    assert missing_credentials("gpt-4o-mini", env={}) == ["OPENAI_API_KEY"]


def test_evaluation_workspace_scopes_auto_approval(tmp_path, monkeypatch):
    monkeypatch.delenv("MINICODE_AUTO_APPROVE", raising=False)

    with _evaluation_workspace(tmp_path):
        assert Path.cwd() == tmp_path
        assert __import__("os").environ["MINICODE_AUTO_APPROVE"] == "1"

    assert "MINICODE_AUTO_APPROVE" not in __import__("os").environ
