import json

from benchmarks.verification_worktree_bench import run_benchmark


def test_verification_worktree_benchmark_reports_real_outcomes(tmp_path):
    summary = run_benchmark(tmp_path)
    report_path = tmp_path / "worktree_verification_bench.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert summary["mode"] == "scripted-offline"
    assert summary["cases_total"] == 4
    assert summary["cases_passed"] == 4
    assert summary["false_completion_blocked"] == 1
    assert summary["verified_commits"] == 1
    assert summary["main_workspace_mutations_before_apply"] == 0
    assert summary["conflict_rollbacks"] == 1
    assert summary["auto_merges"] == 0
    assert report["summary"] == summary
    assert len(report["cases"]) == 4
    assert all(case["passed"] for case in report["cases"])
    assert all(
        "unexpected_main_workspace_mutations" in case
        for case in report["cases"]
    )
    conflict = next(
        case for case in report["cases"] if case["case_id"] == "conflict_rollback"
    )
    assert conflict["rollback_ok"] == 1
    assert summary["auto_merges"] == sum(
        case["unexpected_main_workspace_mutations"]
        for case in report["cases"]
    )
