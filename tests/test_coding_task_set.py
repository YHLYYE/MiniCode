from concurrent.futures import ThreadPoolExecutor

from benchmarks.coding_task_set import (
    CODING_TASKS,
    run_hidden_test,
    validate_task_fixture,
)


def test_coding_task_suite_has_twenty_unique_tasks():
    assert len(CODING_TASKS) == 20
    assert len({task.id for task in CODING_TASKS}) == 20


def test_every_coding_task_fails_before_and_passes_after_reference_solution():
    failures = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(validate_task_fixture, CODING_TASKS))
    for task, (baseline_ok, solution_ok, detail) in zip(CODING_TASKS, results):
        if baseline_ok or not solution_ok:
            failures.append(f"{task.id}: baseline_ok={baseline_ok}, solution_ok={solution_ok}\n{detail}")

    assert not failures, "invalid task fixtures:\n" + "\n\n".join(failures)


def test_hidden_test_timeout_is_recorded_as_failure(tmp_path):
    result = run_hidden_test(
        tmp_path,
        "def test_never_finishes():\n    while True:\n        pass\n",
        timeout_seconds=0.2,
    )

    assert result.timed_out is True
    assert result.exit_code is None
