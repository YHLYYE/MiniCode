# Real Coding Task Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add 20 real coding tasks and a reproducible runner that uses actual model calls and hidden tests.

**Architecture:** Task fixtures live separately from execution. The runner materializes a clean Git repository, executes MiniCode with evaluation hooks, then applies hidden tests and records raw evidence.

**Tech Stack:** Python 3.11+, asyncio, Git, pytest, MiniCode `AgentLoop`

---

### Task 1: Task model and fixture validation

**Files:**
- Create: `benchmarks/coding_task_set.py`
- Create: `tests/test_coding_task_set.py`

- [ ] Write tests that materialize a task, prove the initial repository fails its hidden test, apply the reference solution and prove it passes.
- [ ] Run the tests and confirm failure because the task model does not exist.
- [ ] Implement immutable `CodingTask` definitions and the 20-task suite.
- [ ] Add a shared helper that creates a task workspace and runs its hidden test from a caller-provided file map.
- [ ] Run fixture tests until all 20 tasks satisfy red-baseline and green-solution checks.

### Task 2: Machine-readable run aggregation

**Files:**
- Create: `benchmarks/real_coding_eval.py`
- Create: `tests/test_real_coding_eval.py`

- [ ] Write tests for pass rate, false completion, grouped metrics and empty-run behavior.
- [ ] Run them and confirm failure because aggregation functions are missing.
- [ ] Implement `aggregate_runs()` and report rendering with raw numerator/denominator values.
- [ ] Rerun focused tests.

### Task 3: Isolated workspace execution

**Files:**
- Modify: `benchmarks/real_coding_eval.py`
- Modify: `tests/test_real_coding_eval.py`

- [ ] Write a failing test proving hidden tests do not exist in the workspace during the agent callback and do exist during acceptance.
- [ ] Implement workspace creation, Git baseline commit, hidden-test injection and acceptance execution.
- [ ] Capture the final patch and failure output even when acceptance fails.
- [ ] Rerun focused tests.

### Task 4: Real model runner and hooks

**Files:**
- Modify: `benchmarks/real_coding_eval.py`
- Modify: `tests/test_real_coding_eval.py`

- [ ] Write a fake-loop integration test that records completion claims, tool errors, turns and tokens without using a real API.
- [ ] Implement session construction through `main._build_tools` and `main._make_loop`, using `AFTER_MODEL`, `AFTER_TOOL` and `STOP` hooks for evidence.
- [ ] Ensure missing credentials fail before creating a report.
- [ ] Add CLI options for task ID, category, limit, repeats, output directory, model, turns and cost.

### Task 5: Documentation and verification

**Files:**
- Modify: `README.md`
- Modify: `DESIGN.md`
- Modify: `CONTRIBUTING.md`
- Modify: `ROADMAP.md`

- [ ] Document how to validate fixtures, run a real evaluation and interpret metrics.
- [ ] Run `python -m benchmarks.real_coding_eval --validate-fixtures`.
- [ ] Run a one-task smoke evaluation if credentials are available.
- [ ] Run the full test suite and update quoted test counts.
