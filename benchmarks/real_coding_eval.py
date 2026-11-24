"""Runner and metrics for real coding-task evaluation."""
from __future__ import annotations

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import subprocess
import tempfile
import time
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from benchmarks.coding_task_set import (
    CODING_TASKS,
    CodingTask,
    materialize_task,
    run_hidden_test,
    validate_task_fixture,
)

try:
    from benchmarks.coding_task_set import HiddenTestResult
except ImportError:  # pragma: no cover - retained for older fixtures
    HiddenTestResult = object


def _mean(values):
    return statistics.fmean(values) if values else None


def _group_summary(runs: list[dict], key: str) -> dict:
    grouped = defaultdict(list)
    for run in runs:
        grouped[run[key]].append(run)
    return {
        group: {
            "attempted": len(items),
            "passed": sum(bool(item["passed"]) for item in items),
            "false_completions": sum(bool(item["false_completion"]) for item in items),
        }
        for group, items in sorted(grouped.items())
    }


def aggregate_runs(runs: list[dict]) -> dict:
    attempted = len(runs)
    passed = sum(bool(run["passed"]) for run in runs)
    false_completions = sum(bool(run["false_completion"]) for run in runs)
    return {
        "attempted": attempted,
        "passed": passed,
        "pass_rate": passed / attempted if attempted else None,
        "false_completions": false_completions,
        "false_completion_rate": false_completions / attempted if attempted else None,
        "mean_turns": _mean([run["turns"] for run in runs]),
        "median_turns": statistics.median([run["turns"] for run in runs]) if runs else None,
        "mean_tokens": _mean([run["total_tokens"] for run in runs]),
        "median_tokens": statistics.median([run["total_tokens"] for run in runs]) if runs else None,
        "mean_cost_usd": _mean([run["cost_usd"] for run in runs]),
        "mean_duration_seconds": _mean([run["duration_seconds"] for run in runs]),
        "mean_tool_calls": _mean([run["tool_calls"] for run in runs]),
        "mean_tool_errors": _mean([run["tool_errors"] for run in runs]),
        "by_category": _group_summary(runs, "category"),
        "by_task": _group_summary(runs, "task_id"),
    }


def _git(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr or completed.stdout)
    return completed.stdout


def _initialize_repository(root: Path) -> None:
    gitignore = root / ".gitignore"
    if not gitignore.exists():
        gitignore.write_text(
            ".minicode/\n__pycache__/\n*.pyc\n",
            encoding="utf-8",
            newline="\n",
        )
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "minicode-eval@example.local")
    _git(root, "config", "user.name", "MiniCode Eval")
    _git(root, "add", ".")
    _git(root, "commit", "-q", "-m", "task baseline")


def run_single_task(
    task: CodingTask,
    agent_runner,
    *,
    workspace_root: Path | None = None,
    repeat_index: int = 1,
) -> dict:
    started = time.perf_counter()
    temporary = None
    if workspace_root is None:
        temporary = tempfile.TemporaryDirectory(prefix=f"minicode-eval-{task.id}-")
        root = Path(temporary.name)
    else:
        root = Path(workspace_root)

    try:
        materialize_task(task, root)
        _initialize_repository(root)

        try:
            agent_result = agent_runner(root)
        except Exception as exc:
            agent_result = {
                "claimed_completion": False,
                "termination_reason": "agent_error",
                "agent_error": f"{type(exc).__name__}: {exc}",
                "turns": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "cost_usd": 0.0,
                "model_calls": 0,
                "tool_calls": 0,
                "tool_errors": 1,
                "tool_names": [],
                "answer": "",
            }

        patch = _git(root, "diff", "--no-ext-diff", "HEAD")
        changed_output = _git(root, "status", "--porcelain")
        changed_files = [line[3:] for line in changed_output.splitlines() if line.strip()]
        hidden = run_hidden_test(root, task.hidden_test, task.timeout_seconds)
        passed = hidden.exit_code == 0
        claimed = bool(agent_result.get("claimed_completion", False))
        duration = agent_result.get(
            "duration_seconds", time.perf_counter() - started
        )
        return {
            "task_id": task.id,
            "category": task.category,
            "repeat_index": repeat_index,
            "prompt": task.prompt,
            "claimed_completion": claimed,
            "termination_reason": agent_result.get("termination_reason", ""),
            "passed": passed,
            "false_completion": claimed and not passed,
            "hidden_test_exit_code": hidden.exit_code,
            "hidden_test_timed_out": hidden.timed_out,
            "hidden_test_output": f"{hidden.stdout}\n{hidden.stderr}".strip(),
            "turns": agent_result.get("turns", 0),
            "input_tokens": agent_result.get("input_tokens", 0),
            "output_tokens": agent_result.get("output_tokens", 0),
            "total_tokens": agent_result.get("total_tokens", 0),
            "cost_usd": agent_result.get("cost_usd", 0.0),
            "duration_seconds": duration,
            "model_calls": agent_result.get("model_calls", 0),
            "tool_calls": agent_result.get("tool_calls", 0),
            "tool_errors": agent_result.get("tool_errors", 0),
            "tool_names": agent_result.get("tool_names", []),
            "answer": agent_result.get("answer", ""),
            "agent_error": agent_result.get("agent_error", ""),
            "changed_files": changed_files,
            "patch": patch,
        }
    finally:
        if temporary is not None:
            temporary.cleanup()


def select_tasks(
    *,
    task_id: str | None = None,
    category: str | None = None,
    limit: int | None = None,
) -> list[CodingTask]:
    tasks = list(CODING_TASKS)
    if task_id is not None:
        tasks = [task for task in tasks if task.id == task_id]
        if not tasks:
            raise ValueError(f"unknown task id: {task_id}")
    if category is not None:
        tasks = [task for task in tasks if task.category == category]
        if not tasks:
            raise ValueError(f"unknown task category: {category}")
    if limit is not None:
        if limit < 0:
            raise ValueError("limit must be non-negative")
        tasks = tasks[:limit]
    return tasks


def missing_credentials(model: str, env: dict[str, str] | None = None) -> list[str]:
    env = os.environ if env is None else env
    name = model.lower()
    if name.startswith(("deepseek/", "deepseek-")):
        required = ["DEEPSEEK_API_KEY"]
    elif name.startswith(("anthropic/", "claude-")):
        required = ["ANTHROPIC_API_KEY"]
    elif name.startswith(("openai/", "gpt-", "o1", "o3", "o4")):
        required = ["OPENAI_API_KEY"]
    elif name.startswith(("gemini/", "gemini-")):
        required = ["GEMINI_API_KEY", "GOOGLE_API_KEY"]
    elif any(marker in name for marker in ("ollama", "localhost", "127.0.0.1")):
        required = []
    else:
        required = ["MODEL_API_KEY"]

    if len(required) > 1 and any(env.get(key) for key in required):
        return []
    return [key for key in required if not env.get(key)]


@contextmanager
def _working_directory(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


@contextmanager
def _evaluation_workspace(path: Path):
    """Scope evaluator-only permission bypass to the disposable workspace."""
    previous = os.environ.get("MINICODE_AUTO_APPROVE")
    os.environ["MINICODE_AUTO_APPROVE"] = "1"
    try:
        with _working_directory(path):
            yield
    finally:
        if previous is None:
            os.environ.pop("MINICODE_AUTO_APPROVE", None)
        else:
            os.environ["MINICODE_AUTO_APPROVE"] = previous


def _is_tool_error(output: str) -> bool:
    return output.startswith(
        ("Tool error:", "Security blocked:", "Permission denied", "Error: Tool")
    )


def run_agent_in_workspace(root: Path, task: CodingTask, config) -> dict:
    """Run one real MiniCode session against one materialized task."""
    import main
    from core.agent_loop import AgentLoop
    from core.hooks import HookEvent, HookRegistry
    from core.model_adapter import ModelAdapter
    from core.state import TextDelta, DoneEvent, TerminationReason

    hooks = HookRegistry()
    metrics = {
        "model_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "tool_calls": 0,
        "tool_errors": 0,
        "tool_names": [],
    }

    async def record_model(state, text, tool_calls, usage, stop_reason):
        metrics["model_calls"] += 1
        metrics["input_tokens"] += getattr(usage, "input_tokens", 0) or 0
        metrics["output_tokens"] += getattr(usage, "output_tokens", 0) or 0

    async def record_tool(call, tool, output):
        metrics["tool_calls"] += 1
        metrics["tool_names"].append(call.name)
        if _is_tool_error(output):
            metrics["tool_errors"] += 1

    hooks.register(HookEvent.AFTER_MODEL, record_model)
    hooks.register(HookEvent.AFTER_TOOL, record_tool)

    started = time.perf_counter()
    answer_parts: list[str] = []
    done = None
    agent_error = ""
    termination_reason = ""

    with _evaluation_workspace(root):
        tools, system_prompt, _, memory_manager, prompt_factory = main._build_tools(
            "normal",
            config,
            inject_memory=False,
        )
        loop = AgentLoop(
            tools=tools,
            model_adapter=ModelAdapter(config.model),
            system_prompt=system_prompt,
            max_turns=task.max_turns,
            max_cost_usd=task.max_cost_usd,
            memory_manager=memory_manager,
            system_prompt_factory=prompt_factory,
            hook_registry=hooks,
        )

        async def consume():
            nonlocal done
            async for event in loop.run(task.prompt):
                if isinstance(event, TextDelta):
                    answer_parts.append(event.text)
                elif isinstance(event, DoneEvent):
                    done = event

        try:
            asyncio.run(asyncio.wait_for(consume(), timeout=task.session_timeout_seconds))
        except asyncio.TimeoutError:
            agent_error = f"Timed out after {task.session_timeout_seconds}s"
            termination_reason = "timeout"
        except Exception as exc:
            agent_error = f"{type(exc).__name__}: {exc}"
            termination_reason = "agent_error"
        finally:
            memory_manager.close()

    state = done.state if done is not None else loop.state
    if done is not None:
        reason = state.termination_reason
        termination_reason = reason.value if reason is not None else "completed"
    claimed = bool(
        done is not None
        and state is not None
        and state.termination_reason == TerminationReason.COMPLETED
    )
    return {
        "claimed_completion": claimed,
        "termination_reason": termination_reason,
        "agent_error": agent_error,
        "turns": state.turn_count if state else 0,
        "input_tokens": metrics["input_tokens"],
        "output_tokens": metrics["output_tokens"],
        "total_tokens": state.total_tokens if state else 0,
        "cost_usd": state.total_cost_usd if state else 0.0,
        "duration_seconds": time.perf_counter() - started,
        "model_calls": metrics["model_calls"],
        "tool_calls": metrics["tool_calls"],
        "tool_errors": metrics["tool_errors"],
        "tool_names": metrics["tool_names"],
        "answer": "".join(answer_parts),
    }


def render_markdown(metadata: dict, summary: dict, runs: list[dict]) -> str:
    lines = [
        "# MiniCode 真实编码任务评测",
        "",
        f"- Run ID: `{metadata['run_id']}`",
        f"- Model: `{metadata['model']}`",
        f"- Started: {metadata['started_at']}",
        f"- Repeats per task: {metadata['repeats']}",
        f"- Attempted: {summary['attempted']}",
        f"- Passed: {summary['passed']}/{summary['attempted']}",
        f"- False completions: {summary['false_completions']}/{summary['attempted']}",
        "",
        "| Task | Repeat | Passed | Claimed | False completion | Turns | Tokens | Cost | Seconds | Tools | Errors |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for run in runs:
        lines.append(
            f"| {run['task_id']} | {run['repeat_index']} | "
            f"{'yes' if run['passed'] else 'no'} | "
            f"{'yes' if run['claimed_completion'] else 'no'} | "
            f"{'yes' if run['false_completion'] else 'no'} | "
            f"{run['turns']} | {run['total_tokens']} | "
            f"${run['cost_usd']:.4f} | {run['duration_seconds']:.1f} | "
            f"{run['tool_calls']} | {run['tool_errors']} |"
        )
    lines.extend([
        "",
        "## Category summary",
        "",
        "| Category | Passed | Attempted | False completions |",
        "|---|---:|---:|---:|",
    ])
    for category, group in summary["by_category"].items():
        lines.append(
            f"| {category} | {group['passed']} | {group['attempted']} | "
            f"{group['false_completions']} |"
        )
    return "\n".join(lines) + "\n"


def write_reports(output_dir: Path, metadata: dict, runs: list[dict]) -> tuple[Path, Path]:
    summary = aggregate_runs(runs)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"real_coding_eval_{metadata['run_id']}"
    json_path = output_dir / f"{stem}.json"
    md_path = output_dir / f"{stem}.md"
    payload = {"metadata": metadata, "summary": summary, "runs": runs}
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
        newline="\n",
    )
    md_path.write_text(
        render_markdown(metadata, summary, runs),
        encoding="utf-8",
        newline="\n",
    )
    return json_path, md_path


def _validate_fixtures() -> int:
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(validate_task_fixture, CODING_TASKS))
    failures = []
    for task, (baseline_ok, solution_ok, detail) in zip(CODING_TASKS, results):
        if baseline_ok or not solution_ok:
            failures.append(f"{task.id}\n{detail}")
    if failures:
        print("\n\n".join(failures))
        return 1
    print(f"Validated {len(CODING_TASKS)} tasks: baseline fails, reference solution passes.")
    return 0


def cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run MiniCode on real coding tasks")
    parser.add_argument("--validate-fixtures", action="store_true")
    parser.add_argument("--task", default=None)
    parser.add_argument("--category", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--model", default=None)
    args = parser.parse_args(argv)

    if args.validate_fixtures:
        return _validate_fixtures()

    if args.repeats < 1:
        parser.error("--repeats must be positive")

    from config import Config

    base = Config.from_env()
    model = args.model or base.model
    missing = missing_credentials(model)
    if missing:
        parser.error(
            "missing model credentials: " + ", ".join(missing)
            + "; no evaluation report was created"
        )

    tasks = select_tasks(task_id=args.task, category=args.category, limit=args.limit)
    if not tasks:
        parser.error("no tasks selected")

    started = datetime.now(timezone.utc)
    metadata = {
        "run_id": started.strftime("%Y%m%dT%H%M%SZ"),
        "model": model,
        "started_at": started.isoformat(),
        "repeats": args.repeats,
        "suite_size": len(CODING_TASKS),
        "selected_tasks": [task.id for task in tasks],
    }
    runs: list[dict] = []
    output_dir = Path(args.output_dir)

    for task in tasks:
        task_config = Config(model=model, max_turns=task.max_turns, max_cost_usd=task.max_cost_usd)
        for repeat in range(1, args.repeats + 1):
            print(f"[run] {task.id} repeat={repeat}/{args.repeats}", flush=True)
            run = run_single_task(
                task,
                lambda root, _task=task, _config=task_config: run_agent_in_workspace(
                    root, _task, _config
                ),
                repeat_index=repeat,
            )
            runs.append(run)
            write_reports(output_dir, metadata, runs)

    json_path, md_path = write_reports(output_dir, metadata, runs)
    summary = aggregate_runs(runs)
    print(
        f"Passed {summary['passed']}/{summary['attempted']} runs; "
        f"false completions {summary['false_completions']}."
    )
    print(f"JSON: {json_path}")
    print(f"Report: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
