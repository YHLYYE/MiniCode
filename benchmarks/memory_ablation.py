"""记忆注入的消融实验 —— 把"记忆有没有用"从轶事变成数字。

设计要点：**让"有没有用上那条经验"产生可观测的差异**。
夹具项目里放一个 conftest.py：不设环境变量 `MINICODE_OFFLINE=1` 就直接报错，
于是
  · 不带记忆注入：agent 直接跑 pytest → 撞错误 → 多花回合去猜；
  · 带记忆注入：任务开始时召回"跑 pytest 前必须设 MINICODE_OFFLINE=1"这条经验，
    一次把命令写对。

对比组之间唯一的差别就是**这一条记忆是否注入**（同一个夹具、同一批任务、同一个模型）。

指标：平均回合数、工具错误数、pytest 通过率、平均 token。

为什么是 benchmark 而不是 pytest 用例：它要真调模型（要 key、要网络、要花钱）。
结果写进存档（`--out`），改动大的时候重跑对比。

用法：
    python benchmarks/memory_ablation.py --tasks 5 --max-turns 6
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# 夹具：conftest 强制要求 MINICODE_OFFLINE，否则整个测试会话直接失败。
#
# 关键设计：**报错信息里不写变量名**（只说"按团队约定准备"）。
# 第一版把变量名写在了报错里，结果控制组读一眼 conftest 就自己修好了 ——
# 两组零差异、而且都撞到 max_turns，指标完全没有区分度。
# 真实世界里"团队约定"往往不在仓库里（Claude Code 的 auto memory 文档也明确说
# 它**跳过能从代码库推导的东西**），所以这里必须让约定不可从仓库推导：
# 只有记忆里有那条经验，才可能一次做对。
CONFTEST = '''"""离线夹具：不按团队约定准备环境就拒绝运行。"""
import os
import pytest

if os.environ.get("MINICODE_OFFLINE") != "1":
    raise RuntimeError(
        "测试环境未就绪：请按团队约定准备离线测试环境后再重试（约定不在本仓库内）"
    )


@pytest.fixture
def sample():
    return 42
'''

TEST_FILE = """def test_sample():
    assert 1 + 1 == 2
"""

CONVENTION = ("跑 pytest 之前必须先设置环境变量 MINICODE_OFFLINE=1，"
              "否则 conftest 会直接拒绝运行、整批测试失败")

TASKS = [
    "跑一下 pytest 看看测试结果",
    "用 pytest 验证一下当前仓库的测试是否通过",
    "运行 pytest，如果失败就说明原因",
    "先跑 pytest，再告诉我结论",
    "帮我把 pytest 跑起来并汇报结果",
]


def _prepare_project(base: Path) -> Path:
    proj = base / "fixture"
    proj.mkdir(parents=True, exist_ok=True)
    (proj / "conftest.py").write_text(CONFTEST, encoding="utf-8")
    (proj / "test_sample.py").write_text(TEST_FILE, encoding="utf-8")
    env_src = ROOT / ".env"
    if env_src.exists():
        shutil.copy(env_src, proj / ".env")
    return proj


async def _run_group(proj: Path, inject: bool, max_turns: int, limit: int) -> dict:
    """在同一个夹具项目里跑一组任务，返回该组的指标。"""
    os.chdir(proj)
    # 延迟导入：main 里的 Config 会读 .env，必须在 chdir 之后再导入
    for mod in [m for m in list(sys.modules) if m in ("main", "config_loader")]:
        del sys.modules[mod]
    import main as agent_main

    from core.state import DoneEvent, ToolResult

    config = agent_main.Config.from_env()
    tools, system_prompt, _mode, memory_manager, factory = agent_main._build_tools(
        "normal", config, inject_memory=inject)

    rows = []
    for task in TASKS[:limit]:
        loop = agent_main._make_loop(config, tools, system_prompt, max_turns,
                                     0.05, memory_manager, factory)
        turns = tokens = errors = 0
        passed = False
        async for ev in loop.run(task):
            if isinstance(ev, ToolResult):
                out = ev.output or ""
                if out.startswith(("Tool error:", "Error:", "Security blocked",
                                   "Permission denied")):
                    errors += 1
                if "1 passed" in out:
                    passed = True
            elif isinstance(ev, DoneEvent):
                turns = ev.state.turn_count
                tokens = ev.state.total_tokens
        rows.append({"task": task, "turns": turns, "tokens": tokens,
                     "tool_errors": errors, "passed": passed})

    n = len(rows)
    summary = {
        "inject": inject,
        "n": n,
        "mean_turns": sum(r["turns"] for r in rows) / n,
        "tool_errors": sum(r["tool_errors"] for r in rows),
        "passed": sum(r["passed"] for r in rows),
        "mean_tokens": sum(r["tokens"] for r in rows) / n,
        "rows": rows,
    }
    memory_manager.close()
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="记忆注入消融实验")
    ap.add_argument("--tasks", type=int, default=len(TASKS))
    ap.add_argument("--max-turns", type=int, default=6)
    ap.add_argument("--out", default=str(ROOT / "benchmarks" / "memory_ablation.json"))
    args = ap.parse_args(argv)

    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        # 两组各用一个独立目录：避免一组的记忆写入污染另一组
        results = {}
        try:
            for label, inject in (("no_memory", False), ("with_memory", True)):
                proj = _prepare_project(base / label)
                os.chdir(proj)
                # 预置那条约定（两组完全相同），模拟"上一轮沉淀下来的经验"
                import asyncio as _a
                from capabilities.memory import MemoryManager
                mm = MemoryManager(project_root=proj)
                _a.run(mm.record_procedural(CONVENTION))
                mm.close()
                results[label] = _a.run(_run_group(proj, inject, args.max_turns,
                                                   args.tasks))
        finally:
            # 必须在退出临时目录上下文之前离开它：进程 cwd 停在里面时，
            # Windows 上 rmtree 会失败（拿被占用的目录没办法）。
            os.chdir(ROOT)

    print(f"{'组':<14}{'回合':>6}{'工具错误':>9}{'pytest 通过':>12}{'token':>10}")
    for label, r in results.items():
        print(f"{label:<14}{r['mean_turns']:>6.1f}{r['tool_errors']:>9}"
              f"{r['passed']:>7}/{r['n']:<4}{r['mean_tokens']:>10.0f}")
    w, n = results["with_memory"], results["no_memory"]
    print(f"\n差值：回合 {w['mean_turns'] - n['mean_turns']:+.1f}，"
          f"工具错误 {w['tool_errors'] - n['tool_errors']:+d}，"
          f"通过 {w['passed'] - n['passed']:+d}")

    archive = {"generated_at": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
               "tasks": args.tasks, "results": results}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(archive, ensure_ascii=False, indent=2),
                              encoding="utf-8")
    print(f"已写入存档：{args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
