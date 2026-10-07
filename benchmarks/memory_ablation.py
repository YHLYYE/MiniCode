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
# 夹具的关键设计：**仓库主动误导**。
# README 和 Makefile 都指向一个坏掉的测试入口（make test → 直接 pytest，不准备环境），
# 真正的入口 `run_tests.py` 只有团队约定里说。这样"知道约定"才真的有价值 ——
# 否则 agent 读一眼代码就能自己推导出来，两组必然零差异（第一版就是这样失败的）。
README = """# fixture 项目

## 测试

跑测试请执行：

```bash
make test
```
"""

MAKEFILE = """test:
\tpytest -q
"""

RUN_TESTS = '''"""仓库的测试入口 —— 它会自己准备好离线环境再跑 pytest。"""
import os
import pytest

os.environ.setdefault("MINICODE_OFFLINE", "1")
raise SystemExit(pytest.main(["-q"]))
'''

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

CONVENTION = ("本仓库跑测试要用 `python run_tests.py`（它会自己准备离线环境）；"
              "README 里写的 `make test` 那个 target 是坏的，直接跑 pytest 也会失败")

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
    (proj / "README.md").write_text(README, encoding="utf-8")
    (proj / "Makefile").write_text(MAKEFILE, encoding="utf-8")
    (proj / "run_tests.py").write_text(RUN_TESTS, encoding="utf-8")
    env_src = ROOT / ".env"
    if env_src.exists():
        shutil.copy(env_src, proj / ".env")
    return proj


async def _run_group(proj: Path, inject: bool, max_turns: int, limit: int,
                     repeat: int) -> dict:
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
    for _ in range(repeat):
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
    sorted_turns = sorted(r["turns"] for r in rows)
    median_turns = (sorted_turns[n // 2] if n % 2
                    else (sorted_turns[n // 2 - 1] + sorted_turns[n // 2]) / 2)
    summary = {
        "inject": inject,
        "n": n,
        "mean_turns": sum(r["turns"] for r in rows) / n,
        "median_turns": median_turns,
        "min_turns": sorted_turns[0],
        "max_turns": sorted_turns[-1],
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
    ap.add_argument("--repeat", type=int, default=1,
                    help="每个任务重复几次（单次方差大，≥2 才看得住）")
    ap.add_argument("--out", default=str(ROOT / "benchmarks" / "memory_ablation.json"))
    args = ap.parse_args(argv)

    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        # 两组各用一个独立目录：避免一组的记忆写入污染另一组
        results = {}
        # 非交互模式：基准里拿不到人工确认，L4 会把合法命令也拒掉（fail-closed），
        # 随机污染指标。用显式开关放行，并在输出里说明。
        os.environ["MINICODE_AUTO_APPROVE"] = "1"
        print("[说明] 已设 MINICODE_AUTO_APPROVE=1：基准环境里 L4 无人可问，"
              "默认 fail-closed 会拒绝合法命令，故显式放行。")
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
                                                   args.tasks, args.repeat))
        finally:
            # 必须在退出临时目录上下文之前离开它：进程 cwd 停在里面时，
            # Windows 上 rmtree 会失败（拿被占用的目录没办法）。
            os.chdir(ROOT)

    print(f"{'组':<14}{'平均回合':>10}{'中位回合':>10}{'区间':>10}"
          f"{'工具错误':>9}{'通过':>8}{'token':>10}")
    for label, r in results.items():
        print(f"{label:<14}{r['mean_turns']:>10.1f}{r['median_turns']:>10.1f}"
              f"{str((r['min_turns'], r['max_turns'])):>10}"
              f"{r['tool_errors']:>9}{r['passed']:>5}/{r['n']:<3}"
              f"{r['mean_tokens']:>10.0f}")
    w, n = results["with_memory"], results["no_memory"]
    print(f"\n差值：回合 {w['mean_turns'] - n['mean_turns']:+.1f}，"
          f"工具错误 {w['tool_errors'] - n['tool_errors']:+d}，"
          f"通过 {w['passed'] - n['passed']:+d}")

    archive = {"generated_at": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
               "tasks": args.tasks, "repeat": args.repeat, "results": results}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(archive, ensure_ascii=False, indent=2),
                              encoding="utf-8")
    print(f"已写入存档：{args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
