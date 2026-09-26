"""真实端到端运行 + 存档报告。

这是项目里「真的调 API」的证据工具：跑一次真实任务，把结果
（轮次 / token / 成本 / 耗时 / 完整回答 / 工具调用）存档到
reports/ 目录下的 Markdown 报告，供面试展示。

用法:
    DEEPSEEK_API_KEY=sk-xxx python benchmarks/run_real.py "列出当前目录的文件"
    DEEPSEEK_API_KEY=sk-xxx python benchmarks/run_real.py   # 默认任务

依赖 .env 里的 MINICODE_MODEL / MINICODE_PROVIDER（默认 deepseek）。
"""
import asyncio
import sys
import time
from datetime import datetime
from pathlib import Path

# 让脚本能 import 项目根目录的模块（config / main / core）
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config
import main
from core.state import TextDelta, ToolResult, DoneEvent


async def run_task(task: str) -> dict:
    config = Config.from_env()
    tools, system_prompt, _, memory_manager = main._build_tools("normal", config)
    loop = main._make_loop(config, tools, system_prompt,
                           config.max_turns, config.max_cost_usd, memory_manager)

    start = time.time()
    text_parts: list[str] = []
    tool_results: list[ToolResult] = []
    done: DoneEvent | None = None
    try:
        async for event in loop.run(task):
            if isinstance(event, TextDelta):
                text_parts.append(event.text)
                print(event.text, end="", flush=True)
            elif isinstance(event, ToolResult):
                tool_results.append(event)
                print(f"\n[{event.tool_name}] {event.output[:120]}", flush=True)
            elif isinstance(event, DoneEvent):
                done = event
    finally:
        memory_manager.close()

    elapsed = time.time() - start
    state = done.state if done else None
    return {
        "task": task,
        "turns": state.turn_count if state else 0,
        "tokens": state.total_tokens if state else 0,
        "cost_usd": state.total_cost_usd if state else 0.0,
        "elapsed": elapsed,
        "answer": "".join(text_parts),
        "tool_results": tool_results,
    }


def save_report(result: dict) -> Path:
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = reports_dir / f"run_{ts}.md"

    lines = [
        "# MiniCode 真实运行报告",
        "",
        f"- 时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 任务: {result['task']}",
        f"- 轮次: {result['turns']}",
        f"- Token: {result['tokens']:,}",
        f"- 成本: ${result['cost_usd']:.4f}",
        f"- 耗时: {result['elapsed']:.1f}s",
        "",
        "## 最终回答",
        "",
        result["answer"].strip() or "（无输出）",
        "",
        "## 工具调用",
        "",
    ]
    if result["tool_results"]:
        for tr in result["tool_results"]:
            lines.append(f"- **{tr.tool_name}**: {tr.output[:200]}")
    else:
        lines.append("（无工具调用）")

    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def cli():
    task = sys.argv[1] if len(sys.argv) > 1 else "列出当前目录下的文件"
    result = asyncio.run(run_task(task))
    path = save_report(result)
    print(f"\n\n报告已存档: {path}")


if __name__ == "__main__":
    cli()
