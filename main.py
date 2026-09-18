#!/usr/bin/env python3
"""MiniCode CLI — AI Coding Agent

Usage:
    python main.py                              # 交互式 REPL（推荐）
    python main.py "Fix the login bug"          # 单次任务
    python main.py --mode plan                  # Plan 模式交互
"""

import asyncio
import sys
from pathlib import Path

# Fix Windows GBK encoding for unicode output
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import click

from config import Config
from core.model_adapter import ModelAdapter
from core.agent_loop import (
    AgentLoop, TextDelta, ToolStart, ToolResult, DoneEvent
)
from core.tools.files import ReadTool, WriteTool
from core.tools.shell import BashTool
from core.tools.base import SkillTool, RecallMemoryTool, RememberTool
from capabilities.skill import SkillSystem
from capabilities.memory import MemoryManager
from capabilities.multi_agent import AgentTool
from capabilities.security import resolve_tools_for_mode, ExecutionMode
from prompt.system_prompt import build_system_prompt
from session_store import SessionStore


def _build_tools(mode: str, config: Config):
    """Initialize subsystems and return (tools, system_prompt, exec_mode, memory_manager)."""
    skill_system = SkillSystem()
    skill_system.register_from_source(Path("skills/"), priority=10)
    memory_manager = MemoryManager()

    def _agent_factory(tools, system_prompt, max_turns):
        """Spawn a sub-agent sharing the parent's model and memory store."""
        return AgentLoop(
            tools=tools,
            model_adapter=ModelAdapter(config.model),
            system_prompt=system_prompt,
            max_turns=max_turns,
            max_cost_usd=config.max_cost_usd,
            memory_manager=memory_manager,  # project-scoped memory sharing
        )

    base_tools = [
        ReadTool(), WriteTool(), BashTool(),
        SkillTool(skill_system),
        RecallMemoryTool(memory_manager),
        RememberTool(memory_manager),
    ]

    # Sub-agents get the base toolset only (no Agent → no unbounded recursion).
    agent_tool = AgentTool(
        agent_loop_factory=_agent_factory,
        tool_registry={t.name: t for t in base_tools},
        project_root=Path.cwd(),
    )

    all_tools = base_tools + [agent_tool]

    exec_mode = ExecutionMode.PLAN if mode == "plan" else ExecutionMode.NORMAL
    tools = resolve_tools_for_mode(all_tools, exec_mode)

    system_prompt = build_system_prompt(
        skill_index=skill_system.get_index_for_system_prompt(),
        claude_md=memory_manager.load_claude_md(),
    )
    return tools, system_prompt, exec_mode, memory_manager


@click.command()
@click.argument("task", required=False)
@click.option("--mode", type=click.Choice(["plan", "normal"]), default="normal",
              help="Execution mode: plan (read-only) or normal (full access)")
@click.option("--max-turns", default=20, help="Maximum agent turns per task")
@click.option("--max-cost", default=5.0, help="Maximum USD cost per session")
@click.option("--resume", default=None, help="Resume a previous session by id")
@click.option("--list-sessions", is_flag=True, help="List saved sessions and exit")
def main(task: str | None, mode: str, max_turns: int, max_cost: float,
         resume: str | None, list_sessions: bool):
    """MiniCode — AI Coding Agent

    TASK: (可选) 单次任务。不传则进入交互式 REPL。
    """
    config = Config.from_env()
    config.max_turns = max_turns
    config.max_cost_usd = max_cost

    store = SessionStore()

    # --list-sessions: 列出历史会话后退出
    if list_sessions:
        sessions = store.list_sessions()
        if not sessions:
            print("暂无保存的会话。")
        else:
            print("已保存的会话：")
            for sid in sessions:
                print(f"  {sid}")
        return

    tools, system_prompt, exec_mode, memory_manager = _build_tools(mode, config)

    # --resume: 加载历史会话
    resume_state = None
    if resume:
        resume_state = store.load(resume)
        if resume_state is None:
            print(f"会话 {resume} 不存在。可用: {store.list_sessions()}")
            return
        print(f"[已恢复会话 {resume}，{resume_state.turn_count} 轮历史]")

    if task:
        # 单次任务模式
        _print_header(config, mode, max_turns, max_cost)
        loop = _make_loop(config, tools, system_prompt, max_turns, max_cost,
                          memory_manager)
        if resume_state:
            loop._state = resume_state
        print(f"> {task}\n")
        asyncio.run(_run_task(loop, task))
        if loop.state:
            store.save(loop.state)
    else:
        # 交互式 REPL 模式
        asyncio.run(_interactive_repl(config, tools, system_prompt, exec_mode,
                                      store, resume_state, memory_manager))


def _make_loop(config, tools, system_prompt, max_turns, max_cost,
               memory_manager=None):
    return AgentLoop(
        tools=tools,
        model_adapter=ModelAdapter(config.model),
        system_prompt=system_prompt,
        max_turns=max_turns,
        max_cost_usd=max_cost,
        memory_manager=memory_manager,
    )


def _print_header(config, mode, max_turns, max_cost):
    mode_label = "PLAN (只读)" if mode == "plan" else "NORMAL (全部工具)"
    print(f"Model: {config.model} | 模式: {mode_label}")
    print(f"Max turns: {max_turns} | Max cost: ${max_cost:.2f}")
    print("=" * 60)


async def _interactive_repl(config, tools, system_prompt, exec_mode,
                            store: SessionStore | None = None,
                            resume_state=None, memory_manager=None):
    """交互式 REPL — 启动一次，连续对话，上下文持续。"""
    print("=" * 60)
    print("MiniCode 交互模式")
    print(f"Model: {config.model}")
    print(f"工具: {'只读' if exec_mode == ExecutionMode.PLAN else '全部'}")
    print("命令: /clear 清空 | /save 保存 | /list 会话 | /exit 退出 | /help 帮助")
    print("=" * 60)

    loop = _make_loop(config, tools, system_prompt, config.max_turns,
                      config.max_cost_usd, memory_manager)
    if resume_state:
        loop._state = resume_state

    while True:
        try:
            task = input("\n你 > ").strip()
        except (EOFError, KeyboardInterrupt):
            if loop.state:
                store.save(loop.state)
            print("\n再见！")
            return

        if not task:
            continue

        cmd = task.lower()
        if cmd in ("/exit", "/quit", "/q"):
            if loop.state:
                store.save(loop.state)
            print("再见！")
            return
        if cmd in ("/clear", "/reset"):
            loop = _make_loop(config, tools, system_prompt,
                              config.max_turns, config.max_cost_usd,
                              memory_manager)
            print("[已清空对话历史]")
            continue
        if cmd in ("/save", "/s"):
            if loop.state:
                sid = store.save(loop.state)
                print(f"[会话已保存: {sid}]")
            else:
                print("[暂无对话可保存]")
            continue
        if cmd in ("/list", "/l"):
            sessions = store.list_sessions()
            if not sessions:
                print("暂无保存的会话。")
            else:
                print("已保存的会话：")
                for sid in sessions:
                    print(f"  {sid}")
            continue
        if cmd in ("/help", "/?"):
            print("直接输入任务即可，例如：列出当前目录的文件")
            print("/clear  清空对话历史")
            print("/save   保存当前会话")
            print("/list   列出历史会话")
            print("/exit   退出")
            continue

        print()
        await _run_task(loop, task)


async def _run_task(loop: AgentLoop, task: str):
    """执行一次任务，渲染流式事件。任务间共享 loop 的 state，上下文连续。"""
    try:
        async for event in loop.run(task):
            if isinstance(event, TextDelta):
                sys.stdout.write(event.text)
                sys.stdout.flush()
            elif isinstance(event, ToolStart):
                print(f"\n[{event.tool_name}] ", end="", flush=True)
            elif isinstance(event, ToolResult):
                short = event.output[:200].replace("\n", " ").replace("\r", " ")
                print(f"=> {short}...", flush=True)
                if len(event.output) > 200:
                    print(f"  ({len(event.output)} chars total)", flush=True)
            elif isinstance(event, DoneEvent):
                s = event.state
                print(f"\n{'='*60}")
                print(
                    f"完成: {s.turn_count} turns, "
                    f"{s.total_tokens:,} tokens, "
                    f"${s.total_cost_usd:.4f}"
                )
    except KeyboardInterrupt:
        print("\n\n已中断。")
    except Exception as e:
        print(f"\n[X] 错误: {e}")


if __name__ == "__main__":
    import os

    # Windows 上 C/Rust 扩展（tiktoken 等）在 Python 解释器退出时可能
    # 触发析构崩溃（access violation / segfault）。用 os._exit 绕过清理
    # 阶段，确保进程干净退出。
    try:
        main()
    except SystemExit as e:
        # click 正常退出（sys.exit(code)）
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(e.code if e.code is not None else 0)
    except BaseException:
        # 真实异常：正常传播，保留 traceback
        raise
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
