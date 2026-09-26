"""真实端到端冒烟测试 — 需要 DEEPSEEK_API_KEY（无 key 时自动跳过）。

这是项目里唯一「真的调 API」的测试，证明 agent 端到端能跑通。
"""
import os

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("DEEPSEEK_API_KEY"),
    reason="需要 DEEPSEEK_API_KEY 才能跑真实端到端测试",
)


@pytest.mark.asyncio
async def test_real_agent_completes_simple_task():
    from config import Config
    import main
    from core.state import DoneEvent, TextDelta

    config = Config.from_env()
    tools, system_prompt, _, memory_manager = main._build_tools("normal", config)
    loop = main._make_loop(config, tools, system_prompt,
                           config.max_turns, config.max_cost_usd, memory_manager)
    try:
        events = []
        async for event in loop.run("列出当前目录下的文件"):
            events.append(event)

        assert any(isinstance(e, DoneEvent) for e in events), "agent 应正常完成"
        text = "".join(e.text for e in events if isinstance(e, TextDelta))
        assert text.strip(), "agent 应有输出"
    finally:
        memory_manager.close()
