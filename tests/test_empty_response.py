"""空响应守卫 —— 模型返回「既没有文本、也没有工具调用」时不再空转到 max_turns。

发现过程：审 agent_loop.py 的死代码时顺手实测的。当时第 4 段
`if assistant_text or tool_calls_attachments:` 没有 else 分支 —— 空响应什么都
不追加，第 5 段的结束条件也判不出来，于是循环一直转到 max_turns。
实测（max_turns=20）：**API 被调用 20 次，每次都是真花钱的调用。**

现在：先注入一条提醒让模型补救（最多 2 次），仍为空则明确终止。
"""
import pytest

from core.agent_loop import AgentLoop
from core.state import DoneEvent


class _Chunk:
    def __init__(self, type, text="", name="", input=None, stop_reason="",
                 usage=None, tool_call_id=""):
        self.type = type
        self.text = text
        self.name = name
        self.input = input
        self.stop_reason = stop_reason
        self.usage = usage
        self.tool_call_id = tool_call_id


class _Usage:
    input_tokens = 10
    output_tokens = 0


class _EmptyModel:
    """永远返回空响应"""

    model = "mock"
    max_output_tokens = 8192

    def __init__(self):
        self.calls = 0

    async def chat_streaming(self, messages, system="", tools=None,
                             max_tokens=None):
        self.calls += 1
        yield _Chunk(type="message_stop", stop_reason="end_turn", usage=_Usage())


class _RecoverModel:
    """第一次空响应，之后正常作答 —— 验证"提醒一次就能补救"。"""

    model = "mock"
    max_output_tokens = 8192

    def __init__(self):
        self.calls = 0

    async def chat_streaming(self, messages, system="", tools=None,
                             max_tokens=None):
        self.calls += 1
        if self.calls == 1:
            yield _Chunk(type="message_stop", stop_reason="end_turn",
                         usage=_Usage())
        else:
            yield _Chunk(type="text_delta", text="这是答案")
            yield _Chunk(type="message_stop", stop_reason="end_turn",
                         usage=_Usage())


@pytest.mark.asyncio
async def test_empty_response_does_not_spin_to_max_turns():
    """关键是别再烧 20 次 API 调用。"""
    model = _EmptyModel()
    loop = AgentLoop(tools=[], model_adapter=model, system_prompt="test",
                     max_turns=20)
    events = [e async for e in loop.run("task")]

    done = [e for e in events if isinstance(e, DoneEvent)]
    assert done, "应当以 DoneEvent 明确退出，而不是静默跑到上限"
    # 2 次提醒 + 1 次最终失败 = 3 次调用；远小于 max_turns=20
    assert model.calls == 3, f"预期 3 次调用，实际 {model.calls}"
    assert done[0].state.turn_count == 3


@pytest.mark.asyncio
async def test_empty_response_recovers_if_model_complies():
    """提醒之后模型正常作答 → 任务正常完成，用户看不到异常。"""
    model = _RecoverModel()
    loop = AgentLoop(tools=[], model_adapter=model, system_prompt="test")
    events = [e async for e in loop.run("task")]

    assert model.calls == 2
    done = [e for e in events if isinstance(e, DoneEvent)]
    assert done[0].state.messages[-1].content == "这是答案"
