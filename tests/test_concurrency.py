"""并发工具执行测试 — 多个并发安全工具在同一次调用中执行，且顺序保留"""
import pytest

from core.agent_loop import AgentLoop
from core.state import ToolResult, DoneEvent
from core.tools.base import Tool


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
    input_tokens = 100
    output_tokens = 50


class MockModel:
    model = "mock"
    max_output_tokens = 8192

    def __init__(self, plan):
        self.plan = plan
        self.idx = 0

    async def chat_streaming(self, messages, system="", tools=None,
                             max_tokens=None):
        if self.idx >= len(self.plan):
            yield _Chunk(type="text_delta", text="Done.")
            yield _Chunk(type="message_stop", stop_reason="end_turn",
                         usage=_Usage())
            return
        for c in self.plan[self.idx]:
            yield c
        self.idx += 1


def _tool(name, input, cid):
    return _Chunk(type="tool_use_start", name=name, input=input, tool_call_id=cid)


def _stop():
    return _Chunk(type="message_stop", stop_reason="end_turn", usage=_Usage())


def _text(t):
    return _Chunk(type="text_delta", text=t)


class CountingTool(Tool):
    name = "counting"
    description = "test"
    input_schema = {"q": {"type": "string"}}
    is_readonly = True
    is_concurrency_safe = True

    def __init__(self):
        self.calls = []

    async def execute(self, q):
        self.calls.append(q)
        return f"result:{q}"


@pytest.mark.asyncio
async def test_multiple_safe_tools_execute_in_order():
    """三个并发安全工具在同一轮都被执行，结果按原顺序回流"""
    model = MockModel([
        [_tool("counting", {"q": "a"}, "c1"),
         _tool("counting", {"q": "b"}, "c2"),
         _tool("counting", {"q": "c"}, "c3"),
         _stop()],
        [_text("done"), _stop()],
    ])
    tool = CountingTool()
    loop = AgentLoop(tools=[tool], model_adapter=model, system_prompt="test")

    events = []
    async for e in loop.run("count three"):
        events.append(e)

    # 三个都被执行
    assert sorted(tool.calls) == ["a", "b", "c"]
    # 结果按原顺序回流（不是完成顺序）
    outputs = [e.output for e in events if isinstance(e, ToolResult)]
    assert outputs == ["result:a", "result:b", "result:c"]
    # 正常完成
    assert any(isinstance(e, DoneEvent) for e in events)
