import pytest

from core.agent_loop import AgentLoop
from core.hooks import HookDecision, HookEvent, HookRegistry
from core.state import ToolResult
from core.tools.base import Tool


class CountingTool(Tool):
    name = "count"
    description = "count calls"
    input_schema = {"value": {"type": "string"}}
    is_readonly = True
    is_concurrency_safe = True

    def __init__(self):
        self.call_count = 0

    async def execute(self, value: str) -> str:
        self.call_count += 1
        return f"counted:{value}"


class DestructiveTool(CountingTool):
    name = "danger"
    description = "destructive test tool"
    is_readonly = False
    is_concurrency_safe = False
    is_destructive = True


class FailingTool(Tool):
    name = "fail"
    description = "failing test tool"
    input_schema = {"value": {"type": "string"}}
    is_readonly = True
    is_concurrency_safe = True

    async def execute(self, value: str) -> str:
        raise RuntimeError("boom")


class Chunk:
    def __init__(
        self,
        type,
        text="",
        name="",
        input=None,
        stop_reason="",
        usage=None,
        tool_call_id="",
    ):
        self.type = type
        self.text = text
        self.name = name
        self.input = input
        self.stop_reason = stop_reason
        self.usage = usage
        self.tool_call_id = tool_call_id


class Usage:
    def __init__(self):
        self.input_tokens = 10
        self.output_tokens = 5
        self.cache_read_tokens = 0
        self.cache_write_tokens = 0


class Model:
    model = "mock"
    max_output_tokens = 8192

    def __init__(self, tool_name: str = "count"):
        self.calls = 0
        self.tool_name = tool_name

    async def chat_streaming(self, messages, system="", tools=None, max_tokens=None):
        self.calls += 1
        if self.calls == 1:
            yield Chunk(
                "tool_use_start",
                name=self.tool_name,
                input={"value": "one"},
                tool_call_id="call-1",
            )
            yield Chunk("message_stop", stop_reason="tool_use", usage=Usage())
            return

        yield Chunk("text_delta", text="finished")
        yield Chunk("message_stop", stop_reason="end_turn", usage=Usage())


@pytest.mark.asyncio
async def test_agent_loop_emits_documented_lifecycle_events():
    hooks = HookRegistry()
    seen = []

    for event in HookEvent:
        async def record(*args, _event=event):
            seen.append(_event)

        hooks.register(event, record)

    tool = CountingTool()
    loop = AgentLoop(
        tools=[tool],
        model_adapter=Model(),
        system_prompt="test",
        hook_registry=hooks,
    )

    async for _ in loop.run("count one"):
        pass

    assert seen == [
        HookEvent.USER_PROMPT,
        HookEvent.AFTER_COMPRESSION,
        HookEvent.BEFORE_MODEL,
        HookEvent.AFTER_MODEL,
        HookEvent.BEFORE_TOOL,
        HookEvent.AFTER_TOOL,
        HookEvent.AFTER_COMPRESSION,
        HookEvent.BEFORE_MODEL,
        HookEvent.AFTER_MODEL,
        HookEvent.STOP,
    ]
    assert tool.call_count == 1


@pytest.mark.asyncio
async def test_before_tool_hook_can_deny_without_executing_tool():
    hooks = HookRegistry()

    async def deny(call, tool):
        return HookDecision(allow=False, reason="blocked by policy")

    hooks.register(HookEvent.BEFORE_TOOL, deny)
    tool = CountingTool()
    loop = AgentLoop(
        tools=[tool],
        model_adapter=Model(),
        system_prompt="test",
        hook_registry=hooks,
    )

    results = []
    async for event in loop.run("count one"):
        if isinstance(event, ToolResult):
            results.append(event.output)

    assert tool.call_count == 0
    assert results == ["blocked by policy"]


@pytest.mark.asyncio
async def test_builtin_permission_hook_remains_fail_closed():
    async def deny(*args, **kwargs):
        return False

    tool = DestructiveTool()
    loop = AgentLoop(
        tools=[tool],
        model_adapter=Model(tool_name="danger"),
        system_prompt="test",
    )
    loop._permission.authorize = deny

    results = []
    async for event in loop.run("run dangerous tool"):
        if isinstance(event, ToolResult):
            results.append(event.output)

    assert tool.call_count == 0
    assert results == ["Permission denied by user."]


@pytest.mark.asyncio
async def test_after_tool_hook_observes_execution_failure():
    hooks = HookRegistry()
    observed = []

    async def record_result(call, tool, result):
        observed.append(result)

    hooks.register(HookEvent.AFTER_TOOL, record_result)
    loop = AgentLoop(
        tools=[FailingTool()],
        model_adapter=Model(tool_name="fail"),
        system_prompt="test",
        hook_registry=hooks,
    )

    async for _ in loop.run("run failing tool"):
        pass

    assert observed == ["Tool error: boom"]
