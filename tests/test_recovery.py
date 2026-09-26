"""恢复路径测试 — max_tokens 升级/续写/放弃 + prompt too long 压缩重试"""
import pytest
from core.agent_loop import AgentLoop
from core.state import LoopState, ContinueReason
from core.tools.base import Tool


class MockModel:
    """Minimal model adapter stub for recovery-path unit tests."""
    model = "mock"
    max_output_tokens = 8192


class MockTool(Tool):
    name = "mock_tool"
    description = "test"
    input_schema = {"x": {"type": "string"}}
    is_readonly = True

    async def execute(self, x: str) -> str:
        return f"result: {x}"


def _make_loop(model=None):
    loop = AgentLoop(
        tools=[MockTool()],
        model_adapter=model or MockModel(),
        system_prompt="test",
    )
    loop._state = LoopState()  # initialize state for direct method calls
    return loop


# ── _is_prompt_too_long detection ──

def test_is_prompt_too_long_matches():
    loop = _make_loop()
    assert loop._is_prompt_too_long(Exception("prompt is too long"))
    assert loop._is_prompt_too_long(Exception("maximum context length exceeded"))
    assert loop._is_prompt_too_long(Exception("context_window size"))
    # 400 只有伴随上下文信号才算超长
    assert loop._is_prompt_too_long(
        Exception("Error code: 400 - maximum context length exceeded")
    )


def test_is_prompt_too_long_non_match():
    loop = _make_loop()
    assert not loop._is_prompt_too_long(Exception("some unrelated error"))
    assert not loop._is_prompt_too_long(Exception("division by zero"))
    # 裸 400（无上下文信号）不应误判为超长
    assert not loop._is_prompt_too_long(Exception("Error code: 400"))


# ── _handle_max_tokens: three-tier escalation ──

@pytest.mark.asyncio
async def test_max_tokens_silent_upgrade():
    """Tier 1: 8192 → 64000 silent upgrade"""
    model = MockModel()
    loop = _make_loop(model)
    result = await loop._handle_max_tokens()
    assert result is True
    assert model.max_output_tokens == 64_000
    assert loop._state.transition == ContinueReason.MAX_OUTPUT_TOKENS_UPGRADE


@pytest.mark.asyncio
async def test_max_tokens_continuation_prompt():
    """Tier 2: already at 64K → inject continuation prompt"""
    model = MockModel()
    model.max_output_tokens = 64_000
    loop = _make_loop(model)
    result = await loop._handle_max_tokens()
    assert result is True
    assert loop._state.max_output_tokens_recovery == 1
    assert loop._state.transition == ContinueReason.MAX_OUTPUT_TOKENS_RECOVERY
    # 续写提示已注入
    assert any("Resume directly" in m.content for m in loop._state.messages)


@pytest.mark.asyncio
async def test_max_tokens_give_up():
    """Tier 3: recovery retries exhausted → give up"""
    model = MockModel()
    model.max_output_tokens = 64_000
    loop = _make_loop(model)
    loop._state = LoopState(max_output_tokens_recovery=3)
    result = await loop._handle_max_tokens()
    assert result is False


# ── force_autocompact ──

@pytest.mark.asyncio
async def test_force_autocompact_reduces_messages():
    """force_autocompact 压缩消息并保留 System Prompt"""
    from capabilities.compression import ContextCompressor
    from core.state import Message

    compressor = ContextCompressor()
    state = LoopState(messages=(
        Message(role="system", content="You are an AI."),
        Message(role="user", content="task 1"),
        Message(role="assistant", content="Created file_a.py"),
        Message(role="assistant", content="Created file_b.py"),
        Message(role="user", content="task 2"),
    ))

    new_state = await compressor.force_autocompact(state)

    # System Prompt 保留
    assert new_state.messages[0].role == "system"
    assert "You are an AI" in new_state.messages[0].content
    # 消息数减少
    assert len(new_state.messages) < len(state.messages)
    # 有压缩摘要标记
    assert any("Compressed" in m.content or "Collapsed" in m.content
               for m in new_state.messages)


@pytest.mark.asyncio
async def test_force_autocompact_preserves_recent_edits():
    """压缩后恢复最近编辑的文件"""
    from capabilities.compression import ContextCompressor
    from core.state import Message

    compressor = ContextCompressor()
    compressor.record_edit("auth.py", "def login(): pass")

    state = LoopState(messages=(
        Message(role="system", content="AI"),
        Message(role="user", content="task"),
        Message(role="assistant", content="Edited auth.py"),
    ))

    new_state = await compressor.force_autocompact(state)

    # 恢复的文件内容在消息里
    assert any("auth.py" in m.content for m in new_state.messages)


# ── 流式中断恢复 ──

class _Chunk:
    """极简流式 chunk，模拟 model_adapter.StreamChunk"""
    def __init__(self, type, text="", stop_reason="", usage=None):
        self.type = type
        self.text = text
        self.name = ""
        self.input = {}
        self.tool_call_id = ""
        self.stop_reason = stop_reason
        self.usage = usage


class _Usage:
    input_tokens = 100
    output_tokens = 50


class FlakyModel:
    """第一次输出半截后抛网络异常，第二次正常完成"""
    model = "mock"
    max_output_tokens = 8192

    def __init__(self):
        self.calls = 0

    async def chat_streaming(self, messages, system="", tools=None,
                             max_tokens=None):
        self.calls += 1
        if self.calls == 1:
            yield _Chunk(type="text_delta", text="I suggest using ")
            yield _Chunk(type="text_delta", text="asyncio.gather ")
            raise ConnectionError("Connection reset by peer")
        yield _Chunk(type="text_delta", text="to run them concurrently.")
        yield _Chunk(type="message_stop", stop_reason="end_turn",
                     usage=_Usage())


@pytest.mark.asyncio
async def test_stream_interruption_saves_partial_and_resumes(monkeypatch):
    """网络中断半截输出 → 落盘 + 注入续写提示 + 重试完成"""
    from core.state import DoneEvent

    monkeypatch.setattr("core.agent_loop._STREAM_RETRY_BASE_DELAY", 0)

    model = FlakyModel()
    loop = AgentLoop(
        tools=[MockTool()],
        model_adapter=model,
        system_prompt="test",
    )

    events = []
    async for event in loop.run("do async work"):
        events.append(event)

    done = [e for e in events if isinstance(e, DoneEvent)]
    assert done, "应正常完成（重试后）"
    state = done[0].state

    # 半截内容已落盘（之前会丢失）
    assert any(
        "I suggest using" in m.content
        for m in state.messages if m.role == "assistant"
    ), "半截输出应写入状态"
    # 续写提示已注入
    assert any(
        ("截断" in m.content or "中断" in m.content)
        for m in state.messages if m.role == "user"
    ), "应注入续写提示"
    # 重试了一次
    assert model.calls == 2


@pytest.mark.asyncio
async def test_stream_interruption_gives_up_after_retries(monkeypatch):
    """反复中断超过重试上限 → 放弃并抛出异常"""
    class AlwaysFlaky(FlakyModel):
        async def chat_streaming(self, messages, system="", tools=None,
                                 max_tokens=None):
            self.calls += 1
            yield _Chunk(type="text_delta", text="partial ")
            raise ConnectionError("Connection reset by peer")

    monkeypatch.setattr("core.agent_loop._STREAM_RETRY_BASE_DELAY", 0)

    model = AlwaysFlaky()
    loop = AgentLoop(tools=[MockTool()], model_adapter=model,
                     system_prompt="test")

    with pytest.raises(ConnectionError):
        async for _ in loop.run("task"):
            pass

    # 重试上限 3 次 → 共 4 次调用（1 初始 + 3 重试）
    assert model.calls == 4


def test_is_transient_error():
    """瞬态错误（网络/超时/限流）识别"""
    loop = _make_loop()
    assert loop._is_transient_error(ConnectionError("Connection reset by peer"))
    assert loop._is_transient_error(Exception("Read timed out"))
    assert loop._is_transient_error(Exception("HTTP 429 rate limit exceeded"))
    assert loop._is_transient_error(Exception("503 Service Unavailable"))
    # 永久错误不重试
    assert not loop._is_transient_error(Exception("invalid api key"))
    assert not loop._is_transient_error(Exception("400 bad request"))


def test_strip_orphaned_tool_calls():
    """压缩边界切断配对时，去掉 orphaned tool_calls / tool 消息"""
    from capabilities.compression import _strip_orphaned_tool_calls
    from core.state import Message

    msgs = (
        Message(role="system", content="S"),
        Message(role="user", content="task"),
        # assistant 带 tool_calls，但 tool 结果被压掉 → 应去掉 tool_calls
        Message(role="assistant", content="", tool_calls=[
            {"id": "call_1", "type": "function",
             "function": {"name": "Read", "arguments": "{}"}}
        ]),
        Message(role="user", content="[Collapsed summary]"),
        # orphaned tool 消息（没有对应 assistant）→ 应丢掉
        Message(role="tool", content="result", tool_call_id="call_2"),
        Message(role="assistant", content="final answer"),
    )
    result = _strip_orphaned_tool_calls(msgs)

    # assistant 的 orphaned tool_calls 被去掉
    assert result[2].role == "assistant"
    assert result[2].tool_calls is None
    # orphaned tool 消息被丢掉
    assert not any(m.role == "tool" for m in result)


def test_strip_orphaned_tool_calls_keeps_valid_pair():
    """完整配对的 tool_calls + tool 结果应保留"""
    from capabilities.compression import _strip_orphaned_tool_calls
    from core.state import Message

    msgs = (
        Message(role="assistant", content="", tool_calls=[
            {"id": "call_1", "type": "function",
             "function": {"name": "Read", "arguments": "{}"}}
        ]),
        Message(role="tool", content="file content", tool_call_id="call_1"),
    )
    result = _strip_orphaned_tool_calls(msgs)
    assert result[0].tool_calls is not None
    assert result[1].role == "tool"
    assert result[1].tool_call_id == "call_1"
