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
    assert loop._is_prompt_too_long(Exception("Error code: 400"))


def test_is_prompt_too_long_non_match():
    loop = _make_loop()
    assert not loop._is_prompt_too_long(Exception("some unrelated error"))
    assert not loop._is_prompt_too_long(Exception("division by zero"))


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
