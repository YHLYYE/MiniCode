"""恢复路径可观测性 —— 每条恢复路径触发时都要发一个 RecoveryNotice。

背景：`LoopState.transition` 以前是**只写不读**的死状态，而 5 条恢复路径里有
3 条（网络重试、工具报错、空响应）跟"正常继续"一样写 NEXT_TURN，根本分不出来。
结果就是：恢复路径对用户完全静默 —— 连"系统刚救过一次场"都不知道。

现在每条恢复路径都会设一个**不同的** ContinueReason，并发一个 RecoveryNotice 事件。
"""
import pytest

from core import agent_loop as al
from core.agent_loop import AgentLoop
from core.state import ContinueReason, DoneEvent, RecoveryNotice
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
    input_tokens = 10
    output_tokens = 5


def _text(t):
    return _Chunk(type="text_delta", text=t)


def _stop(reason="end_turn"):
    return _Chunk(type="message_stop", stop_reason=reason, usage=_Usage())


def _tool(name, **kwargs):
    return _Chunk(type="tool_use_start", name=name, input=kwargs,
                  tool_call_id="c1")


class _ScriptedModel:
    """按脚本行动：每一项要么是要抛的异常，要么是一串 chunk。"""

    model = "mock"
    max_output_tokens = 8192

    def __init__(self, script):
        self.script = script
        self.idx = 0
        self.calls = 0

    async def chat_streaming(self, messages, system="", tools=None,
                             max_tokens=None):
        self.calls += 1
        item = self.script[self.idx] if self.idx < len(self.script) else [_stop()]
        self.idx += 1
        if isinstance(item, Exception):
            raise item
        for c in item:
            yield c


async def _run(model, tools=None, **kw):
    loop = AgentLoop(tools=tools or [], model_adapter=model,
                     system_prompt="test", **kw)
    events = [e async for e in loop.run("task")]
    return [e for e in events if isinstance(e, RecoveryNotice)], events


@pytest.mark.asyncio
async def test_prompt_too_long_emits_notice():
    model = _ScriptedModel([
        RuntimeError("prompt is too long: 70000 tokens"),
        [_text("ok"), _stop()],
    ])
    notices, _ = await _run(model)
    assert [n.reason for n in notices] == [ContinueReason.PROMPT_TOO_LONG_RETRY]


@pytest.mark.asyncio
async def test_stream_retry_emits_notice(monkeypatch):
    monkeypatch.setattr(al, "_STREAM_RETRY_BASE_DELAY", 0)   # 别真等 1 秒
    model = _ScriptedModel([
        RuntimeError("connection reset by peer"),
        [_text("ok"), _stop()],
    ])
    notices, _ = await _run(model)
    assert [n.reason for n in notices] == [ContinueReason.STREAM_RETRY]


class _BoomTool(Tool):
    name = "boom"
    input_schema = {}
    is_readonly = True

    async def execute(self):
        raise RuntimeError("工具炸了")


@pytest.mark.asyncio
async def test_tool_error_emits_notice():
    model = _ScriptedModel([
        [_tool("boom"), _stop()],
        [_text("换个方案"), _stop()],
    ])
    notices, _ = await _run(model, tools=[_BoomTool()])
    assert [n.reason for n in notices] == [ContinueReason.TOOL_ERROR_RETRY]


@pytest.mark.asyncio
async def test_empty_response_emits_notice():
    model = _ScriptedModel([
        [_stop()],                      # 空响应
        [_text("补上"), _stop()],
    ])
    notices, _ = await _run(model)
    assert [n.reason for n in notices] == [ContinueReason.EMPTY_RESPONSE_RETRY]


@pytest.mark.asyncio
async def test_max_tokens_upgrade_emits_notice():
    model = _ScriptedModel([
        [_text("被截断的半句"), _stop("length")],
        [_text("续上"), _stop()],
    ])
    notices, _ = await _run(model)
    assert [n.reason for n in notices] == [
        ContinueReason.MAX_OUTPUT_TOKENS_UPGRADE]


@pytest.mark.asyncio
async def test_normal_run_emits_no_notice():
    """正常跑完不该有任何恢复通知 —— 否则提示会变成噪声。"""
    model = _ScriptedModel([[_text("一切正常"), _stop()]])
    notices, events = await _run(model)
    assert notices == []
    assert any(isinstance(e, DoneEvent) for e in events)


def test_every_recovery_reason_has_a_label():
    """每个「非正常继续」的原因，终端都得能翻译成人话。

    以后往 ContinueReason 里加值时忘了加标签，这条会先红。
    """
    import main

    missing = [r for r in ContinueReason
               if r is not ContinueReason.NEXT_TURN and r not in main._RECOVERY_LABELS]
    assert missing == [], f"这些原因没有可读标签：{[r.value for r in missing]}"


def test_every_recovery_reason_has_a_notice_path():
    """每个恢复原因都应当有代码在设它 —— 防止留下"定义了但没人用"的枚举值。"""
    import inspect

    src = inspect.getsource(al)
    unused = [r for r in ContinueReason
              if r is not ContinueReason.NEXT_TURN
              and f"ContinueReason.{r.name}" not in src]
    assert unused == [], f"这些原因在 agent_loop 里没人设：{[r.value for r in unused]}"
