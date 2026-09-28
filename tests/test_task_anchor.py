"""TodoWrite 作为「长任务锚点」的接线测试。

此前四处断线：
1. System Prompt 里一个字都没提这个工具
2. Plan 模式把它一起移除了 —— 只读规划模式里连待办清单都写不了
3. Snip 会把它替换成占位符，而 _cache 只写不读 → 清单永久消失
4. Autocompact 后会丢掉进度
"""
import asyncio
import pytest

from capabilities.compression import ContextCompressor
from capabilities.security import ExecutionMode, resolve_tools_for_mode
from core.agent_loop import AgentLoop
from core.state import DoneEvent, LoopState, Message
from core.tools.task import TodoWriteTool
from prompt.system_prompt import build_system_prompt


TASK_LIST = "## Task List\n☐ 1. 读 auth.py\n◉ 2. 抽出 token 校验\n☐ 3. 补测试"


def _long_session(tool_count: int = 8) -> LoopState:
    msgs = [
        Message(role="system", content="sys"),
        Message(role="user", content="重构认证模块"),
        Message(role="assistant", content="", tool_calls=[{"id": "t0"}]),
        Message(role="tool", content=TASK_LIST, tool_call_id="t0"),
    ]
    for i in range(1, tool_count + 1):
        msgs.append(Message(role="assistant", content="", tool_calls=[{"id": f"t{i}"}]))
        msgs.append(Message(role="tool", content=f"文件块 {i} " + "x" * 3000,
                            tool_call_id=f"t{i}"))
    return LoopState(messages=tuple(msgs))


# ── 1. System Prompt 里要提到这个工具 ──

def test_system_prompt_mentions_todowrite():
    prompt = build_system_prompt()
    assert "TodoWrite" in prompt


# ── 2. Plan 模式要放行 ──

def test_plan_mode_keeps_todowrite_but_not_write():
    class _T:
        def __init__(self, name, readonly):
            self.name = name
            self.is_readonly = readonly

    tools = [_T("Read", True), _T("Write", False), _T("TodoWrite", False),
             _T("Bash", False)]
    kept = {t.name for t in resolve_tools_for_mode(tools, ExecutionMode.PLAN)}
    assert kept == {"Read", "TodoWrite"}


# ── 3. Snip 不能吃掉任务清单 ──

def test_snip_preserves_task_list():
    c = ContextCompressor()
    c.record_task_list(TASK_LIST)
    out = asyncio.run(c._snip(_long_session()))

    joined = "".join(m.content for m in out.messages)
    assert "抽出 token 校验" in joined, "任务清单被 Snip 吃掉了"
    # 其他旧工具结果照常被压
    assert "[[snip:" in joined


def test_snip_still_snips_when_no_task_list():
    """没写过 todo 时行为不变。"""
    c = ContextCompressor()
    out = asyncio.run(c._snip(_long_session()))
    assert "抽出 token 校验" not in "".join(m.content for m in out.messages)


# ── 4. Autocompact 之后要回灌 ──

def test_autocompact_restores_task_list():
    c = ContextCompressor()
    c.record_task_list(TASK_LIST)
    out = asyncio.run(c._autocompact(_long_session()))
    joined = "".join(m.content for m in out.messages)
    assert "[Current task list]" in joined
    assert "抽出 token 校验" in joined
    assert len(out.messages) < 5


def test_force_autocompact_restores_task_list():
    c = ContextCompressor()
    c.record_task_list(TASK_LIST)
    out = asyncio.run(c.force_autocompact(_long_session()))
    joined = "".join(m.content for m in out.messages)
    assert "[Current task list]" in joined
    assert "抽出 token 校验" in joined


# ── 5. 端到端：loop 执行 TodoWrite 时会登记 ──

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


class _MockModel:
    model = "mock"
    max_output_tokens = 8192

    def __init__(self, plan):
        self.plan = plan
        self.idx = 0

    async def chat_streaming(self, messages, system="", tools=None,
                             max_tokens=None):
        if self.idx >= len(self.plan):
            yield _Chunk(type="text_delta", text="done")
            yield _Chunk(type="message_stop", stop_reason="end_turn",
                         usage=_Usage())
            return
        for c in self.plan[self.idx]:
            yield c
        self.idx += 1


@pytest.mark.asyncio
async def test_loop_registers_task_list_after_todowrite():
    plan = [
        [
            _Chunk(type="tool_use_start", name="TodoWrite",
                   input={"todos": [{"content": "读 auth.py", "status": "pending"}]},
                   tool_call_id="c1"),
            _Chunk(type="message_stop", stop_reason="end_turn", usage=_Usage()),
        ],
        [
            _Chunk(type="text_delta", text="done"),
            _Chunk(type="message_stop", stop_reason="end_turn", usage=_Usage()),
        ],
    ]
    loop = AgentLoop(tools=[TodoWriteTool()], model_adapter=_MockModel(plan),
                     system_prompt="test")
    events = [e async for e in loop.run("go")]

    assert any(isinstance(e, DoneEvent) for e in events)
    assert loop._compressor._task_list.startswith("## Task List")
    assert "读 auth.py" in loop._compressor._task_list
