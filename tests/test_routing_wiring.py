"""二阶段路由接线测试 —— 路由结果真的进了 System Prompt 的动态段。

背景：SkillSystem.route() 早就实现了，但此前唯一的调用点是 Skill 名打错时
给一句「最接近的推荐」，等于没有路由。现在每轮任务会按当前任务跑一次
route()，把 top-k 候选注入 System Prompt 的末尾。
"""
from pathlib import Path

import pytest

from capabilities.skill import SkillSystem
from core.agent_loop import AgentLoop, _with_system_prompt
from core.state import LoopState, Message
from prompt.system_prompt import build_system_prompt


SEP = "\n---DYNAMIC---\n"


def _ss() -> SkillSystem:
    s = SkillSystem()
    s.register_from_source(Path(__file__).parent.parent / "skills", priority=10)
    return s


# ── routing_hint 本身 ──

def test_hint_empty_for_irrelevant_task():
    assert _ss().routing_hint("量子计算的哈密顿量怎么对角化") == ""
    assert _ss().routing_hint("今天天气怎么样") == ""


def test_hint_present_for_relevant_task():
    hint = _ss().routing_hint("帮我审查一下这段代码有没有bug")
    assert "## Task Routing" in hint
    assert "code-review" in hint


def test_hint_respects_top_k():
    hint = _ss().routing_hint("review this code for bugs", top_k=2)
    bullets = [ln for ln in hint.splitlines() if ln.startswith("- ")]
    assert 1 <= len(bullets) <= 2


def test_hint_confidence_floor_filters_noise():
    """低于阈值的噪声候选不该出现在提示里。"""
    hint = _ss().routing_hint("把这个函数重构一下", top_k=3)
    assert "refactor" in hint
    assert "debug" not in hint


def test_hint_empty_when_no_skills_registered():
    assert SkillSystem().routing_hint("review this code") == ""


def test_hint_covers_chinese_debug_phrasing():
    """中文调试类问法不该被路由到写测试。"""
    hint = _ss().routing_hint("为什么这个测试挂了，帮我查一下", top_k=1)
    assert "debug" in hint


# ── 注入位置：必须压在动态段最末尾 ──

def test_hint_is_appended_last():
    ss = _ss()
    prompt = build_system_prompt(
        skill_index=ss.get_index_for_system_prompt(),
        claude_md="project conventions",
        routing_hint=ss.routing_hint("检查一下有没有安全漏洞"),
    )
    dynamic = prompt.split(SEP)[1]
    assert "security-audit" in dynamic
    assert dynamic.rstrip().endswith(
        'Load one with the Skill tool: name="<skill-name>".'
    )


def test_static_prefix_identical_across_tasks():
    """核心性质：只有路由块变化，它之前的所有内容逐字节一致 → 前缀缓存可用。"""
    ss = _ss()
    index = ss.get_index_for_system_prompt()
    p1 = build_system_prompt(index, "ctx", ss.routing_hint("帮我审查这段代码有没有bug"))
    p2 = build_system_prompt(index, "ctx", ss.routing_hint("给这个模块补单元测试"))

    assert p1 != p2
    assert p1.split(SEP)[0] == p2.split(SEP)[0], "静态段被改动了"

    before1, _, tail1 = p1.split(SEP)[1].partition("## Task Routing")
    before2, _, tail2 = p2.split(SEP)[1].partition("## Task Routing")
    assert before1 == before2, "路由块之前的内容被改动了"
    assert tail1 != tail2


def test_no_routing_block_when_hint_empty():
    ss = _ss()
    prompt = build_system_prompt(ss.get_index_for_system_prompt(), "", "")
    assert "## Task Routing" not in prompt


# ── AgentLoop 侧：每任务重建 ──

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

    def __init__(self, turns):
        self.turns = turns
        self.idx = 0

    async def chat_streaming(self, messages, system="", tools=None,
                             max_tokens=None):
        text = "done" if self.idx >= len(self.turns) else self.turns[self.idx]
        self.idx += 1
        yield _Chunk(type="text_delta", text=text)
        yield _Chunk(type="message_stop", stop_reason="end_turn", usage=_Usage())


@pytest.mark.asyncio
async def test_loop_rebuilds_prompt_per_task_and_preserves_history():
    captured = []

    def factory(task):
        captured.append(task)
        return f"SYSTEM<{task}>"

    loop = AgentLoop(tools=[], model_adapter=_MockModel(["a", "b"]),
                     system_prompt="STATIC", system_prompt_factory=factory)

    async for _ in loop.run("first task"):
        pass
    async for _ in loop.run("second task"):
        pass

    assert captured == ["first task", "second task"]
    messages = loop.state.messages
    assert messages[0].role == "system"
    assert messages[0].content == "SYSTEM<second task>"
    assert [m.content for m in messages if m.role == "user"] == [
        "first task", "second task"
    ]


@pytest.mark.asyncio
async def test_loop_without_factory_uses_constructor_prompt():
    loop = AgentLoop(tools=[], model_adapter=_MockModel(["a"]),
                     system_prompt="STATIC")
    async for _ in loop.run("task"):
        pass
    assert loop.state.messages[0].content == "STATIC"


@pytest.mark.asyncio
async def test_routing_failure_never_blocks_the_task():
    """路由炸了也必须能跑完 —— 回退到构造时那份 prompt。"""
    def broken(task):
        raise RuntimeError("routing blew up")

    loop = AgentLoop(tools=[], model_adapter=_MockModel(["a"]),
                     system_prompt="STATIC", system_prompt_factory=broken)
    events = [e async for e in loop.run("task")]
    assert loop.state.messages[0].content == "STATIC"
    assert events


def test_with_system_prompt_prepends_when_missing():
    state = LoopState(messages=(Message(role="user", content="hi"),))
    out = _with_system_prompt(state, "SYS")
    assert [m.role for m in out.messages] == ["system", "user"]
