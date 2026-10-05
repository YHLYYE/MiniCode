"""前缀缓存命中量必须真的被读出来。

此前 `Usage.cache_read_tokens` / `cache_write_tokens` 只是两个声明——
全项目没有一行代码给它们赋值，也没有一行读它们。结果是简历/文档里那句
"布局对前缀缓存友好、PREFIX 逐字节一致"**无法自证**：布局是照着缓存友好写的，
但到底命中多少，自己代码里看不见。

现在 `ModelAdapter._usage_from()` 把 provider 报的缓存计数拍平进来，
`LoopState.accumulate_usage()` 逐轮累加，`main.py` 在收尾那行打出来。
"""
import pytest

from core.model_adapter import ModelAdapter, Usage
from core.state import LoopState


class _Details:
    def __init__(self, cached):
        self.cached_tokens = cached


class _OpenAIUsage:
    """OpenAI 兼容（DeepSeek 同协议）：缓存在 prompt_tokens_details.cached_tokens"""

    def __init__(self, prompt=1000, completion=200, cached=800):
        self.prompt_tokens = prompt
        self.completion_tokens = completion
        self.prompt_tokens_details = _Details(cached)


class _AnthropicUsage:
    """Anthropic：走 cache_read_input_tokens / cache_creation_input_tokens"""

    def __init__(self, read=700, write=120):
        self.prompt_tokens = 900
        self.completion_tokens = 150
        self.cache_read_input_tokens = read
        self.cache_creation_input_tokens = write


class _PlainUsage:
    """只有基础字段的 provider —— 不能因此炸"""

    prompt_tokens = 500
    completion_tokens = 60


def test_usage_from_openai_shape():
    u = ModelAdapter._usage_from(_OpenAIUsage())
    assert u["input_tokens"] == 1000
    assert u["output_tokens"] == 200
    assert u["cache_read_tokens"] == 800
    assert u["cache_write_tokens"] == 0


def test_usage_from_anthropic_shape():
    u = ModelAdapter._usage_from(_AnthropicUsage())
    assert u["cache_read_tokens"] == 700
    assert u["cache_write_tokens"] == 120


def test_usage_from_plain_shape_does_not_guess():
    u = ModelAdapter._usage_from(_PlainUsage())
    assert u["input_tokens"] == 500
    assert u["cache_read_tokens"] == 0
    assert u["cache_write_tokens"] == 0


def test_accumulate_usage_adds_cache_counters():
    state = LoopState().accumulate_usage(Usage(
        input_tokens=1000, output_tokens=200,
        cache_read_tokens=800, cache_write_tokens=50,
    ))
    assert state.cache_read_tokens == 800
    assert state.cache_write_tokens == 50
    # 累加而不是覆盖
    state = state.accumulate_usage(Usage(
        input_tokens=10, output_tokens=5, cache_read_tokens=20,
    ))
    assert state.cache_read_tokens == 820
    assert state.total_tokens == 1215


@pytest.mark.asyncio
async def test_loop_carries_cache_counters_to_the_end():
    """端到端：一轮里 provider 报了缓存命中，收尾的 state 必须带着它。"""
    from core.agent_loop import AgentLoop
    from core.state import DoneEvent

    class _Chunk:
        def __init__(self, type, text="", stop_reason="", usage=None):
            self.type = type
            self.text = text
            self.stop_reason = stop_reason
            self.usage = usage
            self.name = ""
            self.input = None
            self.tool_call_id = ""

    class _Model:
        model = "mock"
        max_output_tokens = 8192

        async def chat_streaming(self, messages, system="", tools=None,
                                 max_tokens=None):
            yield _Chunk("text_delta", text="ok")
            yield _Chunk("message_stop", stop_reason="end_turn", usage=Usage(
                input_tokens=1000, output_tokens=200,
                cache_read_tokens=800, cache_write_tokens=50,
            ))

    loop = AgentLoop(tools=[], model_adapter=_Model(), system_prompt="t")
    events = [e async for e in loop.run("go")]
    done = [e for e in events if isinstance(e, DoneEvent)]
    assert done
    assert done[0].state.cache_read_tokens == 800
    assert done[0].state.cache_write_tokens == 50


def test_session_store_round_trips_cache_counters(tmp_path):
    """存档要带上缓存计数，否则 --resume 之后这个数会凭空归零。"""
    from session_store import SessionStore
    from core.state import Message

    store = SessionStore(tmp_path)
    state = LoopState(
        messages=(Message("user", "hi"),),
        total_tokens=1000, cache_read_tokens=800, cache_write_tokens=50,
    )
    sid = store.save(state)
    loaded = store.load(sid)

    assert loaded.cache_read_tokens == 800
    assert loaded.cache_write_tokens == 50
