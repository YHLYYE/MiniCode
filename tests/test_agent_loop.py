"""Agent Loop unit tests — using Mock LLM adapter"""
import pytest
from core.agent_loop import AgentLoop
from core.state import (
    LoopState, Message, TextDelta, ToolResult, DoneEvent, BudgetExceeded,
)
from core.tools.base import Tool
from capabilities.memory import MemoryManager, MemoryType


# ── Mock Tool ──

class MockTool(Tool):
    """A search tool that returns predefined results"""
    name = "mock_search"
    description = "Search for text patterns"
    input_schema = {"query": {"type": "string"}}
    is_readonly = True
    is_concurrency_safe = True

    def __init__(self, results: list[str] | None = None):
        self.results = results or ["found: test_function in app.py:42"]
        self.call_count = 0

    async def execute(self, query: str) -> str:
        idx = min(self.call_count, len(self.results) - 1)
        self.call_count += 1
        return self.results[idx]


# ── Mock Model Adapter ──

class MockStreamChunk:
    """Lightweight chunk that mimics core.model_adapter.StreamChunk"""
    def __init__(self, type, text="", name="", input=None, stop_reason="", usage=None, tool_call_id=""):
        self.type = type
        self.text = text
        self.name = name
        self.input = input
        self.stop_reason = stop_reason
        self.usage = usage
        self.tool_call_id = tool_call_id


class MockUsage:
    def __init__(self, input_tokens=100, output_tokens=50):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cache_read_tokens = 0
        self.cache_write_tokens = 0


class MockModelAdapter:
    """Returns predetermined response sequences"""
    model = "mock"
    max_output_tokens = 8192

    def __init__(self, response_plan: list[list]):
        """
        response_plan: list of lists of MockStreamChunk
        Each inner list = one API call's chunk sequence
        """
        self.response_plan = response_plan
        self.call_index = 0

    async def chat_streaming(self, messages, system="", tools=None, max_tokens=None):
        if self.call_index >= len(self.response_plan):
            yield MockStreamChunk(type="text_delta", text="Done.")
            yield MockStreamChunk(
                type="message_stop", stop_reason="end_turn",
                usage=MockUsage()
            )
            return

        for chunk in self.response_plan[self.call_index]:
            yield chunk
        self.call_index += 1


# ── Helper to make StreamChunk-compatible objects ──

def make_text(text: str):
    return MockStreamChunk(type="text_delta", text=text)

def make_tool_use(name: str, input: dict | None = None, tool_call_id: str = ""):
    chunk = MockStreamChunk(
        type="tool_use_start", name=name, input=input or {}
    )
    chunk.tool_call_id = tool_call_id or f"call_test_{name}"
    return chunk

def make_stop(reason="end_turn", input_tokens=100, output_tokens=50):
    return MockStreamChunk(
        type="message_stop", stop_reason=reason,
        usage=MockUsage(input_tokens, output_tokens)
    )


# ── Tests ──

@pytest.mark.asyncio
async def test_agent_loop_simple_answer():
    """Single turn: model answers without calling tools → loop exits"""
    mock = MockModelAdapter([
        [make_text("Hello! How can I help?"), make_stop("end_turn")]
    ])

    loop = AgentLoop(
        tools=[MockTool()],
        model_adapter=mock,
        system_prompt="You are a helpful assistant.",
    )

    events = []
    async for event in loop.run("Hi"):
        events.append(event)

    # Should have text and DoneEvent
    texts = [e.text for e in events if isinstance(e, TextDelta)]
    assert "Hello!" in "".join(texts)
    assert any(isinstance(e, DoneEvent) for e in events)
    done = [e for e in events if isinstance(e, DoneEvent)][0]
    assert done.state.turn_count == 1  # single turn (one API call)


@pytest.mark.asyncio
async def test_agent_loop_with_tool_call():
    """Model calls a tool, agent executes it, model responds → completes"""
    mock = MockModelAdapter([
        # Turn 1: model calls a tool
        [
            make_tool_use("mock_search", {"query": "test_function"}),
            make_stop("end_turn"),
        ],
        # Turn 2: model returns final answer
        [
            make_text("Found test_function at line 42"),
            make_stop("end_turn"),
        ],
    ])

    tool = MockTool()
    loop = AgentLoop(
        tools=[tool],
        model_adapter=mock,
        system_prompt="You are a test agent.",
    )

    events = []
    async for event in loop.run("Find test_function"):
        events.append(event)

    # Tool should have been called
    assert tool.call_count == 1

    # Should have a ToolResult event
    tool_results = [e for e in events if isinstance(e, ToolResult)]
    assert len(tool_results) == 1
    assert "test_function" in tool_results[0].output

    # Should complete
    assert any(isinstance(e, DoneEvent) for e in events)


@pytest.mark.asyncio
async def test_agent_loop_tool_not_found():
    """Model calls a tool that doesn't exist → error injected as tool result"""
    mock = MockModelAdapter([
        [
            make_tool_use("nonexistent_tool", {"arg": "val"}),
            make_stop("end_turn"),
        ],
        # After error, model responds
        [
            make_text("Sorry, that tool doesn't exist."),
            make_stop("end_turn"),
        ],
    ])

    loop = AgentLoop(
        tools=[MockTool()],
        model_adapter=mock,
        system_prompt="Test agent.",
    )

    events = []
    async for event in loop.run("Try invalid tool"):
        events.append(event)

    # Should get error message in tool result
    tool_results = [e for e in events if isinstance(e, ToolResult)]
    assert any("not found" in r.output for r in tool_results)

    # Should still complete
    assert any(isinstance(e, DoneEvent) for e in events)


@pytest.mark.asyncio
async def test_unknown_tool_is_recorded_in_memory_like_any_other_failure(tmp_path):
    """工具名不存在也算失败，要和其他工具错误一样沉淀进情景记忆。

    回归背景：这条路径是提前 return 的，绕过了 tool_errors 收集 —— 于是
    情景记忆里只留下"任务完成"，复盘时看不出这轮是因为工具名写错才绕路。
    """
    mm = MemoryManager(project_root=tmp_path)
    mock = MockModelAdapter([
        [make_tool_use("nonexistent_tool", {"arg": "val"}), make_stop("end_turn")],
        [make_text("改用已有工具"), make_stop("end_turn")],
    ])
    loop = AgentLoop(tools=[MockTool()], model_adapter=mock,
                     system_prompt="Test agent.", memory_manager=mm)
    async for _ in loop.run("call a missing tool"):
        pass

    hits = await mm.search("Tool not found", top_k=5,
                           memory_type=MemoryType.EPISODIC)
    assert any("Tool not found" in h.content for h in hits), (
        "未知工具的失败没有进情景记忆"
    )
    mm.close()


@pytest.mark.asyncio
async def test_agent_loop_tool_execution_error():
    """Tool raises an exception → error captured, agent continues"""
    class FailingTool(Tool):
        name = "failing_tool"
        description = "Always fails"
        input_schema = {"arg": {"type": "string"}}
        is_readonly = True
        async def execute(self, arg: str) -> str:
            raise RuntimeError("Simulated failure")

    mock = MockModelAdapter([
        [
            make_tool_use("failing_tool", {"arg": "test"}),
            make_stop("end_turn"),
        ],
        [
            make_text("Tool failed, trying alternative..."),
            make_stop("end_turn"),
        ],
    ])

    loop = AgentLoop(
        tools=[FailingTool()],
        model_adapter=mock,
        system_prompt="Test agent.",
    )

    events = []
    async for event in loop.run("Test failure recovery"):
        events.append(event)

    # Should have ToolResult with error
    tool_results = [e for e in events if isinstance(e, ToolResult)]
    assert any("Simulated failure" in r.output or "Tool error" in r.output
               for r in tool_results)

    # Agent should continue after error (not crash)
    assert any(isinstance(e, DoneEvent) for e in events)


@pytest.mark.asyncio
async def test_agent_loop_multi_turn():
    """Multiple tool calls across turns → all executed"""
    mock = MockModelAdapter([
        # Turn 1
        [make_tool_use("mock_search", {"query": "step1"}), make_stop("end_turn")],
        # Turn 2
        [make_tool_use("mock_search", {"query": "step2"}), make_stop("end_turn")],
        # Turn 3 — final answer
        [make_text("All steps done."), make_stop("end_turn")],
    ])

    tool = MockTool(results=["step1 result", "step2 result"])
    loop = AgentLoop(
        tools=[tool],
        model_adapter=mock,
        system_prompt="Test agent.",
    )

    events = []
    async for event in loop.run("Multi-step task"):
        events.append(event)

    # Both tool calls executed
    assert tool.call_count == 2

    # All results present
    tool_results = [e for e in events if isinstance(e, ToolResult)]
    assert len(tool_results) == 2

    # Completed
    assert any(isinstance(e, DoneEvent) for e in events)


@pytest.mark.asyncio
async def test_loop_state_immutability():
    """LoopState methods return new instances (frozen dataclass)"""
    state = LoopState()
    state2 = state.add_message(Message(role="user", content="test"))

    assert state is not state2
    assert len(state.messages) == 0
    assert len(state2.messages) == 1
    assert state2.messages[0].role == "user"
    assert state2.messages[0].content == "test"


@pytest.mark.asyncio
async def test_loop_state_accumulate_usage():
    """Token accumulation tracks cost correctly"""
    from core.model_adapter import Usage  # real Usage type

    state = LoopState()
    usage = Usage(input_tokens=1000, output_tokens=500)
    state2 = state.accumulate_usage(usage)

    assert state2.total_tokens == 1500
    assert state2.total_cost_usd > 0  # should be non-zero
    # 1000 * 3/1M + 500 * 15/1M = 0.0105
    assert 0.01 <= state2.total_cost_usd <= 0.02


@pytest.mark.asyncio
async def test_budget_exceeded():
    """Agent stops when budget is exceeded"""
    mock = MockModelAdapter([
        [
            make_text("Doing expensive work..."),
            make_stop("end_turn", input_tokens=500_000, output_tokens=500_000),
        ],
    ])

    # Set budget very low
    loop = AgentLoop(
        tools=[MockTool()],
        model_adapter=mock,
        system_prompt="Test agent.",
        max_cost_usd=0.001,  # $0.001 — will be exceeded immediately
    )

    with pytest.raises(BudgetExceeded):
        async for event in loop.run("Expensive task"):
            pass


# ── Memory write-loop integration tests ──

@pytest.mark.asyncio
async def test_memory_records_task_completion(tmp_path):
    """Task DoneEvent persists a completion event to episodic memory."""
    mm = MemoryManager(project_root=tmp_path)
    mock = MockModelAdapter([[make_text("All done."), make_stop("end_turn")]])

    loop = AgentLoop(
        tools=[MockTool()],
        model_adapter=mock,
        system_prompt="Test.",
        memory_manager=mm,
    )
    async for _ in loop.run("Do a thing"):
        pass

    results = await mm.search("Do a thing", memory_type=MemoryType.EPISODIC)
    assert any("Task:" in e.content for e in results), [e.content for e in results]


@pytest.mark.asyncio
async def test_memory_records_tool_error(tmp_path):
    """Tool errors are persisted to episodic memory during recovery."""
    class FailingTool(Tool):
        name = "failing_tool"
        description = "Always fails"
        input_schema = {"arg": {"type": "string"}}
        is_readonly = True
        async def execute(self, arg: str) -> str:
            raise RuntimeError("distinctive_boom_xyz")

    mm = MemoryManager(project_root=tmp_path)
    mock = MockModelAdapter([
        [make_tool_use("failing_tool", {"arg": "x"}), make_stop("end_turn")],
        [make_text("Alternative."), make_stop("end_turn")],
    ])

    loop = AgentLoop(
        tools=[FailingTool()],
        model_adapter=mock,
        system_prompt="Test.",
        memory_manager=mm,
    )
    async for _ in loop.run("test"):
        pass

    results = await mm.search("distinctive_boom_xyz",
                              memory_type=MemoryType.EPISODIC)
    assert any("distinctive_boom_xyz" in e.content for e in results), \
        [e.content for e in results]


@pytest.mark.asyncio
async def test_memory_records_file_edit(tmp_path):
    """Successful Write persists a file-edit event to episodic memory."""
    from pathlib import Path

    class WriteMockTool(Tool):
        name = "Write"
        description = "Mock write"
        input_schema = {
            "file_path": {"type": "string"},
            "content": {"type": "string"},
        }
        is_readonly = False
        async def execute(self, file_path: str, content: str) -> str:
            return f"Created {file_path} ({len(content)} chars)"

    # Path must be inside cwd to pass the security path allowlist
    file_path = str(Path.cwd() / "memtest_unique_xyz.py")

    mm = MemoryManager(project_root=tmp_path)
    mock = MockModelAdapter([
        [make_tool_use("Write", {"file_path": file_path, "content": "x=1"}),
         make_stop("end_turn")],
        [make_text("Done."), make_stop("end_turn")],
    ])

    loop = AgentLoop(
        tools=[WriteMockTool()],
        model_adapter=mock,
        system_prompt="Test.",
        memory_manager=mm,
    )
    async for _ in loop.run("edit file"):
        pass

    results = await mm.search("memtest_unique_xyz",
                              memory_type=MemoryType.EPISODIC)
    assert any("Edited" in e.content for e in results), [e.content for e in results]


# ── 自沉淀的"提炼"这一环：错误 → 修法配对 ──
# 回归背景：简历上写着"错误→修法自动提炼"，但这条能力此前**没有任何测试**。

class _BoomTool(Tool):
    name = "Boom"
    input_schema = {"x": {"type": "integer"}}

    async def execute(self, x: int = 0) -> str:
        raise RuntimeError("boom: 故意的")


class _WriteStub(Tool):
    name = "Write"
    input_schema = {"file_path": {"type": "string"},
                    "content": {"type": "string"}}
    is_readonly = False

    async def execute(self, file_path: str, content: str = "") -> str:
        return f"Updated {file_path}"


@pytest.mark.asyncio
async def test_distills_error_and_fix_into_procedural_memory(tmp_path):
    """先报错、再成功改文件 → 应该提炼出一条「错误 → 修法」的程序性记忆。

    这是零模型调用的：错误列表和成功的编辑本来就在循环里。
    """
    mock = MockModelAdapter([
        [make_tool_use("Boom", {"x": 1}), make_stop("end_turn")],
        [make_tool_use("Write", {"file_path": "demo.py", "content": "print(1)"}),
         make_stop("end_turn")],
        [make_text("修好了"), make_stop("end_turn")],
    ])
    mm = MemoryManager(project_root=tmp_path)
    loop = AgentLoop(tools=[_BoomTool(), _WriteStub()], model_adapter=mock,
                     system_prompt="Test.", memory_manager=mm)
    async for _ in loop.run("让工具炸一次再修好"):
        pass

    hits = await mm.search("boom", memory_type=MemoryType.PROCEDURAL)
    assert hits, "没有提炼出任何程序性记忆"
    text = hits[0].content
    assert "→ 修法：" in text, text
    assert "demo.py" in text, text
    mm.close()


@pytest.mark.asyncio
async def test_no_distillation_without_a_successful_fix(tmp_path):
    """只报错、没修好 → 不许写经验（否则记忆里全是噪声）。"""
    mock = MockModelAdapter([
        [make_tool_use("Boom", {"x": 1}), make_stop("end_turn")],
        [make_text("算了"), make_stop("end_turn")],
    ])
    mm = MemoryManager(project_root=tmp_path)
    loop = AgentLoop(tools=[_BoomTool()], model_adapter=mock,
                     system_prompt="Test.", memory_manager=mm)
    async for _ in loop.run("炸一次就算了"):
        pass

    hits = await mm.search("boom", memory_type=MemoryType.PROCEDURAL)
    assert hits == [], f"没修好却写了经验：{[h.content for h in hits]}"
    mm.close()


# ── 用户纠正的自动沉淀（不可从代码推导的信息）──
# 现有自动沉淀（工具报错 / 任务完成 / 错误→修法）都是模型自己也能从仓库推出来的信息，
# 消融四版都测不出显著收益，原因就在这。用户纠正/团队约定才是真正值钱的那类。

def _plain_plan():
    return [[make_text("好的"), make_stop("end_turn")]]


@pytest.mark.asyncio
async def test_user_correction_is_sedimented_automatically(tmp_path):
    """有历史时，用户的纠正意见应该自动落成程序性记忆。"""
    mm = MemoryManager(project_root=tmp_path)
    loop = AgentLoop(tools=[], model_adapter=MockModelAdapter(_plain_plan()),
                     system_prompt="Test.", memory_manager=mm)
    async for _ in loop.run("先看看这个仓库"):
        pass
    # 第二轮：用户纠正
    loop2_model = MockModelAdapter(_plain_plan())
    loop2 = AgentLoop(tools=[], model_adapter=loop2_model,
                      system_prompt="Test.", memory_manager=mm)
    loop2._state = loop.state          # 模拟 REPL：接着上一轮
    async for _ in loop2.run("以后不要用 Tab 缩进"):
        pass

    hits = await mm.search("Tab", memory_type=MemoryType.PROCEDURAL)
    assert any("用户纠正" in h.content for h in hits), [h.content for h in hits]
    mm.close()


@pytest.mark.asyncio
async def test_first_message_is_never_treated_as_a_correction(tmp_path):
    """第一句任务里出现"不要"也不算纠正 —— 纠正按定义针对上一轮。"""
    mm = MemoryManager(project_root=tmp_path)
    loop = AgentLoop(tools=[], model_adapter=MockModelAdapter(_plain_plan()),
                     system_prompt="Test.", memory_manager=mm)
    async for _ in loop.run("不要用 Tab，帮我检查这个文件"):
        pass
    hits = await mm.search("Tab", memory_type=MemoryType.PROCEDURAL)
    assert not any("用户纠正" in h.content for h in hits), [h.content for h in hits]
    mm.close()


@pytest.mark.asyncio
async def test_long_message_with_marker_is_not_a_correction(tmp_path):
    """长描述里带"不要"多半是任务，不是纠正（长度上限就是防这个）。"""
    mm = MemoryManager(project_root=tmp_path)
    loop = AgentLoop(tools=[], model_adapter=MockModelAdapter(_plain_plan()),
                     system_prompt="Test.", memory_manager=mm)
    async for _ in loop.run("先看看这个仓库"):
        pass
    long_task = "帮我重构这个模块，注意不要破坏现有接口，" + "细节" * 120
    loop2 = AgentLoop(tools=[], model_adapter=MockModelAdapter(_plain_plan()),
                      system_prompt="Test.", memory_manager=mm)
    loop2._state = loop.state
    async for _ in loop2.run(long_task):
        pass
    hits = await mm.search("破坏现有接口", memory_type=MemoryType.PROCEDURAL)
    assert not any("用户纠正" in h.content for h in hits), [h.content for h in hits]
    mm.close()
