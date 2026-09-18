"""Agent Loop unit tests — using Mock LLM adapter"""
import asyncio
import pytest
from core.agent_loop import AgentLoop
from core.state import (
    LoopState, Message, ContinueReason,
    TextDelta, ToolStart, ToolResult, ToolError, DoneEvent,
    BudgetExceeded,
)
from core.tools.base import Tool, ToolCall
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
    from core.agent_loop import BudgetExceeded

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
