import pytest

from capabilities.verification import VerificationResult
from core.agent_loop import AgentLoop
from core.state import DoneEvent, TerminationReason
from core.tools.base import Tool
from tests.test_agent_loop import (
    MockModelAdapter, make_stop, make_tool_use, make_text
)


class FailingThenPassingVerifier:
    def __init__(self):
        self.calls = 0

    async def verify(self):
        self.calls += 1
        return VerificationResult(
            command="fake",
            passed=self.calls > 1,
            exit_code=1 if self.calls == 1 else 0,
            output="tests failed" if self.calls == 1 else "ok",
        )


class AlwaysFailVerifier:
    async def verify(self):
        return VerificationResult(
            command="fake",
            passed=False,
            exit_code=1,
            output="still broken",
        )


@pytest.mark.asyncio
async def test_agent_loop_retries_when_verification_fails():
    verifier = FailingThenPassingVerifier()
    loop = AgentLoop(
        tools=[],
        model_adapter=MockModelAdapter([
            [make_text("done"), make_stop()],
            [make_text("fixed"), make_stop()],
        ]),
        system_prompt="Test.",
        verifier=verifier,
        verify_retries=2,
    )

    events = [event async for event in loop.run("fix tests")]
    done = next(event for event in events if isinstance(event, DoneEvent))

    assert verifier.calls == 2
    assert done.state.termination_reason == TerminationReason.COMPLETED
    assert "External verification failed" in done.state.messages[-2].content


@pytest.mark.asyncio
async def test_agent_loop_exhausted_verification_is_not_success():
    loop = AgentLoop(
        tools=[],
        model_adapter=MockModelAdapter([
            [make_text("done"), make_stop()],
            [make_text("done again"), make_stop()],
        ]),
        system_prompt="Test.",
        verifier=AlwaysFailVerifier(),
        verify_retries=1,
    )

    events = [event async for event in loop.run("fix tests")]
    done = next(event for event in events if isinstance(event, DoneEvent))

    assert done.state.termination_reason == TerminationReason.VERIFY_FAILED


class BlockingTool(Tool):
    name = "BlockingAgent"
    input_schema = {}
    description = "returns a structured completion blocker"

    async def execute(self):
        return "verification failed"

    def completion_blocker(self):
        return "isolated subagent verification failed"


@pytest.mark.asyncio
async def test_tool_completion_blocker_prevents_parent_success():
    loop = AgentLoop(
        tools=[BlockingTool()],
        model_adapter=MockModelAdapter([
            [make_tool_use("BlockingAgent"), make_stop()],
            [make_text("done"), make_stop()],
            [make_text("still done"), make_stop()],
        ]),
        system_prompt="Test.",
        verify_retries=1,
    )

    events = [event async for event in loop.run("run isolated task")]
    done = next(event for event in events if isinstance(event, DoneEvent))

    assert done.state.termination_reason == TerminationReason.VERIFY_FAILED
    assert any(
        "isolated subagent verification failed" in message.content
        for message in done.state.messages
    )
