"""SessionStore 持久化测试 — 保存/加载/列表/删除"""
import pytest
from session_store import SessionStore
from core.state import LoopState, Message, ContinueReason


@pytest.fixture
def store(tmp_path):
    return SessionStore(sessions_dir=tmp_path)


def _make_state():
    return LoopState(
        messages=(
            Message(role="system", content="You are an AI."),
            Message(role="user", content="hello"),
            Message(role="assistant", content="hi", tool_calls=[
                {"id": "call_1", "type": "function",
                 "function": {"name": "read", "arguments": "{}"}}
            ]),
            Message(role="tool", content="file content", tool_call_id="call_1"),
        ),
        turn_count=2,
        total_tokens=150,
        total_cost_usd=0.001,
        active_skills=("code_review",),
    )


def test_save_and_load_roundtrip(store):
    state = _make_state()
    sid = store.save(state)

    loaded = store.load(sid)

    assert loaded is not None
    assert loaded.turn_count == 2
    assert loaded.total_tokens == 150
    assert loaded.total_cost_usd == 0.001
    assert loaded.active_skills == ("code_review",)
    assert len(loaded.messages) == 4
    # 消息内容保留
    assert loaded.messages[0].role == "system"
    assert loaded.messages[1].content == "hello"
    # tool_calls 和 tool_call_id 保留
    assert loaded.messages[2].tool_calls is not None
    assert loaded.messages[3].tool_call_id == "call_1"


def test_load_nonexistent(store):
    assert store.load("nonexistent") is None


def test_list_sessions(store):
    store.save(_make_state())
    store.save(_make_state())
    sessions = store.list_sessions()
    assert len(sessions) == 2


def test_delete(store):
    sid = store.save(_make_state())
    assert store.delete(sid) is True
    assert store.load(sid) is None
    assert store.delete(sid) is False  # already gone


def test_resume_state_has_next_turn_transition(store):
    """加载的会话 transition 应为 NEXT_TURN，可继续对话"""
    sid = store.save(_make_state())
    loaded = store.load(sid)
    assert loaded.transition == ContinueReason.NEXT_TURN
