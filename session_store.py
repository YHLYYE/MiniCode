"""Session persistence — save/load conversation state across process restarts

REPL mode already keeps context continuous within one process (via shared
LoopState). SessionStore extends this across process boundaries: each
conversation is serialized to JSON under .minicode/sessions/, and can be
resumed later with `python main.py --resume <id>`.
"""

import json
import uuid
from datetime import datetime
from pathlib import Path

from core.state import LoopState, Message, ContinueReason


class SessionStore:
    """Serialize/deserialize LoopState to JSON files."""

    def __init__(self, sessions_dir: Path | None = None):
        self._dir = sessions_dir or Path(".minicode/sessions")
        self._dir.mkdir(parents=True, exist_ok=True)

    def save(self, state: LoopState, session_id: str | None = None) -> str:
        """Save a LoopState. Returns the session id (auto-generated if None)."""
        if session_id is None:
            # uuid suffix guarantees uniqueness even for rapid successive saves
            # (time.time_ns() and datetime.now() both lack sufficient resolution
            # on Windows for back-to-back calls).
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            session_id = f"{ts}_{uuid.uuid4().hex[:8]}"

        data = {
            "session_id": session_id,
            "saved_at": datetime.now().isoformat(),
            "messages": [
                {
                    "role": m.role,
                    "content": m.content,
                    "tool_call_id": m.tool_call_id,
                    "tool_calls": m.tool_calls,
                }
                for m in state.messages
            ],
            "turn_count": state.turn_count,
            "total_tokens": state.total_tokens,
            "total_cost_usd": state.total_cost_usd,
            "active_skills": list(state.active_skills),
            # 存档时记下"最后一轮为什么结束/继续"。它**不进 LoopState**（见 load），
            # 只作为元信息，让 --resume 时能说一句"上次结束于：输出被截断"。
            "last_transition": state.transition.value if state.transition else None,
        }

        path = self._dir / f"{session_id}.json"
        path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return session_id

    def load(self, session_id: str) -> LoopState | None:
        """Load a LoopState by session id. Returns None if not found."""
        path = self._dir / f"{session_id}.json"
        if not path.exists():
            return None

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

        messages = tuple(
            Message(
                role=m["role"],
                content=m.get("content", ""),
                tool_call_id=m.get("tool_call_id"),
                tool_calls=m.get("tool_calls"),
            )
            for m in data.get("messages", [])
        )

        return LoopState(
            messages=messages,
            turn_count=data.get("turn_count", 0),
            total_tokens=data.get("total_tokens", 0),
            total_cost_usd=data.get("total_cost_usd", 0.0),
            max_output_tokens_recovery=0,
            auto_compact_attempts=0,
            # 刻意不还原存档里的 last_transition：恢复会话是**开始一个新任务**，
            # 这一轮的"为什么继续"还没发生。那份旧值只用于展示（见 last_transition）。
            transition=ContinueReason.NEXT_TURN,
            active_skills=tuple(data.get("active_skills", [])),
        )

    def last_transition(self, session_id: str) -> ContinueReason | None:
        """读存档里记的「上次结束/继续的原因」，仅用于展示。

        存在意义：恢复会话时能说一句"上次结束于：输出被截断（升级上限）"，
        而不是只给一个光秃秃的轮数。
        """
        path = self._dir / f"{session_id}.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        raw = data.get("last_transition")
        if not raw:
            return None
        try:
            return ContinueReason(raw)
        except ValueError:
            return None

    def list_sessions(self) -> list[str]:
        """List available session ids (most recent first)."""
        if not self._dir.exists():
            return []
        sessions = [p.stem for p in self._dir.glob("*.json")]
        sessions.sort(reverse=True)
        return sessions

    def delete(self, session_id: str) -> bool:
        """Delete a session file. Returns True if deleted."""
        path = self._dir / f"{session_id}.json"
        if path.exists():
            path.unlink()
            return True
        return False
