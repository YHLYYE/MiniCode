"""Memory system — three-type memory with pluggable storage backends

Reference: Cognitive science memory classification (how-claude-code-works ch08)
+ DeepAgents' "pluggable state and store backends".

Three memory types:
1. Procedural (程序性记忆) — "how to do things" — long-term
2. Episodic (情景记忆) — "what happened" — mid-term
3. User Profile (用户画像) — "user preferences" — long-term

Storage is pluggable via MemoryStore:
- SQLiteMemoryStore (default) — stdlib sqlite3, transactional, concurrent-safe
- JSONMemoryStore — zero-dependency JSON + n-gram vector search
- Custom backends (ChromaDB / Redis) implement the same interface

Design note: Model parameters are frozen → true "self-evolution" is impossible.
We implement pragmatic persistence + retrieval, not weight updates.

Engineering decision: Originally planned ChromaDB for episodic semantic search,
but ChromaDB's default ONNX embedding crashes on Windows (onnxruntime access
violation). Replaced with a pure-Python character n-gram hashing vector +
cosine similarity — no external model, zero native dependencies.
"""

import hashlib
import json
import re
import sqlite3
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path


def _normalize(text: str) -> str:
    """Lowercase; replace punctuation/underscores with spaces; collapse ws.

    Uses Unicode-aware \\W so non-ASCII text (e.g. Chinese) is preserved —
    only punctuation and underscores become separators.
    """
    return re.sub(r"[\W_]+", " ", text.lower()).strip()


def _tokenize(text: str) -> set[str]:
    """Lowercase alphanumeric token set (punctuation stripped)."""
    return set(_normalize(text).split())


class NGramEmbeddingFunction:
    """Pure-Python character n-gram feature-hashing vectorizer.

    Maps normalized text to a fixed-dimension vector via hashed character
    n-grams (1/2/3-grams), L2-normalized. Signed hashing (the ±1 sign
    trick) reduces collision bias, and longer grams are weighted higher
    since they are more discriminative. Substring overlap becomes cosine
    similarity — a reasonable proxy for keyword-level memory retrieval.
    """

    def __init__(self, dim: int = 1024):
        self.dim = dim

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        text = _normalize(text)
        if not text:
            return vec
        for n, weight in ((1, 1.0), (2, 2.0), (3, 3.0)):
            for i in range(len(text) - n + 1):
                gram = text[i:i + n]
                h = int(hashlib.md5(gram.encode("utf-8")).hexdigest(), 16)
                bucket = h % self.dim
                sign = 1.0 if (h >> 64) & 1 else -1.0
                vec[bucket] += weight * sign
        norm = sum(v * v for v in vec) ** 0.5
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec

    def similarity(self, text_a: str, text_b: str) -> float:
        """Cosine similarity between two texts."""
        va = self.embed(text_a)
        vb = self.embed(text_b)
        return sum(a * b for a, b in zip(va, vb))


class MemoryType(Enum):
    PROCEDURAL = "procedural"
    EPISODIC = "episodic"
    USER_PROFILE = "user_profile"


@dataclass
class MemoryEntry:
    id: str
    memory_type: MemoryType
    content: str
    context: str = ""
    file_paths: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    timestamp: float = 0.0
    success: bool = True
    type: str = ""  # backward-compat alias for memory_type.value

    @classmethod
    def make(cls, memory_type: MemoryType, content: str, **kwargs) -> "MemoryEntry":
        kwargs.setdefault("id", f"{memory_type.value}_{int(time.time() * 1000)}")
        kwargs.setdefault("timestamp", time.time())
        return cls(
            memory_type=memory_type,
            content=content,
            type=memory_type.value,
            **kwargs,
        )


# ── Serialization helpers (module-level, shared by backends) ──

def _entry_to_dict(entry: MemoryEntry) -> dict:
    return {
        "id": entry.id, "memory_type": entry.memory_type.value,
        "content": entry.content, "context": entry.context,
        "file_paths": entry.file_paths, "tags": entry.tags,
        "timestamp": entry.timestamp, "success": entry.success,
    }


def _dict_to_entry(d: dict) -> MemoryEntry:
    mt = d.get("memory_type", d.get("type", "episodic"))
    try:
        memory_type = MemoryType(mt)
    except ValueError:
        memory_type = MemoryType.EPISODIC
    return MemoryEntry(
        id=d.get("id", ""), memory_type=memory_type,
        content=d.get("content", ""), context=d.get("context", ""),
        file_paths=d.get("file_paths", []), tags=d.get("tags", []),
        timestamp=d.get("timestamp", 0.0), success=d.get("success", True),
        type=memory_type.value,
    )


# ── Shared ranking helpers (used by all backends) ──

def _normalize_key(s: str) -> str:
    """Normalize a profile key/query: lowercase, _/- → space."""
    return _normalize(s)


def _rank_procedural(entries: list[dict], query: str, top_k: int) -> list[dict]:
    """Keyword-overlap ranking for procedural memory (shared by backends).

    Tokens are punctuation-stripped; successful entries rank above failed
    ones (a failed attempt stays recallable, just demoted).
    """
    query_tokens = _tokenize(query)
    scored = []
    for e in entries:
        text = e.get("content", "") + " " + " ".join(e.get("tags", []))
        overlap = len(query_tokens & _tokenize(text))
        if overlap > 0:
            score = overlap * (1.0 if e.get("success", True) else 0.5)
            scored.append((e, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [e for e, _ in scored[:top_k]]


_MIN_SIMILARITY = 0.01  # substring-level overlap floor for episodic search


def _rank_episodic(entries: list[dict], query: str, top_k: int,
                   embedder: "NGramEmbeddingFunction") -> list[dict]:
    """n-gram cosine ranking for episodic memory (shared by backends)."""
    scored = []
    for e in entries:
        sim = embedder.similarity(query, e.get("content", ""))
        if sim > _MIN_SIMILARITY:
            score = sim * (1.0 if e.get("success", True) else 0.5)
            scored.append((e, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [e for e, _ in scored[:top_k]]


# ── Pluggable memory store backends ──

class MemoryStore(ABC):
    """Pluggable memory storage backend (borrowed from DeepAgents).

    Implement this interface to swap storage media (SQLite, ChromaDB,
    Redis, ...) without changing MemoryManager's public API.
    """

    @abstractmethod
    def add(self, entry: MemoryEntry) -> None:
        """Persist a memory entry."""

    @abstractmethod
    def search(self, memory_type: MemoryType | None, query: str,
               top_k: int) -> list[MemoryEntry]:
        """Search memories, optionally filtered by type."""

    @abstractmethod
    def get_profile(self, key: str) -> str | None:
        """Get a user profile value by key."""

    @abstractmethod
    def set_profile(self, key: str, value: str) -> None:
        """Set a user profile key-value pair."""


class JSONMemoryStore(MemoryStore):
    """Default backend — zero-dependency JSON files + n-gram vector search.

    Storage layout:
    - procedural.json  (list of entries, keyword search)
    - episodic.json    (list of entries, n-gram cosine similarity)
    - profile.json     (key-value dict)
    """

    def __init__(self, memories_dir: Path):
        self._dir = memories_dir
        self._procedural_path = self._dir / "procedural.json"
        self._episodic_path = self._dir / "episodic.json"
        self._profile_path = self._dir / "profile.json"
        self._embedder = NGramEmbeddingFunction()

    # ── MemoryStore interface ──

    def add(self, entry: MemoryEntry) -> None:
        if entry.memory_type == MemoryType.USER_PROFILE:
            self.set_profile(entry.context, entry.content)
            return
        path = (
            self._procedural_path
            if entry.memory_type == MemoryType.PROCEDURAL
            else self._episodic_path
        )
        self._store_json_list(path, _entry_to_dict(entry))

    def search(self, memory_type: MemoryType | None, query: str,
               top_k: int) -> list[MemoryEntry]:
        if memory_type == MemoryType.USER_PROFILE:
            return self._search_profile(query)
        if memory_type == MemoryType.PROCEDURAL:
            return self._search_procedural(query, top_k)
        if memory_type == MemoryType.EPISODIC:
            return self._search_episodic(query, top_k)
        # No type filter → episodic + procedural
        return self._search_episodic(query, top_k) + self._search_procedural(query, top_k)

    def get_profile(self, key: str) -> str | None:
        if not self._profile_path.exists():
            return None
        try:
            profile = json.loads(self._profile_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        return profile.get(key)

    def set_profile(self, key: str, value: str) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        profile = {}
        if self._profile_path.exists():
            try:
                profile = json.loads(self._profile_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                profile = {}
        profile[key] = value
        self._profile_path.write_text(
            json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")

    # ── Internal storage helpers ──

    def _store_json_list(self, path: Path, entry_dict: dict, max_entries: int = 200):
        self._dir.mkdir(parents=True, exist_ok=True)
        entries = []
        if path.exists():
            try:
                entries = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                entries = []
        entries.append(entry_dict)
        entries = entries[-max_entries:]
        path.write_text(
            json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")

    def _load_json_list(self, path: Path) -> list[dict]:
        if not path.exists():
            return []
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []

    # ── Search backends ──

    def _search_procedural(self, query: str, top_k: int) -> list[MemoryEntry]:
        entries = self._load_json_list(self._procedural_path)
        return [_dict_to_entry(e)
                for e in _rank_procedural(entries, query, top_k)]

    def _search_episodic(self, query: str, top_k: int) -> list[MemoryEntry]:
        entries = self._load_json_list(self._episodic_path)
        if not entries:
            return []
        return [_dict_to_entry(e)
                for e in _rank_episodic(entries, query, top_k, self._embedder)]

    def _search_profile(self, query: str) -> list[MemoryEntry]:
        if not self._profile_path.exists():
            return []
        try:
            profile = json.loads(self._profile_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []

        q = _normalize_key(query)
        results = []
        for key, value in profile.items():
            if _normalize_key(key) in q or q in _normalize_key(key):
                results.append(MemoryEntry(
                    id=f"profile_{key}", memory_type=MemoryType.USER_PROFILE,
                    content=value, context=key, tags=["user_profile"],
                    timestamp=0.0, success=True, type="user_profile",
                ))
        return results


class SQLiteMemoryStore(MemoryStore):
    """SQLite backend — single-file, transactional, concurrent-safe.

    Replaces the JSON backend's read-modify-write (O(n) per add) with
    incremental INSERTs and indexed SELECTs. WAL mode + a lock make it
    safe under concurrent sub-agent writes.

    Storage layout:
    - memories table: procedural/episodic entries (id, memory_type, ...)
    - profile table:  key-value user preferences
    """

    def __init__(self, db_path: Path, max_entries: int = 200):
        self._db_path = db_path
        self._max_entries = max_entries
        self._embedder = NGramEmbeddingFunction()
        self._lock = threading.Lock()
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(
            str(self._db_path), check_same_thread=False
        )
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT,
                    memory_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    context TEXT DEFAULT '',
                    file_paths TEXT DEFAULT '[]',
                    tags TEXT DEFAULT '[]',
                    timestamp REAL DEFAULT 0,
                    success INTEGER DEFAULT 1
                );
                CREATE INDEX IF NOT EXISTS idx_memories_type
                    ON memories(memory_type);
                CREATE INDEX IF NOT EXISTS idx_memories_ts
                    ON memories(timestamp);
                CREATE TABLE IF NOT EXISTS profile (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                """
            )
            self._conn.commit()

    # ── MemoryStore interface ──

    def add(self, entry: MemoryEntry) -> None:
        if entry.memory_type == MemoryType.USER_PROFILE:
            self.set_profile(entry.context, entry.content)
            return
        with self._lock:
            self._conn.execute(
                "INSERT INTO memories "
                "(id, memory_type, content, context, file_paths, tags, "
                "timestamp, success) VALUES (?,?,?,?,?,?,?,?)",
                (
                    entry.id,
                    entry.memory_type.value,
                    entry.content,
                    entry.context,
                    json.dumps(entry.file_paths),
                    json.dumps(entry.tags),
                    entry.timestamp,
                    1 if entry.success else 0,
                ),
            )
            self._prune(entry.memory_type)
            self._conn.commit()

    def search(self, memory_type: MemoryType | None, query: str,
               top_k: int) -> list[MemoryEntry]:
        if memory_type == MemoryType.USER_PROFILE:
            return self._search_profile(query)
        if memory_type == MemoryType.PROCEDURAL:
            entries = self._load_type(MemoryType.PROCEDURAL)
            return [_dict_to_entry(e)
                    for e in _rank_procedural(entries, query, top_k)]
        if memory_type == MemoryType.EPISODIC:
            entries = self._load_type(MemoryType.EPISODIC)
            return [_dict_to_entry(e)
                    for e in _rank_episodic(entries, query, top_k, self._embedder)]
        # No type filter → episodic (cosine) + procedural (keyword)
        episodic = [_dict_to_entry(e)
                    for e in _rank_episodic(self._load_type(MemoryType.EPISODIC),
                                            query, top_k, self._embedder)]
        procedural = [_dict_to_entry(e)
                      for e in _rank_procedural(self._load_type(MemoryType.PROCEDURAL),
                                                query, top_k)]
        return episodic + procedural

    def get_profile(self, key: str) -> str | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT value FROM profile WHERE key = ?", (key,)
            ).fetchone()
        return row[0] if row else None

    def set_profile(self, key: str, value: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO profile (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value),
            )
            self._conn.commit()

    # ── Internal helpers ──

    def _load_type(self, memory_type: MemoryType) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, memory_type, content, context, file_paths, "
                "tags, timestamp, success FROM memories "
                "WHERE memory_type = ? ORDER BY timestamp DESC",
                (memory_type.value,),
            ).fetchall()
        return [
            {
                "id": r[0],
                "memory_type": r[1],
                "content": r[2],
                "context": r[3],
                "file_paths": json.loads(r[4] or "[]"),
                "tags": json.loads(r[5] or "[]"),
                "timestamp": r[6],
                "success": bool(r[7]),
            }
            for r in rows
        ]

    def _load_profile(self) -> dict:
        with self._lock:
            rows = self._conn.execute(
                "SELECT key, value FROM profile"
            ).fetchall()
        return dict(rows)

    def _search_profile(self, query: str) -> list[MemoryEntry]:
        profile = self._load_profile()
        q = _normalize_key(query)
        results = []
        for key, value in profile.items():
            if _normalize_key(key) in q or q in _normalize_key(key):
                results.append(MemoryEntry(
                    id=f"profile_{key}", memory_type=MemoryType.USER_PROFILE,
                    content=value, context=key, tags=["user_profile"],
                    timestamp=0.0, success=True, type="user_profile",
                ))
        return results

    def _prune(self, memory_type: MemoryType) -> None:
        """Delete oldest entries beyond max_entries for a given type."""
        self._conn.execute(
            "DELETE FROM memories WHERE memory_type = ? AND rowid NOT IN ("
            "  SELECT rowid FROM memories WHERE memory_type = ? "
            "  ORDER BY timestamp DESC LIMIT ?)",
            (memory_type.value, memory_type.value, self._max_entries),
        )

    def close(self) -> None:
        """Close the connection (checkpoints WAL, releases the file handle)."""
        with self._lock:
            self._conn.close()


class MemoryManager:
    """High-level memory API — delegates storage to a pluggable MemoryStore.

    Manages session-level context (CLAUDE.md, session summary) directly,
    and delegates entry-level memory (procedural/episodic/profile) to the
    configured store (default: JSONMemoryStore).
    """

    def __init__(self, project_root: Path | None = None,
                 store: MemoryStore | None = None):
        self._root = project_root or Path.cwd()
        self._claude_md_path = self._root / "CLAUDE.md"
        self._session_summary_path = self._root / ".minicode" / "session_summary.md"
        self._store = store or SQLiteMemoryStore(
            self._root / ".minicode" / "memories.sqlite"
        )

    # ── Session-level context ──

    def load_claude_md(self) -> str:
        if not self._claude_md_path.exists():
            return (
                "[No CLAUDE.md found. Create one at the project root "
                "to store coding conventions, preferences, and project "
                "context. The agent will update it as it learns about "
                "your project.]"
            )
        return self._claude_md_path.read_text(encoding="utf-8")

    def load_session_summary(self) -> str:
        if not self._session_summary_path.exists():
            return ""
        return self._session_summary_path.read_text(encoding="utf-8")

    def update_claude_md(self, convention: str):
        existing = ""
        if self._claude_md_path.exists():
            existing = self._claude_md_path.read_text(encoding="utf-8")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        new_entry = f"\n## Convention (added {timestamp})\n{convention}\n"
        self._claude_md_path.write_text(existing + new_entry, encoding="utf-8")

    def record_session_summary(self, summary: str):
        self._session_summary_path.parent.mkdir(parents=True, exist_ok=True)
        self._session_summary_path.write_text(summary, encoding="utf-8")

    # ── Entry-level memory (delegated to store) ──

    async def record_procedural(self, pattern: str, context: str = "") -> MemoryEntry:
        entry = MemoryEntry.make(
            MemoryType.PROCEDURAL, pattern,
            context=context, tags=["procedural", "pattern"],
        )
        self._store.add(entry)
        return entry

    async def record_episodic(self, event: str, file_paths: list[str] | None = None,
                              success: bool = True,
                              tags: list[str] | None = None) -> MemoryEntry:
        entry = MemoryEntry.make(
            MemoryType.EPISODIC, event,
            file_paths=file_paths or [], tags=tags or ["episodic"],
            success=success,
        )
        self._store.add(entry)
        return entry

    async def record_user_profile(self, key: str, value: str) -> MemoryEntry:
        entry = MemoryEntry.make(
            MemoryType.USER_PROFILE, value,
            context=key, tags=["user_profile"],
        )
        self._store.add(entry)
        return entry

    # ── Backward-compatible convenience methods ──

    async def record_file_edit(self, file_path: str, old: str, new: str,
                               context: str = ""):
        await self.record_episodic(f"Edited {file_path}", file_paths=[file_path])

    async def record_error_fix(self, error: str, fix: str,
                               file_paths: list[str] | None = None):
        await self.record_episodic(
            f"Error: {error[:300]}\nFix: {fix[:300]}",
            file_paths=file_paths or [],
        )

    # ── Search ──

    async def search(self, query: str, top_k: int = 5,
                     memory_type: MemoryType | None = None) -> list[MemoryEntry]:
        return self._store.search(memory_type, query, top_k)

    def close(self) -> None:
        """Release the underlying store's resources (e.g. SQLite connection)."""
        close = getattr(self._store, "close", None)
        if close is not None:
            close()
