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

import asyncio
import atexit
import hashlib
import json
import math
import re
import sqlite3
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from capabilities.tokenize import keyword_tokens


def _normalize(text: str) -> str:
    """Lowercase; replace punctuation/underscores with spaces; collapse ws.

    Uses Unicode-aware \\W so non-ASCII text (e.g. Chinese) is preserved —
    only punctuation and underscores become separators.
    """
    return re.sub(r"[\W_]+", " ", text.lower()).strip()


def _tokenize(text: str) -> set[str]:
    """关键词集合 —— 走全项目共用的双语分词器。

    这里以前是 `set(_normalize(text).split())`，而 `_normalize` 用 `[\\W_]+`
    切分、Python 的 `\\w` **包含中文** → 中文整句不会被切开，程序性记忆的
    关键词交集恒为空（中文经验永远搜不到）。英文路径一直是对的，所以
    `test_procedural_search_strips_punctuation` 那种用例照样通过、掩盖了这条。
    现在与 skill 路由共用 `capabilities/tokenize.py`。
    """
    return keyword_tokens(text)


class NGramEmbeddingFunction:
    """Pure-Python character n-gram feature-hashing vectorizer.

    Maps normalized text to a fixed-dimension vector via hashed character
    n-grams (1/2/3-grams), L2-normalized. Signed hashing (the ±1 sign
    trick) reduces collision bias, and longer grams are weighted higher
    since they are more discriminative. Substring overlap becomes cosine
    similarity — a reasonable proxy for keyword-level memory retrieval.
    """

    def __init__(self, dim: int = 1024, use_unigrams: bool = True,
                 unigram_weight: float = 1.0):
        """use_unigrams=False 时只保留 2/3-gram。

        为什么留这个开关：1-gram 让任意两段英文短文本都有非零相似度
        （"把 README 的标题改成中文" 会和 "Task: 修 docx 依赖" 撞出 0.05），
        是检索噪声的主要来源。默认保持老行为，实验侧才关掉它做对照。
        """
        self.dim = dim
        self.use_unigrams = use_unigrams
        self.unigram_weight = unigram_weight

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        text = _normalize(text)
        if not text:
            return vec
        gram_spec = ((2, 2.0), (3, 3.0))
        if self.use_unigrams:
            gram_spec = ((1, self.unigram_weight),) + gram_spec
        for n, weight in gram_spec:
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
    # 同一条经验被重复写入的次数。写入侧按内容指纹去重：重复不是新增条目，
    # 而是把这一条的时间戳推新、计数 +1（"最近又用到"的经验不该被裁剪掉）。
    hit_count: int = 1

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
        "hit_count": entry.hit_count,
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
        hit_count=int(d.get("hit_count", 1) or 1),
    )


def _fingerprint(text: str) -> str:
    """内容指纹 —— 去重用的稳定键（规范化后取短哈希）。

    为什么不用原文比较：同一条经验两次写进来，空白/大小写的细微差别会让它看起来
    是两条，于是 200 条的容量被"同一个教训"占满。
    """
    normalized = _normalize(text or "")
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:16]


# ── Shared ranking helpers (used by all backends) ──

def _normalize_key(s: str) -> str:
    """Normalize a profile key/query: lowercase, _/- → space."""
    return _normalize(s)


def _bm25_scores(docs: list[list[str]], query: str,
                 k1: float = 1.2, b: float = 0.75) -> list[float]:
    """标准 Okapi BM25（词项用项目统一的双语分词结果）。"""
    lengths = [len(d) or 1 for d in docs]
    avgdl = (sum(lengths) / len(lengths)) if lengths else 1.0
    df: dict[str, int] = {}
    for d in docs:
        for term in set(d):
            df[term] = df.get(term, 0) + 1
    n = len(docs)
    query_terms = keyword_tokens(query)
    scores = []
    for tokens, dl in zip(docs, lengths):
        tf: dict[str, int] = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1
        score = 0.0
        for term in query_terms:
            f = tf.get(term, 0)
            if not f:
                continue
            idf = math.log(1 + (n - df.get(term, 0) + 0.5) / (df.get(term, 0) + 0.5))
            score += idf * (f * (k1 + 1)) / (f + k1 * (1 - b + b * dl / avgdl))
        scores.append(score)
    return scores


def _rank_procedural(entries: list[dict], query: str, top_k: int,
                     min_matches: int = 1) -> list[dict]:
    """程序性记忆排序 —— BM25（带 IDF）。

    原来是"关键词交集计数"，问题在**没有 IDF**：'报错/文件/测试' 这类高频词
    和 `docx` 这类关键标识符同权，而程序性记忆恰恰最靠标识符和报错串。

    选型来自 benchmarks/memory_scorer_eval.py 的对照实验（15 条正例 / 9 条反例）：
      BM25（IDF）        top1 0.67  top3 0.93  AUC 0.887   ← 选它
      2/3-gram 余弦      top1 0.73  top3 0.93  AUC 0.804
      1/2/3-gram 余弦    top1 0.67  top3 0.87  AUC 0.808
      BM25 + 余弦 RRF    top1 0.67  top3 0.87  AUC 0.646   ← 融合反而最差，不采纳
    融合稀释了单路强打分器 —— 与 RAG 项目里"RRF 提召回、降排序质量"同因。

    `min_matches`：严格模式（注入用）要求**至少命中 2 个词**。为什么不靠分数门槛：
    BM25 分数没有上界，"统计仓库有多少行代码" 只和某条记忆共享一个「代码」
    就能拿到正分。手动查询保持 min_matches=1（宁可多给）。
    """
    if not entries:
        return []
    docs = [keyword_tokens(e.get("content", "")) for e in entries]
    scores = _bm25_scores(docs, query)
    query_terms = keyword_tokens(query)
    scored = []
    for entry, tokens, score in zip(entries, docs, scores):
        if score <= 0:
            continue
        matched = len(set(tokens) & query_terms)
        if matched < min_matches:
            continue
        scored.append((entry, score * (1.0 if entry.get("success", True) else 0.5)))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [e for e, _ in scored[:top_k]]


def _rrf_fuse(ranked_lists: list[list], top_k: int, k: int = 60) -> list:
    """按名次融合多个来源的排名（不用原始分 —— 它们量纲不同）。

    为什么要融合而不是"按类型排优先级"：早先无类型搜索是
    `画像 + 情景 + 程序性` 直接拼接，等于**情景永远压过程序性**，
    即使真正相关的是一条程序性经验（实测 top1 只有 0.27）。
    各类型的分数不可比（BM25 无上界、余弦在 [-1,1]），所以只能比名次。
    注意这与"不做 RRF 融合"不矛盾：那条结论针对的是**同一个候选集合上的两个
    打分器**（融合会稀释强打分器）；这里是**互不相交的三类结果**，融合用来定
    跨类型顺序。
    """
    fused: dict[str, float] = {}
    by_id: dict[str, object] = {}
    for ranked in ranked_lists:
        for rank, item in enumerate(ranked):
            key = getattr(item, "id", None) or getattr(item, "content", str(rank))
            by_id.setdefault(key, item)
            fused[key] = fused.get(key, 0.0) + 1.0 / (k + rank + 1)
    order = sorted(fused.items(), key=lambda kv: -kv[1])
    return [by_id[key] for key, _ in order[:top_k]]


_MIN_SIMILARITY = 0.01  # substring-level overlap floor for episodic search

# 注入场景要"别吵"，手动查询要"别漏" —— 所以两个门槛分开。
# 标定数据（固化在 tests/test_memory.py，来自 benchmarks/memory_scorer_eval.py 的语料）：
#   真正需要情景记忆的正例，2/3-gram 余弦最低 0.111（docx 那对）
#   反例（语料里没有相关记忆）最高 0.095
# 取 0.10 落在两者之间。余量确实很窄 —— 这也是为什么自动注入还配了
# "程序性至少命中 2 个词"和"每轮最多 3 条"两道限制，并且必须有开关。
# 手动查询仍用 _MIN_SIMILARITY=0.01（宁可多给，模型自己筛）。
MEMORY_INJECT_MIN_SIMILARITY = 0.10

# 主提示词里 CLAUDE.md 的预算。参考 Claude Code 的 auto memory 加载策略
# （MEMORY.md 前 200 行或前 25KB，先到为准），这里压得更紧：
# 它每轮都进系统提示词，不设上限就是拿 token 换噪声。
MAX_CLAUDE_MD_LINES = 200
MAX_CLAUDE_MD_CHARS = 8000

# 记忆注入的预算：最多几条、总共多少字符。
MEMORY_HINT_TOP_K = 3
MEMORY_HINT_MAX_CHARS = 600

_TYPE_LABELS = {
    "procedural": "经验",
    "episodic": "事件",
    "user_profile": "偏好",
}


def _truncate_claude_md(text: str) -> str:
    """按行数/字符双上限截断 CLAUDE.md，并明确标注被截断过。"""
    lines = text.splitlines()
    truncated = False
    if len(lines) > MAX_CLAUDE_MD_LINES:
        lines = lines[:MAX_CLAUDE_MD_LINES]
        truncated = True
    out = "\n".join(lines)
    if len(out) > MAX_CLAUDE_MD_CHARS:
        out = out[:MAX_CLAUDE_MD_CHARS]
        truncated = True
    if truncated:
        out += "\n[... 已截断：CLAUDE.md 超出预算，完整内容见文件]"
    return out


def format_memory_hint(entries: list, top_k: int = MEMORY_HINT_TOP_K,
                       max_chars: int = MEMORY_HINT_MAX_CHARS) -> str:
    """把召回到的记忆拼成系统提示词里的一小段。

    两个约束都不是洁癖：① 这段**每轮都进上下文**，超预算就是拿 token 换噪声；
    ② 记忆可能过时或错配，所以措辞上写明"仅供参考，不符请忽略"，
    避免模型把旧经验当成事实照搬。
    """
    if not entries:
        return ""
    header = "## 可能相关的历史记忆（仅供参考，若与当前任务不符请忽略）"
    lines = [header]
    used = len(header)
    for entry in entries[:top_k]:
        content = (getattr(entry, "content", "") or "").strip().replace("\n", " ")[:200]
        if not content:
            continue
        label = _TYPE_LABELS.get(getattr(entry, "type", ""), getattr(entry, "type", ""))
        line = f"- [{label}] {content}"
        if used + len(line) > max_chars:
            break
        lines.append(line)
        used += len(line)
    return "\n".join(lines) if len(lines) > 1 else ""


def _rank_episodic(entries: list[dict], query: str, top_k: int,
                   embedder: "NGramEmbeddingFunction",
                   min_similarity: float = _MIN_SIMILARITY) -> list[dict]:
    """n-gram cosine ranking for episodic memory (shared by backends)."""
    scored = []
    for e in entries:
        sim = embedder.similarity(query, e.get("content", ""))
        if sim > min_similarity:
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
        # 情景记忆用 2/3-gram（关掉 1-gram）：1-gram 让任意两段英文短文本都有
        # 非零相似度，是检索噪声的主要来源。选型依据见 benchmarks/memory_scorer_eval.py
        # （去掉 1-gram 后 top1 0.67→0.73、top3 0.87→0.93）。
        self._embedder = NGramEmbeddingFunction(use_unigrams=False)
        self._max_entries = 200     # 与 SQLiteMemoryStore 的每类上限保持一致

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
        # 按内容指纹去重（与 SQLiteMemoryStore 行为一致）：重复出现的是"最近又用到"
        # 的那条经验，不是新经验 —— 否则 200 条容量会被同一个教训占满。
        fingerprint = _fingerprint(entry.content)
        entries = self._load_json_list(path)
        for e in entries:
            if _fingerprint(e.get("content", "")) == fingerprint:
                e["hit_count"] = int(e.get("hit_count", 1) or 1) + 1
                e["timestamp"] = entry.timestamp
                e["success"] = entry.success     # 同一条经验：最近一次的结果说了算
                self._write_json_list(path, entries)
                return
        self._store_json_list(path, _entry_to_dict(entry))

    def _write_json_list(self, path: Path, entries: list[dict]) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(entries[-self._max_entries:], ensure_ascii=False, indent=2),
                        encoding="utf-8")

    def search(self, memory_type: MemoryType | None, query: str,
               top_k: int, min_similarity: float | None = None) -> list[MemoryEntry]:
        if memory_type == MemoryType.USER_PROFILE:
            return self._search_profile(query)
        if memory_type == MemoryType.PROCEDURAL:
            return self._search_procedural(query, top_k)
        if memory_type == MemoryType.EPISODIC:
            return self._search_episodic(query, top_k, min_similarity)
        # 无类型过滤 → 三类各出一份排名再**按名次融合**（不比原始分：余弦有界、
        # BM25 无界）。以前是三类直接拼接，等于情景永远压过程序性。
        strict = min_similarity is not None
        return _rrf_fuse([
            self._search_profile(query),
            self._search_episodic(query, top_k, min_similarity),
            self._search_procedural(query, top_k, strict=strict),
        ], top_k)

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

    def _search_procedural(self, query: str, top_k: int,
                           strict: bool = False) -> list[MemoryEntry]:
        entries = self._load_json_list(self._procedural_path)
        return [_dict_to_entry(e)
                for e in _rank_procedural(entries, query, top_k,
                                          min_matches=2 if strict else 1)]

    def _search_episodic(self, query: str, top_k: int,
                         min_similarity: float | None = None) -> list[MemoryEntry]:
        entries = self._load_json_list(self._episodic_path)
        if not entries:
            return []
        return [_dict_to_entry(e)
                for e in _rank_episodic(
                    entries, query, top_k, self._embedder,
                    _MIN_SIMILARITY if min_similarity is None else min_similarity)]

    def _search_profile(self, query: str) -> list[MemoryEntry]:
        if not self._profile_path.exists():
            return []
        try:
            profile = json.loads(self._profile_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []

        q = _normalize_key(query)
        # 空查询不能命中全部画像：`q in key` 在 q 为空串时恒为真 —— 那会让
        # "任务为空时的提示词组装"把所有偏好一次性灌进上下文。
        if not q:
            return []
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
        # 同上：情景记忆固定用 2/3-gram（选型依据见 benchmarks/memory_scorer_eval.py）
        self._embedder = NGramEmbeddingFunction(use_unigrams=False)
        self._lock = threading.Lock()
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(
            str(self._db_path), check_same_thread=False
        )
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._init_schema()
        atexit.register(self.close)

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
                    success INTEGER DEFAULT 1,
                    fingerprint TEXT DEFAULT '',
                    hit_count INTEGER DEFAULT 1
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
            self._migrate()

    def _migrate(self) -> None:
        """给老库补上 fingerprint / hit_count 两列，并回填指纹。

        为什么必须显式处理：老库已经存在，`CREATE TABLE IF NOT EXISTS` 不会动它 ——
        少了这两列，去重和计数会**静默失效**（写进去的每一条都变成新条目）。
        """
        cols = {row[1] for row in self._conn.execute("PRAGMA table_info(memories)")}
        for name, ddl in (("fingerprint", "TEXT DEFAULT ''"),
                          ("hit_count", "INTEGER DEFAULT 1")):
            if name not in cols:
                self._conn.execute(f"ALTER TABLE memories ADD COLUMN {name} {ddl}")
        rows = self._conn.execute(
            "SELECT rowid, content FROM memories "
            "WHERE fingerprint IS NULL OR fingerprint = ''"
        ).fetchall()
        for rowid, content in rows:
            self._conn.execute(
                "UPDATE memories SET fingerprint = ?, "
                "hit_count = COALESCE(hit_count, 1) WHERE rowid = ?",
                (_fingerprint(content), rowid),
            )
        self._conn.commit()

    # ── MemoryStore interface ──

    def add(self, entry: MemoryEntry) -> None:
        if entry.memory_type == MemoryType.USER_PROFILE:
            self.set_profile(entry.context, entry.content)
            return
        fingerprint = _fingerprint(entry.content)
        with self._lock:
            existing = self._conn.execute(
                "SELECT id FROM memories WHERE memory_type = ? AND fingerprint = ? "
                "LIMIT 1", (entry.memory_type.value, fingerprint),
            ).fetchone()
            if existing is not None:
                # 同一条经验重复出现：不新增条目，只把时间戳推新、计数 +1
                # （"最近又用到"的经验不该被按时间裁剪掉）
                self._conn.execute(
                    "UPDATE memories SET hit_count = COALESCE(hit_count, 1) + 1, "
                    "timestamp = ?, success = ? WHERE id = ?",
                    (entry.timestamp, 1 if entry.success else 0, existing[0]),
                )
            else:
                self._conn.execute(
                    "INSERT INTO memories "
                    "(id, memory_type, content, context, file_paths, tags, "
                    "timestamp, success, fingerprint, hit_count) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (
                        entry.id,
                        entry.memory_type.value,
                        entry.content,
                        entry.context,
                        json.dumps(entry.file_paths),
                        json.dumps(entry.tags),
                        entry.timestamp,
                        1 if entry.success else 0,
                        fingerprint,
                        1,
                    ),
                )
            self._prune(entry.memory_type)
            self._conn.commit()

    def search(self, memory_type: MemoryType | None, query: str,
               top_k: int, min_similarity: float | None = None) -> list[MemoryEntry]:
        floor = _MIN_SIMILARITY if min_similarity is None else min_similarity
        if memory_type == MemoryType.USER_PROFILE:
            return self._search_profile(query)
        if memory_type == MemoryType.PROCEDURAL:
            entries = self._load_type(MemoryType.PROCEDURAL)
            return [_dict_to_entry(e)
                    for e in _rank_procedural(
                        entries, query, top_k,
                        min_matches=2 if min_similarity is not None else 1)]
        if memory_type == MemoryType.EPISODIC:
            entries = self._load_type(MemoryType.EPISODIC)
            return [_dict_to_entry(e)
                    for e in _rank_episodic(entries, query, top_k, self._embedder,
                                            floor)]
        # 无类型过滤 → 三类各出一份排名，再按**名次**融合。两个历史坑：
        #   ① 曾把用户画像排除在外（RecallMemory 因此永远拿不到用户偏好）；
        #   ② 曾是"三类直接拼接 + 每类各取 top_k"，既让情景永远压过程序性
        #      （实测 top1 只有 0.27），又让返回条数最多到 2~3 倍 top_k。
        # 分数不可比（余弦有界 / BM25 无界），所以融合只比名次。
        strict = min_similarity is not None
        return _rrf_fuse([
            self._search_profile(query),
            [_dict_to_entry(e)
             for e in _rank_episodic(self._load_type(MemoryType.EPISODIC),
                                     query, top_k, self._embedder, floor)],
            [_dict_to_entry(e)
             for e in _rank_procedural(self._load_type(MemoryType.PROCEDURAL),
                                       query, top_k,
                                       min_matches=2 if strict else 1)],
        ], top_k)

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
                "tags, timestamp, success, hit_count FROM memories "
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
                "hit_count": r[8] if len(r) > 8 else 1,
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
        if not q:      # 同上：空查询不该命中全部偏好
            return []
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
    configured store (default: SQLiteMemoryStore).
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
        """项目约定文本（进主提示词）。

        缺文件时给"去创建"的提示文案；存在时**按预算截断** —— 这份内容每轮都进
        系统提示词，越写越长就是在每轮烧 token。
        """
        if not self._claude_md_path.exists():
            return (
                "[No CLAUDE.md found. Create one at the project root to store "
                "coding conventions and project context; whatever it contains "
                "is injected into this prompt.]"
            )
        return _truncate_claude_md(self._claude_md_path.read_text(encoding="utf-8"))

    def project_rules(self) -> str:
        """CLAUDE.md 的真实内容；文件不存在时返回空串。

        与 load_claude_md 的区别：那个在主提示词里要给出"去创建 CLAUDE.md"的
        提示文案，所以缺文件时会返回一段占位说明。子 Agent 的提示词拼装需要的
        是**真实约定**，没有就该是空的，不该把占位说明当约定灌进去。
        """
        if not self._claude_md_path.exists():
            return ""
        return self._claude_md_path.read_text(encoding="utf-8")

    def load_session_summary(self) -> str:
        """上一次会话的规则式摘要（REPL 退出时写入），注入 System Prompt。"""
        if not self._session_summary_path.exists():
            return ""
        # 摘要要进每一轮的 System Prompt，做个上限避免旧摘要越滚越大
        return self._session_summary_path.read_text(
            encoding="utf-8"
        )[:2000]

    def record_session_summary(self, summary: str):
        """REPL 退出时调用，见 main.py。CLAUDE.md 保持人工维护，不自动改写。"""
        self._session_summary_path.parent.mkdir(parents=True, exist_ok=True)
        self._session_summary_path.write_text(summary, encoding="utf-8")

    # ── Entry-level memory (delegated to store) ──

    async def record_procedural(self, pattern: str, context: str = "") -> MemoryEntry:
        entry = MemoryEntry.make(
            MemoryType.PROCEDURAL, pattern,
            context=context, tags=["procedural", "pattern"],
        )
        await asyncio.to_thread(self._store.add, entry)
        return entry

    async def record_episodic(self, event: str, file_paths: list[str] | None = None,
                              success: bool = True,
                              tags: list[str] | None = None) -> MemoryEntry:
        entry = MemoryEntry.make(
            MemoryType.EPISODIC, event,
            file_paths=file_paths or [], tags=tags or ["episodic"],
            success=success,
        )
        await asyncio.to_thread(self._store.add, entry)
        return entry

    async def record_user_profile(self, key: str, value: str) -> MemoryEntry:
        entry = MemoryEntry.make(
            MemoryType.USER_PROFILE, value,
            context=key, tags=["user_profile"],
        )
        await asyncio.to_thread(self._store.add, entry)
        return entry

    # ── Backward-compatible convenience methods ──

    async def record_file_edit(self, file_path: str):
        """记一条"改过这个文件"的情景记忆。

        只记路径、**不记改动内容**：记忆是给后续检索用的，把代码正文写进去既费
        token 又容易过时，全文另有 ContextCompressor 的 record_edit 负责回灌。
        （原来签名里还挂着 old/new 两个参数，但从未被使用，调用方传的是空串 —— 已删除。）
        """
        await self.record_episodic(f"Edited {file_path}", file_paths=[file_path])

    # ── Search ──

    async def search(self, query: str, top_k: int = 5,
                     memory_type: MemoryType | None = None,
                     min_similarity: float | None = None) -> list[MemoryEntry]:
        return await asyncio.to_thread(
            self._store.search, memory_type, query, top_k, min_similarity
        )

    def search_sync(self, query: str, top_k: int = 5,
                    memory_type: MemoryType | None = None,
                    min_similarity: float | None = None) -> list[MemoryEntry]:
        """同步版检索 —— 给"同步的提示词组装"用。

        为什么要有它：`search()` 是 async（内部套了 asyncio.to_thread），而系统提示词的
        组装（main.py 的 system_prompt_for）是同步函数。底层 store.search 本来就是同步的，
        所以这里直接调它 —— 不必为了注入记忆把整条提示词工厂改成异步。
        """
        return self._store.search(memory_type, query, top_k, min_similarity)

    def close(self) -> None:
        """Release the underlying store's resources (e.g. SQLite connection)."""
        close = getattr(self._store, "close", None)
        if close is not None:
            close()
