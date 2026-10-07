"""三种记忆分类测试 — 程序性/情景/用户画像"""
import pytest
import asyncio
from capabilities.memory import (
    MAX_CLAUDE_MD_LINES, MemoryManager, MemoryType, MemoryEntry,
    JSONMemoryStore, SQLiteMemoryStore, format_memory_hint,
)


@pytest.fixture
def memory_manager(tmp_path):
    """Use tmp_path to isolate memory files between tests."""
    return MemoryManager(project_root=tmp_path)


@pytest.mark.asyncio
async def test_record_procedural(memory_manager):
    entry = await memory_manager.record_procedural(
        "NPE fix: check null first, then use Optional",
        context="java",
    )
    assert entry.memory_type == MemoryType.PROCEDURAL

    results = await memory_manager.search(
        "npe null", memory_type=MemoryType.PROCEDURAL
    )
    assert len(results) > 0
    assert "null" in results[0].content.lower()


@pytest.mark.asyncio
async def test_record_user_profile(memory_manager):
    await memory_manager.record_user_profile("java_version", "17")
    await memory_manager.record_user_profile("naming_style", "snake_case")

    # 下划线 vs 空格归一化匹配
    results = await memory_manager.search(
        "java version", memory_type=MemoryType.USER_PROFILE
    )
    assert len(results) == 1
    assert results[0].content == "17"


@pytest.mark.asyncio
async def test_user_profile_miss(memory_manager):
    """不存在的 key 返回空"""
    results = await memory_manager.search(
        "nonexistent_key", memory_type=MemoryType.USER_PROFILE
    )
    assert results == []


@pytest.mark.asyncio
async def test_search_without_type_filter(memory_manager):
    """无类型过滤时搜索所有类型（不崩溃即可）"""
    await memory_manager.record_procedural("test pattern for search")
    results = await memory_manager.search("test pattern", top_k=5)
    assert isinstance(results, list)


def test_memory_entry_make():
    """MemoryEntry.make 正确设置 memory_type 和 type"""
    entry = MemoryEntry.make(
        MemoryType.PROCEDURAL, "content", tags=["test"]
    )
    assert entry.memory_type == MemoryType.PROCEDURAL
    assert entry.type == "procedural"
    assert entry.id.startswith("procedural_")


def test_memory_type_enum_values():
    """三种记忆类型的枚举值正确"""
    assert MemoryType.PROCEDURAL.value == "procedural"
    assert MemoryType.EPISODIC.value == "episodic"
    assert MemoryType.USER_PROFILE.value == "user_profile"


def test_procedural_keyword_search(memory_manager):
    """程序性记忆用关键词检索（非向量）"""
    async def run():
        await memory_manager.record_procedural("how to fix NPE error")
        await memory_manager.record_procedural("how to write async code")
        results = await memory_manager.search(
            "NPE", memory_type=MemoryType.PROCEDURAL
        )
        return results

    results = asyncio.run(run())
    assert len(results) >= 1
    assert any("NPE" in r.content for r in results)


def test_sqlite_store_persistence(tmp_path):
    """SQLite 后端跨 MemoryManager 实例持久化"""
    async def run():
        mm1 = MemoryManager(project_root=tmp_path)
        await mm1.record_procedural("sqlite persistence check pattern")
        mm2 = MemoryManager(project_root=tmp_path)
        return await mm2.search("sqlite persistence",
                                memory_type=MemoryType.PROCEDURAL)

    results = asyncio.run(run())
    assert any("sqlite persistence" in e.content for e in results)


def test_json_store_backend_still_works(tmp_path):
    """JSON 后端仍可作为显式 store 使用"""
    async def run():
        mm = MemoryManager(project_root=tmp_path,
                           store=JSONMemoryStore(tmp_path / "memories"))
        await mm.record_procedural("json backend pattern")
        return await mm.search("json backend",
                               memory_type=MemoryType.PROCEDURAL)

    results = asyncio.run(run())
    assert any("json backend" in e.content for e in results)


def test_sqlite_store_prunes_old_entries(tmp_path):
    """SQLite 后端按类型裁剪超出 max_entries 的旧条目"""
    store = SQLiteMemoryStore(tmp_path / "mem.sqlite", max_entries=5)
    mm = MemoryManager(project_root=tmp_path, store=store)

    async def run():
        for i in range(10):
            await mm.record_procedural(f"pattern number {i}")
        return await mm.search("pattern number",
                               memory_type=MemoryType.PROCEDURAL, top_k=100)

    results = asyncio.run(run())
    assert len(results) == 5


def test_procedural_search_strips_punctuation(tmp_path):
    """检索能匹配带标点的记忆（此前 'NPE:' 分词后匹配不到 'NPE'）"""
    mm = MemoryManager(project_root=tmp_path)

    async def run():
        await mm.record_procedural("fix NPE: null-check before deref")
        return await mm.search("NPE", memory_type=MemoryType.PROCEDURAL)

    results = asyncio.run(run())
    assert any("NPE" in e.content for e in results)


def test_successful_memory_ranks_above_failed(tmp_path):
    """成功经验排在失败尝试之前。

    注意两条记忆必须是**不同内容**：写入侧现在按内容指纹去重，同内容的
    "失败版 + 成功版" 会被合并成一条（最近一次的结果说了算），那是刻意的。
    """
    mm = MemoryManager(project_root=tmp_path)

    async def run():
        await mm.record_episodic("deploy strategy X failed with timeout",
                                 success=False)
        await mm.record_episodic("deploy strategy X succeeded after retry",
                                 success=True)
        return await mm.search("deploy strategy", memory_type=MemoryType.EPISODIC)

    results = asyncio.run(run())
    assert len(results) >= 2
    assert results[0].success is True


def test_chinese_content_is_searchable(tmp_path):
    """中文内容在归一化后仍可检索（Unicode 感知分词）"""
    mm = MemoryManager(project_root=tmp_path)

    async def run():
        await mm.record_episodic("修复了空指针异常的 bug")
        return await mm.search("空指针", memory_type=MemoryType.EPISODIC)

    results = asyncio.run(run())
    assert any("空指针" in e.content for e in results)


def test_chinese_procedural_memory_is_searchable(tmp_path):
    """程序性记忆 + 中文 —— 这条以前是**恒空**的。

    上面那条测的是情景记忆（n-gram 余弦，中文一直没问题）。程序性记忆走
    关键词交集，而旧分词器对中文不切词（Python 的 \\w 含中文）→
    `{'蜂巢取快递验证码摁错怎么办'}` 与语料交集为空 → 中文经验永远召回不到。
    中文用户是主要用户，所以这是实打实的功能缺失，不是洁癖。
    """
    mm = MemoryManager(project_root=tmp_path)

    async def run():
        await mm.record_procedural("蜂巢取快递验证码摁错时，重新输入一次就好")
        await mm.record_procedural("unrelated python tip")
        return await mm.search("取快递验证码摁错", memory_type=MemoryType.PROCEDURAL)

    results = asyncio.run(run())
    assert results, "中文程序性记忆搜不到"
    assert "蜂巢取快递" in results[0].content


def test_keyword_tokens_are_bilingual():
    """"共用分词器"的行为契约：英文按词、中文按字符 bigram。"""
    from capabilities.tokenize import keyword_tokens

    tokens = keyword_tokens("用 bge-m3 做 dense 检索")
    assert {"bge", "m3", "dense"} <= tokens
    assert "检索" in tokens          # 中文 bigram
    # 标点与下划线都不是 token
    assert not any(t in tokens for t in ("，", "_", "。"))


def test_project_rules_is_empty_when_claude_md_is_missing(tmp_path):
    """缺失时返回空串 —— 子 Agent 提示词拼装需要「没有就是没有」。

    而 load_claude_md 缺失时返回的是一段「去创建 CLAUDE.md」的提示文案（那是给
    主提示词用的）。两者混用，就会把占位说明当成项目约定灌给每一个子 Agent。
    """
    mm = MemoryManager(project_root=tmp_path)
    assert mm.project_rules() == ""
    assert "No CLAUDE.md" in mm.load_claude_md()      # 另一个接口的行为保持不变


def test_project_rules_reads_the_file(tmp_path):
    (tmp_path / "CLAUDE.md").write_text("禁止用 Tab 缩进", encoding="utf-8")
    mm = MemoryManager(project_root=tmp_path)
    assert mm.project_rules() == "禁止用 Tab 缩进"


# ── 不带类型过滤的搜索：三类都要搜，且条数受 top_k 约束 ──
# 回归背景：两个后端原来都是"episodic + procedural"，把用户画像排除在外，
# 而 RecallMemoryTool 的描述里承诺了 "project conventions"；而且每类各取
# top_k 再相加，传 top_k=3 会拿到最多 6 条，top_k 契约失效。

@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["sqlite", "json"])
async def test_untyped_search_covers_all_three_types_and_respects_top_k(tmp_path, backend):
    store = (None if backend == "sqlite"
             else JSONMemoryStore(tmp_path / "json_memories"))
    mm = MemoryManager(project_root=tmp_path, store=store)
    for i in range(3):
        await mm.record_procedural(f"重建索引再跑测试 pattern{i}")
        await mm.record_episodic(f"Task: 重建索引 pattern{i}")
    await mm.record_user_profile("index_habit", "改完索引先重建再跑测试")

    hits = await mm.search("重建索引", top_k=3)
    assert len(hits) <= 3, f"top_k=3 却返回了 {len(hits)} 条"

    # 画像必须能通过"不带类型"的搜索拿到（key 命中）
    profile_hits = await mm.search("index_habit", top_k=5)
    assert any(h.type == "user_profile" for h in profile_hits), (
        "用户画像在无类型搜索里查不到 —— RecallMemory 就永远拿不到用户偏好"
    )
    mm.close()


# ── 会话摘要（此前是死代码：定义了但没人调用） ──

def test_session_summary_round_trip(tmp_path):
    mm = MemoryManager(project_root=tmp_path)
    assert mm.load_session_summary() == ""

    mm.record_session_summary("Files modified (2 total):\n  - a.py\n  - b.py")
    loaded = mm.load_session_summary()
    assert "a.py" in loaded and "b.py" in loaded


def test_session_summary_is_capped(tmp_path):
    mm = MemoryManager(project_root=tmp_path)
    mm.record_session_summary("x" * 5000)
    assert len(mm.load_session_summary()) == 2000


def test_claude_md_fallback_promises_nothing_it_cannot_do(tmp_path):
    """提示词里不能再写「agent 会自动更新 CLAUDE.md」——那是死代码。"""
    mm = MemoryManager(project_root=tmp_path)
    text = mm.load_claude_md()
    assert "will update it" not in text
    assert "No CLAUDE.md found" in text


def test_update_claude_md_is_gone():
    """未被任何人调用的写入口已删除，避免文档宣称一个不存在的能力。"""
    assert not hasattr(MemoryManager, "update_claude_md")


# ── CLAUDE.md 的加载预算（参考 Claude Code：MEMORY.md 前 200 行 / 25KB）──

def test_claude_md_is_truncated_to_budget(tmp_path):
    """主提示词里的 CLAUDE.md 必须有上限 —— 它每轮都进系统提示词。"""
    big = "\n".join(f"约定第 {i} 条" for i in range(MAX_CLAUDE_MD_LINES + 50))
    (tmp_path / "CLAUDE.md").write_text(big, encoding="utf-8")
    mm = MemoryManager(project_root=tmp_path)

    got = mm.load_claude_md()
    assert len(got.splitlines()) <= MAX_CLAUDE_MD_LINES + 1   # 多出来那行是截断标注
    assert "已截断" in got
    assert "约定第 0 条" in got
    assert f"约定第 {MAX_CLAUDE_MD_LINES + 49} 条" not in got
    mm.close()


def test_claude_md_under_budget_is_untouched(tmp_path):
    text = "禁止用 Tab 缩进\n测试必须能离线跑"
    (tmp_path / "CLAUDE.md").write_text(text, encoding="utf-8")
    mm = MemoryManager(project_root=tmp_path)
    assert mm.load_claude_md() == text        # 没超预算不加任何标注
    assert mm.project_rules() == text         # 子 Agent 那条路不受影响
    mm.close()


# ── 记忆注入的格式、预算与同步检索 ──

def _entry(content: str, kind: str = "procedural") -> MemoryEntry:
    return MemoryEntry.make(MemoryType(kind), content)


def test_memory_hint_is_empty_without_entries():
    assert format_memory_hint([]) == ""


def test_memory_hint_labels_type_and_warns_it_is_advisory():
    hint = format_memory_hint([
        _entry("报错 No module named docx 时先装依赖", "procedural"),
        _entry("Task: 重构 auth | 7 turns", "episodic"),
    ])
    assert "仅供参考" in hint                     # 记忆可能过时/错配，不能当事实
    assert "[经验]" in hint and "[事件]" in hint
    assert "No module named docx" in hint


def test_memory_hint_respects_top_k_and_char_budget():
    entries = [_entry("x" * 300) for _ in range(10)]
    hint = format_memory_hint(entries, top_k=3, max_chars=400)
    bullets = [ln for ln in hint.splitlines() if ln.startswith("- ")]
    assert 1 <= len(bullets) <= 3
    assert len(hint) <= 400 + 120             # 标题不计入，条目必须守着预算


@pytest.mark.asyncio
async def test_search_sync_matches_search(tmp_path):
    """同步版是给提示词组装用的，结果必须和异步版一致。"""
    mm = MemoryManager(project_root=tmp_path)
    await mm.record_procedural("重建索引再跑测试")
    mm.close()

    mm2 = MemoryManager(project_root=tmp_path)
    async_hits = await mm2.search("重建索引", top_k=3)
    sync_hits = mm2.search_sync("重建索引", top_k=3)
    assert [e.content for e in async_hits] == [e.content for e in sync_hits]
    mm2.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["sqlite", "json"])
async def test_empty_query_does_not_return_every_profile(tmp_path, backend):
    """空查询不能命中全部偏好 —— 任务为空时的提示词组装会把它整段灌进上下文。"""
    store = (None if backend == "sqlite"
             else JSONMemoryStore(tmp_path / "json_memories"))
    mm = MemoryManager(project_root=tmp_path, store=store)
    await mm.record_user_profile("naming_style", "snake_case")
    await mm.record_user_profile("prefer", "先跑测试再提交")

    assert await mm.search("", top_k=5) == []
    assert mm.search_sync("   ") == []
    mm.close()


# ── 三个计数器：写入 / 被检索命中 / 被采纳 ──
# 回归背景：原来只有一个 hit_count，记的是"被写进来几次"（去重用）——
# 回答不了"这条经验有没有被用上"。注入时记 retrieval，任务结束时按结局记 adoption。

@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["sqlite", "json"])
async def test_retrieval_and_adoption_counters(tmp_path, backend):
    store = (None if backend == "sqlite"
             else JSONMemoryStore(tmp_path / "json_memories"))
    mm = MemoryManager(project_root=tmp_path, store=store)
    await mm.record_procedural("重建索引再跑测试")

    hits = mm.search_sync("重建索引", top_k=3)
    assert hits, "检索不到就没法测计数"
    assert mm.record_injection(hits) == len(hits)      # retrieval_count +1

    # 任务成功结束 → 注入过的记忆记一次"有用"
    assert mm.record_task_outcome(success=True) == len(hits)
    entry = mm.search_sync("重建索引", top_k=3)[0]
    assert entry.retrieval_count == 1
    assert entry.adoption_count == 1

    # 同一个任务重复结算不许重复记账（批次已清空）
    assert mm.record_task_outcome(success=True) == 0

    # 出错的任务：adoption 记 −1（"注入了但还是出错"是可观测的负面信号）
    mm.record_injection(mm.search_sync("重建索引", top_k=3))
    mm.record_task_outcome(success=False)
    entry2 = mm.search_sync("重建索引", top_k=3)[0]
    assert entry2.retrieval_count == 2
    assert entry2.adoption_count == 0
    mm.close()


def test_store_bump_rejects_unknown_counter(tmp_path):
    store = JSONMemoryStore(tmp_path / "json_memories")
    with pytest.raises(ValueError):
        store.bump(["x"], "not_a_counter", 1)


# ── 领域词典（查询侧别名扩展）──
# 动机：用户用中文提问、记忆里是英文标识符时，纯字面匹配是 0 命中
# （实测固定的 3 个 MISS 里 2 个属于这一类）。

def test_alias_expansion_is_bidirectional():
    from capabilities.memory import expand_aliases

    # 中文 → 补英文标识符
    assert "file" in expand_aliases("上次改了哪个文件")
    assert "install" in expand_aliases("上次那个装不上的依赖")
    # 英文 → 补中文（反向）
    assert "文件" in expand_aliases("which file did I edit")


def test_alias_expansion_leaves_unknown_text_alone():
    from capabilities.memory import expand_aliases

    plain = "把 README 的标题改成中文"
    assert expand_aliases(plain) == plain, "没有别名命中时不该动原文"
    assert expand_aliases("") == ""


# ── 排序加成（精排）：不再只靠 BM25 原始分 ──

def test_ranking_bonus_rewards_coverage_and_usage():
    """命中覆盖率高、被用过且被证明有用的条目，排序加成更高。

    注意：这一项**不影响谁能进候选**（那是 `_match_is_evidence` 的职责），
    只影响先后顺序。评测语料里的计数器都是 0，所以它不扰动已有基线数字 ——
    这条测试就是它的行为契约。
    """
    from capabilities.memory import _ranking_bonus

    base = {"success": True}
    one_match = _ranking_bonus(base, {"docx"})
    three_match = _ranking_bonus(base, {"docx", "install", "依赖"})
    assert three_match > one_match, "命中更多的应该排在前面"

    used = {"success": True, "retrieval_count": 4, "adoption_count": 3}
    assert _ranking_bonus(used, {"docx"}) > _ranking_bonus(base, {"docx"}), \
        "被检索过、被采纳过的经验应该加权"

    failed = {"success": False}
    assert _ranking_bonus(failed, {"docx"}) < _ranking_bonus(base, {"docx"}), \
        "失败过的尝试仍要降权"


# ── 注入场景的相关性下限（标定数据固化在这里）──

def test_remember_tool_documents_project_examples():
    """Remember 的三类选择必须有**本项目自己的例子**，而不是抽象英文定义。

    回归背景：早先描述只有一句 Which memory category to write to，模型只能靠名词猜，
    很容易把"可照做的规则"写成"发生过的日志" —— 而类型会连带决定检索算法
    （程序性走 BM25、情景走余弦），选错就捞不回来。
    """
    from core.tools.base import RememberTool

    desc = RememberTool.input_schema["memory_type"]["description"]
    assert "procedural" in desc and "episodic" in desc and "user_profile" in desc
    # 项目自己的例子（而不是 how to do X 这种抽象说法）
    assert "MINICODE_OFFLINE" in desc or "重建" in desc, desc
    assert "boundary" in desc.lower() or "边界" in desc, desc
    # user_profile 必须带 key 这条约束也要写在描述里
    assert "key" in desc
