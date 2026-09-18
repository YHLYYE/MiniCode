"""三种记忆分类测试 — 程序性/情景/用户画像"""
import pytest
import asyncio
from capabilities.memory import (
    MemoryManager, MemoryType, MemoryEntry, JSONMemoryStore, SQLiteMemoryStore,
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
    """成功经验排在失败尝试之前"""
    mm = MemoryManager(project_root=tmp_path)

    async def run():
        await mm.record_episodic("deploy strategy X", success=False)
        await mm.record_episodic("deploy strategy X", success=True)
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
