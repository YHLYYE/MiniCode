"""三种记忆分类测试 — 程序性/情景/用户画像"""
import pytest
import asyncio
from capabilities.memory import MemoryManager, MemoryType, MemoryEntry


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
