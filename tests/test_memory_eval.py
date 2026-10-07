"""召回评测守门 —— 用**真实记忆管理器**跑一遍，把质量钉住。

和 benchmarks/memory_scorer_eval.py 的分工：那边是"选打分器"的对照实验
（拿几个打分器互相比），这边是"防劣化"的守门测试（生产路径必须达标）。
任何一次改动让召回质量掉下去，这里就会红。
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.memory_scorer_eval import CASES, MEMORIES   # noqa: E402
from capabilities.memory import (MEMORY_INJECT_MIN_SIMILARITY,  # noqa: E402
                                 MemoryManager)

_CONTENT_BY_ID = {m["id"]: m["content"] for m in MEMORIES}
# 及格线取实测值往下留一点余量，掉下来就该有人看。两组数字对应两种消费方：
#   自动注入（严格）：top1 0.48 / top3 0.95，**反例零漏**（宁缺毋滥）
#   手动检索（宽松）：top1 0.10 / top3 0.95，反例 8/8 会漏（宁可多给，模型自己筛）
# 注入侧一路从 0.40/0.67 提到 0.48/0.95，靠四件事：融合从按名次 RRF 改成按名次交错、
# "≥2 词命中"改成"≥2 词或命中 ASCII 标识符"、评测语料 12→24 条（IDF 才有区分度）、
# 查询侧领域词典（中文问句 ↔ 英文标识符双向扩展）。
# 注意 top1 偏低是刻意的取舍：查询扩展让更多条目命中（top3 涨），代价是第一位更挤。
MIN_TOP3_STRICT, MIN_TOP1_STRICT = 0.90, 0.45
MIN_TOP3_LOOSE = 0.90


@pytest.mark.asyncio
async def test_production_recall_meets_the_floor(tmp_path):
    """生产配置（BM25 程序性 + 2/3-gram 情景 + 注入门槛）的真实召回率。"""
    mm = MemoryManager(project_root=tmp_path)
    for m in MEMORIES:
        if m["type"] == "procedural":
            await mm.record_procedural(m["content"])
        else:
            await mm.record_episodic(m["content"])

    top1 = top3 = 0
    for query, expected in CASES:
        hits = mm.search_sync(query, top_k=3,
                              min_similarity=MEMORY_INJECT_MIN_SIMILARITY)
        got = [h.content for h in hits]
        want = [_CONTENT_BY_ID[i] for i in expected]
        top1 += bool(got[:1] and got[0] in want)
        top3 += bool(set(got) & set(want))
    mm.close()

    n = len(CASES)
    assert top3 / n >= MIN_TOP3_STRICT, f"注入模式 top-3 命中率掉到 {top3 / n:.2f}"
    assert top1 / n >= MIN_TOP1_STRICT, f"注入模式 top-1 命中率掉到 {top1 / n:.2f}"


@pytest.mark.asyncio
async def test_manual_recall_trades_precision_for_recall(tmp_path):
    """手动检索（RecallMemory 工具）走宽松门槛：宁可多给，模型自己筛。"""
    mm = MemoryManager(project_root=tmp_path)
    for m in MEMORIES:
        if m["type"] == "procedural":
            await mm.record_procedural(m["content"])
        else:
            await mm.record_episodic(m["content"])

    top3 = 0
    for query, expected in CASES:
        hits = mm.search_sync(query, top_k=3)      # 不传 min_similarity → 宽松
        got = [h.content for h in hits]
        top3 += bool(set(got) & {_CONTENT_BY_ID[i] for i in expected})
    mm.close()
    n = len(CASES)
    assert top3 / n >= MIN_TOP3_LOOSE, f"宽松模式 top-3 掉到 {top3 / n:.2f}"


@pytest.mark.asyncio
async def test_irrelevant_tasks_get_no_memory_hits(tmp_path):
    """反例：语料里没有相关记忆时，注入门槛必须把它们全部挡在门外。"""
    from benchmarks.memory_scorer_eval import NEGATIVES

    mm = MemoryManager(project_root=tmp_path)
    for m in MEMORIES:
        if m["type"] == "procedural":
            await mm.record_procedural(m["content"])
        else:
            await mm.record_episodic(m["content"])

    leaked = []
    for query in NEGATIVES:
        hits = mm.search_sync(query, top_k=3,
                              min_similarity=MEMORY_INJECT_MIN_SIMILARITY)
        if hits:
            leaked.append((query, [h.content[:40] for h in hits]))
    mm.close()
    assert not leaked, f"无关任务漏进了记忆：{leaked}"


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["sqlite", "json"])
async def test_user_corrections_are_recallable_with_natural_queries(tmp_path, backend):
    """用户纠正走更低门槛 —— 否则"记了纠正却召不回"等于没记。

    实测发现：4 条纠正用自然问法去查，有 3 条因为只命中 1 个中文 bigram 被严格规则
    筛掉（门槛当初是为"长任务描述"调的，对短句纠正不适用）。纠正本质上是"以后都要
    照做"的规则，更接近 CLAUDE.md 的常驻语义，所以单独开一条通道。
    """
    from capabilities.memory import (MEMORY_INJECT_MIN_SIMILARITY,
                                     JSONMemoryStore, MemoryManager)

    store = (None if backend == "sqlite"
             else JSONMemoryStore(tmp_path / "json_memories"))
    mm = MemoryManager(project_root=tmp_path, store=store)
    await mm.record_procedural("用户纠正：以后不要用 Tab 缩进，改用 4 个空格",
                               context="user_correction")

    for query in ("按项目约定，缩进用什么？", "这个仓库的缩进规范是什么"):
        hits = mm.search_sync(query, top_k=3,
                              min_similarity=MEMORY_INJECT_MIN_SIMILARITY)
        assert hits and "Tab" in hits[0].content, (query, [h.content for h in hits])

    # 无关任务仍然不许命中（降低门槛不能变成开闸放水）
    assert mm.search_sync("把 README 的标题改成中文", top_k=3,
                          min_similarity=MEMORY_INJECT_MIN_SIMILARITY) == []
    mm.close()
