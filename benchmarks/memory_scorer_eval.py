"""记忆检索打分的对照实验 —— 用数据选打分器，而不是凭感觉。

三个候选（都在这份脚本里可跑，互相对照）：
  A  1/2/3-gram 余弦（现在的生产实现，char n-gram 哈希 + 余弦）
  B  2/3-gram 余弦（关掉 1-gram：1-gram 是"任意英文都非零相似"的噪声来源）
  C  BM25 词项打分（带 IDF，用项目统一的双语分词器）
  C+ C 与 A 的 RRF 融合（对应 RAG 项目里"混合召回"的同一套思路）

指标按"注入场景真正需要的东西"设计：
  top1 / top3   正例（确实有相关记忆）的命中率 —— 检索可不可用
  AUC           成对排序准确率：相关记忆的得分 > 无关记忆的得分 的比例。
                **与分数尺度无关**，所以能公平比较"余弦"和"BM25"这两种量纲
                完全不同的打分器；它直接回答"能不能把有用和无水分开"。
  pos_min / neg_max  诊断值，只用来定位失败样本，不参与挑打分器。

实验设计上踩过两次坑，写在这里备忘：
  ① 最初把"无关最高分"算成整个矩阵的最大值（含别的查询的无关行）—— 被跨行噪声主导；
  ② 再改成"正例最低分 − 反例最高分"—— BM25 分数无上界，余弦在 [-1,1]，
     直接比原始分是量纲错误。所以最终用 AUC。

用法：
    python benchmarks/memory_scorer_eval.py            # 打印对照表
    python benchmarks/memory_scorer_eval.py --json xx  # 同时写存档
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from capabilities.memory import NGramEmbeddingFunction      # noqa: E402
from capabilities.tokenize import keyword_tokens            # noqa: E402

# ── 语料：用真实形态的记忆内容（报错/路径/任务/依赖，中英混排）──
MEMORIES: list[dict] = [
    {"id": "p1", "type": "procedural",
     "content": "No module named docx → 先 pip install python-docx 再重跑"},
    {"id": "p2", "type": "procedural",
     "content": "改了索引要重建再跑测试，否则用的是旧向量"},
    {"id": "p3", "type": "procedural",
     "content": "NameError: name 're' is not defined → 检查模块顶部有没有 import re"},
    {"id": "p4", "type": "procedural",
     "content": "pytest 卡在联网下载模型 → 设环境变量让测试离线跑"},
    {"id": "p5", "type": "procedural",
     "content": "写文件之后要 git diff 确认，别把临时调试代码提交上去"},
    {"id": "p6", "type": "procedural",
     "content": "Windows 上删临时目录前先关掉 sqlite 连接，否则文件被占用"},
    {"id": "e1", "type": "episodic", "content": "Task: 修 docx 依赖 | 5 turns"},
    {"id": "e2", "type": "episodic", "content": "Tool errors: - Bash: NameError re | 3 turns"},
    {"id": "e3", "type": "episodic", "content": "Task: 给界面加语料切换 | 12 turns | 88k tokens"},
    {"id": "e4", "type": "episodic", "content": "Edited core/agent_loop.py"},
    {"id": "e5", "type": "episodic", "content": "Task: 补记忆去重与提炼 | 9 turns"},
    {"id": "e6", "type": "episodic", "content": "Tool errors: - Grep: path outside project | 2 turns"},
]

# 每条查询给出"应该命中"的记忆 id（可多条）
CASES: list[tuple[str, set[str]]] = [
    ("又报 No module named docx 了", {"p1", "e1"}),
    ("docx 依赖装不上怎么办", {"p1"}),
    ("重建索引", {"p2"}),
    ("改完索引要不要重跑测试", {"p2"}),
    ("NameError: name 're' is not defined", {"p3", "e2"}),
    ("补一下 import re", {"p3"}),
    ("pytest 一直卡在下载", {"p4"}),
    ("怎么让测试离线跑", {"p4"}),
    ("提交之前要检查什么", {"p5"}),
    ("sqlite 文件被占用删不掉", {"p6"}),
    ("上次改了哪个文件", {"e4"}),
    ("记忆去重那个任务跑了多久", {"e5"}),
    # 转述型（和记忆没有字面重叠）—— 用来区分"字面匹配"和更宽的匹配
    ("上次那个装不上的依赖", {"p1"}),
    ("怎么让跑测试的时候别联网", {"p4"}),
    ("提交前怎么避免把调试代码带上去", {"p5"}),
]

# 反例：这些任务在本语料里**没有**相关记忆，打分器不该给出高分。
# 它们大多和某条记忆词面相近（"测试""文件""索引"），是最能暴露噪声的地方。
NEGATIVES: list[str] = [
    "把 README 的标题改成中文",
    "帮这个函数加类型注解",
    "统计仓库一共有多少行代码",
    "解释一下这段话在说什么",
    "今天天气怎么样",
    "给项目加一个 CI 配置文件",
    "帮我看看这个正则为什么匹配不到",
    "把这个模块重命名一下",
]


# ── 打分器 ──

def score_ngram(query: str, memories: list[dict], embedder: NGramEmbeddingFunction) -> list[float]:
    return [embedder.similarity(query, m["content"]) for m in memories]


def score_bm25(query: str, memories: list[dict], k1: float = 1.2, b: float = 0.75) -> list[float]:
    """标准 Okapi BM25（用项目统一的双语分词器）。"""
    docs = [keyword_tokens(m["content"]) for m in memories]
    lengths = [len(d) or 1 for d in docs]
    avgdl = sum(lengths) / len(lengths)
    df: Counter = Counter()
    for d in docs:
        df.update(set(d))
    n = len(docs)
    q_terms = keyword_tokens(query)
    scores = []
    for tokens, dl in zip(docs, lengths):
        tf = Counter(tokens)
        s = 0.0
        for term in q_terms:
            f = tf.get(term, 0)
            if not f:
                continue
            idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
            s += idf * (f * (k1 + 1)) / (f + k1 * (1 - b + b * dl / avgdl))
        scores.append(s)
    return scores


def rrf(*rank_lists: list[float], k: int = 60) -> list[float]:
    """按名次融合（与 RAG 项目里的 RRF 同一套做法）。"""
    fused = [0.0] * len(rank_lists[0])
    for scores in rank_lists:
        order = sorted(range(len(scores)), key=lambda i: -scores[i])
        for rank, idx in enumerate(order):
            fused[idx] += 1.0 / (k + rank + 1)
    return fused


# ── 评估 ──

def evaluate(scorer, memories=MEMORIES, cases=CASES,
             negatives=NEGATIVES) -> dict:
    """按"正例命中 + 反例噪声"两段来评估。

    早先那版把"无关最高分"算成整个矩阵的最大值（含别的查询的无关行），
    结果被跨行噪声主导，四个打分器还给出完全一样的命中率 —— 指标不成立，
    结论也就不能用。现在正例、反例分开算。
    """
    top1, top3 = 0, 0
    pos_strength: list[float] = []
    rows = []
    for query, expected in cases:
        scores = scorer(query, memories)
        order = sorted(range(len(memories)), key=lambda i: -scores[i])
        got_ids = [memories[i]["id"] for i in order[:3]]
        hit1 = bool(got_ids[:1] and got_ids[0] in expected)
        hit3 = bool(set(got_ids) & expected)
        top1 += hit1
        top3 += hit3
        pos_strength.append(max(scores[i] for i, m in enumerate(memories)
                                if m["id"] in expected))
        rows.append({"query": query, "expected": sorted(expected),
                     "top3": got_ids, "hit1": hit1, "hit3": hit3})

    neg_rows = []
    neg_strength: list[float] = []
    for query in negatives:
        scores = scorer(query, memories)
        best = max(scores)
        neg_strength.append(best)
        neg_rows.append({"query": query, "max_score": best})

    # AUC：所有 (正例, 反例) 组合里，正例得分更高的比例（并列算 0.5）
    wins = sum(1.0 if p > n else 0.5 if p == n else 0.0
               for p in pos_strength for n in neg_strength)
    auc = wins / (len(pos_strength) * len(neg_strength))

    n = len(cases)
    return {
        "n": n,
        "auc": auc,
        "relevant_min": min(pos_strength),
        "negative_max": max(neg_strength),
        "top1": top1 / n,
        "top3": top3 / n,
        "rows": rows,
        "negative_rows": neg_rows,
    }


def run_eval() -> dict:
    emb_123 = NGramEmbeddingFunction()
    emb_23 = NGramEmbeddingFunction(use_unigrams=False)

    def scorer_c(query, memories):
        return score_bm25(query, memories)

    def scorer_hybrid(query, memories):
        return rrf(score_bm25(query, memories),
                   score_ngram(query, memories, emb_123))

    scorers = {
        "A 1/2/3-gram 余弦（现状）": lambda q, ms: score_ngram(q, ms, emb_123),
        "B 2/3-gram 余弦（去掉 1-gram）": lambda q, ms: score_ngram(q, ms, emb_23),
        "C BM25（带 IDF）": scorer_c,
        "C+ BM25 与 A 的 RRF 融合": scorer_hybrid,
    }
    results = {name: evaluate(fn) for name, fn in scorers.items()}
    return {"generated_at": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
            "n_memories": len(MEMORIES), "results": results}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="记忆检索打分器对照实验")
    ap.add_argument("--json", default=None, help="把结果写到指定路径")
    args = ap.parse_args(argv)

    out = run_eval()
    print(f"语料 {out['n_memories']} 条记忆 / {next(iter(out['results'].values()))['n']} 条查询\n")
    print(f"{'打分器':<30}{'top1':>7}{'top3':>7}{'AUC':>7}{'正例最低':>10}{'反例最高':>10}")
    for name, r in out["results"].items():
        print(f"{name:<30}{r['top1']:>7.2f}{r['top3']:>7.2f}{r['auc']:>7.3f}"
              f"{r['relevant_min']:>10.3f}{r['negative_max']:>10.3f}")

    best = max(out["results"].items(), key=lambda kv: (kv[1]["auc"], kv[1]["top3"]))
    print(f"\n按 AUC 选：{best[0]}")
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                   encoding="utf-8")
        print(f"已写入存档：{args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
