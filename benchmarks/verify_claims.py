"""基准守护（MiniCode）—— 重跑基准，和文档里引用的数字对账。

和 AgenticRAG-QA 那半边的区别：这里的基准是**确定性的纯本地计算**
（不调 API、不加载模型），所以可以真的重跑再比对，而不是只验证出处。

三道检查：
  ① 重跑    —— 现跑一遍基准，拿到当前代码的真实值
  ② 取值    —— 文档里写的数字，和现跑出来的值一致吗？
  ③ 有出处  —— 这个字符串确实出现在那份文档里吗？

退出码非 0 表示有声明过期。改完被基准覆盖的代码，跑一遍这个再提交。

用法：
    python benchmarks/verify_claims.py
    python benchmarks/verify_claims.py --resume "C:/path/to/简历.md"
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import io
import sys
from dataclasses import dataclass, field
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
ROOT = BENCH_DIR.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(BENCH_DIR))

DEFAULT_RESUME = "../简历-2026秋招.md"


@dataclass
class Claim:
    label: str
    source: str                       # 计算出来的值里的键
    quoted: dict[str, str] = field(default_factory=dict)
    # 已经被取代的旧值：文档里只要还能搜到就算失败。
    # 这正是我们踩过的坑——代码改了，文档里旧数字还留着。
    stale: list[str] = field(default_factory=list)


CLAIMS: list[Claim] = [
    # ── 压缩比基准（record.py，模拟 200 文件 / 58K token 长会话）──
    Claim("原始（无压缩）token", "before", {"README.md": "58,030", "DESIGN.md": "58,030"}),
    Claim("Snip 后 token", "snip", {"README.md": "8,110", "DESIGN.md": "8,110"}),
    Claim("Snip 压缩比", "snip_ratio", {"README.md": "86%", "DESIGN.md": "86%"}),
    Claim("Collapse 后 token", "collapse", {"README.md": "2,959", "DESIGN.md": "2,959"}),
    Claim("Collapse 压缩比", "collapse_ratio", {"README.md": "95%", "DESIGN.md": "95%"}),
    Claim("Autocompact 后 token", "autocompact", {"README.md": "28", "DESIGN.md": "28"}),
    Claim("Autocompact 压缩比", "autocompact_ratio", {"README.md": "99.95%", "DESIGN.md": "99.95%"}),

    # ── 端到端成本（e2e.py，模拟读 30 文件）──
    Claim("关压缩：累计 input token", "e2e_off_tokens",
          {"README.md": "1,173,075", "DESIGN.md": "1,173,075"}),
    Claim("开压缩：累计 input token", "e2e_on_tokens",
          {"README.md": "425,520", "DESIGN.md": "425,520"}),
    Claim("关压缩：最终上下文 token", "e2e_off_final", {"DESIGN.md": "75,657"}),
    Claim("开压缩：最终上下文 token", "e2e_on_final", {"DESIGN.md": "13,354"}),
    Claim("Token 成本降低", "e2e_saved_pct",
          {"README.md": "63.7%", "DESIGN.md": "63.7%", DEFAULT_RESUME: "63.7%"},
          stale=["成本降低：74.8%", "降低 74.8%"]),
    Claim("开压缩：累计 input token（旧值不得残留）", "e2e_on_tokens",
          {"README.md": "425,520", "DESIGN.md": "425,520"},
          stale=["296,058"]),
    Claim("Token 成本降低（绝对量）", "e2e_saved_k", {"DESIGN.md": "748K"},
          stale=["747K"]),
]


def _compute() -> dict[str, float]:
    import e2e
    import record

    values: dict[str, float] = {}

    # 压缩比：把基准脚本的打印吞掉，只取它返回的值
    with contextlib.redirect_stdout(io.StringIO()):
        r = asyncio.run(record.benchmark_compression())
    before = r["before"]
    for key in ("snip", "collapse", "autocompact"):
        values[key] = r[key]
        values[f"{key}_ratio"] = (before - r[key]) / before * 100
    values["before"] = before

    # 端到端：同一份模拟，分别关压缩 / 开压缩
    with contextlib.redirect_stdout(io.StringIO()):
        off = asyncio.run(e2e.simulate_session(30, 1_000_000))
        on = asyncio.run(e2e.simulate_session(30, 20_000))
    values["e2e_off_tokens"] = off["total_input_tokens"]
    values["e2e_on_tokens"] = on["total_input_tokens"]
    values["e2e_off_final"] = off["final_context_tokens"]
    values["e2e_on_final"] = on["final_context_tokens"]
    saved = off["total_input_tokens"] - on["total_input_tokens"]
    values["e2e_saved_pct"] = saved / off["total_input_tokens"] * 100
    values["e2e_saved_k"] = saved / 1000
    return values


def _fmt(value: float, like: str) -> str:
    """按文档里那个字符串的格式（千分位 / 小数位 / 百分号）格式化真实值。"""
    suffix = ""
    body = like
    for s in ("%", "K", "k"):
        if body.endswith(s):
            suffix, body = s, body[:-1]
            break
    decimals = len(body.split(".")[1]) if "." in body else 0
    out = f"{value:,.{decimals}f}" if "," in body else f"{value:.{decimals}f}"
    return out + suffix


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="MiniCode 基准守护")
    ap.add_argument("--resume", default=DEFAULT_RESUME,
                    help=f"简历路径（默认 {DEFAULT_RESUME}；不存在则跳过相关声明）")
    args = ap.parse_args(argv)

    resume_path = Path(args.resume) if Path(args.resume).is_absolute() else (ROOT / args.resume)

    print("=" * 74)
    print("MiniCode 基准守护 — 重跑基准并对账文档引用")
    print("=" * 74)
    print("跑基准中（record.py + e2e.py，纯本地确定性计算）...")
    values = _compute()
    print("完成。\n")

    failures: list[str] = []
    checked = 0

    for c in CLAIMS:
        actual = values[c.source]
        ok = True
        notes: list[str] = []

        # ② 取值 + ③ 有出处
        for doc, shown in c.quoted.items():
            path = resume_path if doc == DEFAULT_RESUME else (ROOT / doc)
            expected = _fmt(actual, shown)
            if expected != shown:
                ok = False
                notes.append(f"取值不符：{doc} 写 {shown}，现跑应为 {expected}")
            if not path.exists():
                notes.append(f"文档不存在，跳过引用检查：{doc}")
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            if shown not in text:
                ok = False
                notes.append(f"{doc} 里找不到字符串 {shown}")
            for s in c.stale:
                if s in text:
                    ok = False
                    notes.append(f"{doc} 里残留旧值 {s}")

        checked += 1
        if not ok:
            failures.append(c.label)
        print(f"{'✅' if ok else '❌'} {c.label}")
        print(f"     现跑真实值 {actual:,.4f}")
        for doc, shown in c.quoted.items():
            print(f"     引用 {doc} -> {shown}")
        if c.stale:
            print(f"     不得残留 {'、'.join(c.stale)}")
        for n in notes:
            print(f"     · {n}")

    print("\n" + "=" * 74)
    if failures:
        print(f"结果：❌ {len(failures)}/{checked} 条与当前代码不符")
        for f in failures:
            print(f"   · {f}")
        print("\n要么改代码，要么改文档 —— 别让文档里的数字停在旧版本上。")
        return 1
    print(f"结果：✅ 全部 {checked} 条通过（重跑值 == 文档值）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
