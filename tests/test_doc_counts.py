"""文档里写的测试条数，必须和实际收集到的条数一致。

为什么值得单开一条测试：这个数字在仓库里漂过三次 —— README 停在 151、
DESIGN 停在 203、ROADMAP 和 CONTRIBUTING 停在 204，而实际早已不是。
它不影响运行，但面试官随手翻 README 看到「151 unit tests」而项目里有 211 条，
那就是「文档落后于代码」的现成证据。

做法：拿 pytest 自己的 `--collect-only` 结果当准数，再扫文档里的声明逐个对齐。
`--collect-only` 不执行测试体，所以不会递归调回本文件。
"""
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ["README.md", "DESIGN.md", "ROADMAP.md", "CONTRIBUTING.md"]

# 只认「条数」这一种说法：`204 tests` / `151 unit tests` / `203 个测试`。
# 历史叙述里出现的「当时 159 条测试全绿」故意不匹配 —— 那是当时的真实条数。
PATTERNS = [
    re.compile(r"(\d+)\s*(?:unit\s+)?tests?\b"),
    re.compile(r"(\d+)\s*个测试"),
]


def _collected_count() -> int:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "--collect-only", "-q"],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
    )
    match = re.search(r"(\d+)\s+tests? collected", proc.stdout or "")
    assert match, f"拿不到收集条数：\nstdout={proc.stdout}\nstderr={proc.stderr}"
    return int(match.group(1))


def test_docs_quote_the_real_test_count():
    actual = _collected_count()
    mismatches = []
    for name in DOCS:
        text = (ROOT / name).read_text(encoding="utf-8")
        for pattern in PATTERNS:
            for match in pattern.finditer(text):
                if int(match.group(1)) == actual:
                    continue
                line_no = text[:match.start()].count("\n") + 1
                mismatches.append(
                    f"{name}:{line_no} 写「{match.group(0).strip()}」，实际 {actual}"
                )
    assert not mismatches, (
        "文档里的测试条数已过期（改完记得四个文件一起改）：\n  "
        + "\n  ".join(mismatches)
    )
