"""静态检查闸门 —— 防止「没人跑 linter」这件事再发生一次。

存在理由：`core/tools/shell.py` 曾经用了 `re.search` 但没 `import re`，
而当时 159 个测试全绿（没有任何测试真正执行过 Bash 工具本体）。
未定义名字 / 未使用导入 / 重复定义这三类问题，pyflakes 一秒就能发现。

pyflakes 不在环境里时跳过，不让纯运行环境因为缺一个开发依赖而红。
"""
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("pyflakes")

ROOT = Path(__file__).resolve().parent.parent


def test_static_analysis_is_clean():
    targets = [
        str(p) for p in ROOT.rglob("*.py") if "__pycache__" not in p.parts
    ]
    proc = subprocess.run(
        [sys.executable, "-m", "pyflakes", *targets],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, (
        "pyflakes 发现问题（未定义名字 / 未使用导入 / 重复定义）：\n"
        + proc.stdout + proc.stderr
    )
