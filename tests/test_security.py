"""安全修复回归测试 — 路径穿越 / 任意读 / rm 绕过 / 拒绝执行 / fail-closed"""
import pytest
from pathlib import Path

from capabilities.security import (
    RuleFilter, AIRiskClassifier, SecurityBlock, RiskLevel,
)
from core.tools.base import ToolCall


def _write(path) -> ToolCall:
    return ToolCall("Write", {"file_path": str(path)})


def _read(path) -> ToolCall:
    return ToolCall("Read", {"file_path": str(path)})


def _bash(cmd) -> ToolCall:
    return ToolCall("Bash", {"command": cmd})


# ── 路径穿越 / 任意读 ──

def test_write_path_traversal_blocked():
    """Write 用 .. 穿越出项目目录 → 拦截"""
    f = RuleFilter()
    outside = Path.cwd() / ".." / "secret.txt"
    with pytest.raises(SecurityBlock):
        f.check(_write(outside))


def test_write_sibling_prefix_blocked():
    """同名前缀目录（minicode-evil）不因 startswith 放行"""
    f = RuleFilter()
    sibling = Path(str(Path.cwd()) + "-evil") / "pwn.txt"
    with pytest.raises(SecurityBlock):
        f.check(_write(sibling))


def test_read_outside_project_blocked():
    """Read 任意路径（如 ~/.ssh/id_rsa）→ 拦截"""
    f = RuleFilter()
    outside = Path.cwd() / ".." / ".." / ".ssh" / "id_rsa"
    with pytest.raises(SecurityBlock):
        f.check(_read(outside))


def test_write_inside_project_allowed():
    """项目目录内的读写不被误伤"""
    f = RuleFilter()
    f.check(_write(Path.cwd() / "foo.txt"))  # 不抛异常
    f.check(_read(Path.cwd() / "src" / "main.py"))


# ── rm 命令绕过 ──

def test_rm_fr_blocked():
    """rm -fr（标志反转）不再绕过"""
    f = RuleFilter()
    with pytest.raises(SecurityBlock):
        f.check(_bash("rm -fr /"))
    with pytest.raises(SecurityBlock):
        f.check(_bash("rm -rf /"))
    with pytest.raises(SecurityBlock):
        f.check(_bash("rm --recursive /tmp"))


def test_proc_sys_blocked_at_l1():
    """/proc、/sys 现在也在 L1 层拦截（此前只在 BashTool 层）"""
    f = RuleFilter()
    with pytest.raises(SecurityBlock):
        f.check(_bash("cat /proc/self/environ"))


# ── AI 分类器 fail-closed ──

@pytest.mark.asyncio
async def test_classifier_fails_closed_on_no_model():
    """无模型时返回 HIGH（fail-closed），而非 MEDIUM 自动放行"""
    c = AIRiskClassifier(model=None)
    assert await c.classify(ToolCall("Bash", {"command": "x"})) == RiskLevel.HIGH


@pytest.mark.asyncio
async def test_classifier_fails_closed_on_exception():
    """分类器异常时返回 HIGH"""
    class BadModel:
        async def chat(self, **kwargs):
            raise RuntimeError("boom")

    c = AIRiskClassifier(model=BadModel())
    assert await c.classify(ToolCall("Bash", {"command": "x"})) == RiskLevel.HIGH


@pytest.mark.asyncio
async def test_classifier_invalid_risk_fails_closed():
    """非法风险值（如大写 HIGH）返回 HIGH，而非崩溃回落到 MEDIUM"""
    class UppercaseModel:
        async def chat(self, **kwargs):
            return '{"risk": "HIGH", "reason": "test"}'

    c = AIRiskClassifier(model=UppercaseModel())
    assert await c.classify(ToolCall("Bash", {"command": "x"})) == RiskLevel.HIGH
