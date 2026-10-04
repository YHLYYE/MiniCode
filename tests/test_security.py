"""安全修复回归测试 — 路径穿越 / 任意读 / rm 绕过 / 拒绝执行 / fail-closed"""
import pytest
from pathlib import Path

from capabilities.security import (
    RuleFilter, AIRiskClassifier, SecurityBlock, RiskLevel,
)
from capabilities.security import PermissionManager, ToolSelfCheck
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
# 注：`rm -rf / rm -fr / rm --recursive` 的参数化用例在 test_hardening.py 的
# test_unix_rm_still_blocked 里，那里覆盖更全（含 --force）+ 还带着平台对照，
# 所以这里不再重复一份。

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


# ── 破坏性工具必须进入 L3，不能拿 LOW 直接放行 ──

class _CountingRiskModel:
    """记录被调用次数，返回 low 风险（避免触发 L4 人工确认）。"""

    def __init__(self):
        self.calls = 0

    async def chat(self, **kwargs):
        self.calls += 1
        return '{"risk": "low", "reason": "test"}'


def test_self_check_escalates_destructive_tools():
    sc = ToolSelfCheck()
    assert sc.check(ToolCall("Write", {"file_path": "a.py"}),
                    is_destructive=True) == RiskLevel.MEDIUM
    assert sc.check(ToolCall("Edit", {"file_path": "a.py"}),
                    is_destructive=True) == RiskLevel.MEDIUM
    # 只读工具不受影响
    assert sc.check(ToolCall("Read", {"file_path": "a.py"})) == RiskLevel.LOW


@pytest.mark.asyncio
async def test_write_reaches_ai_classifier_but_read_does_not():
    """Write 此前在 L2 拿 LOW 就自动放行，L3/L4 只对 Bash 生效。"""
    model = _CountingRiskModel()
    pm = PermissionManager(model=model)

    assert await pm.authorize(ToolCall("Read", {"file_path": "a.py"}),
                              is_destructive=False) is True
    assert model.calls == 0, "只读工具不该调用 AI 分类器"

    assert await pm.authorize(ToolCall("Write", {"file_path": "a.py"}),
                              is_destructive=True) is True
    assert model.calls == 1, "写文件必须过一次 AI 风险分类"


@pytest.mark.asyncio
async def test_edit_reaches_human_approval_when_classifier_says_high():
    """L3 判高风险 → 进入 L4 人工确认；无输入时 fail-closed 拒绝。"""
    class HighModel:
        async def chat(self, **kwargs):
            return '{"risk": "high", "reason": "writes code"}'

    pm = PermissionManager(model=HighModel())
    # pytest 下 stdin 为 EOF → _request_approval 捕获 EOFError → False
    assert await pm.authorize(ToolCall("Edit", {"file_path": "a.py"}),
                              is_destructive=True) is False
