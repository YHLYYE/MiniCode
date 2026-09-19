"""集成测试 — 验证 main.py 接线完整性。

防止「能力/工具定义了但没接进工具集」这种问题复发：直接断言
_build_tools 产出的工具名集合，新增工具时若忘了接进 main.py，这里会报错。
"""
from config import Config
from main import _build_tools


def test_normal_mode_registers_all_tools(tmp_path, monkeypatch):
    """normal 模式应注册全部 8 个工具"""
    monkeypatch.chdir(tmp_path)  # 避免在真实项目目录创建 .minicode
    tools, _, _, memory_manager = _build_tools("normal", Config())
    try:
        assert {t.name for t in tools} == {
            "Read", "Write", "Bash", "TodoWrite",
            "Skill", "RecallMemory", "Remember", "Agent",
        }
    finally:
        memory_manager.close()


def test_plan_mode_keeps_only_readonly_tools(tmp_path, monkeypatch):
    """plan 模式只保留只读工具"""
    monkeypatch.chdir(tmp_path)
    tools, _, _, memory_manager = _build_tools("plan", Config())
    try:
        assert {t.name for t in tools} == {"Read", "Skill", "RecallMemory"}
    finally:
        memory_manager.close()
