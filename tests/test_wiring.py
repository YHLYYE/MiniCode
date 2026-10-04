"""集成测试 — 验证 main.py 接线完整性。

防止「能力/工具定义了但没接进工具集」这种问题复发：直接断言
_build_tools 产出的工具名集合，新增工具时若忘了接进 main.py，这里会报错。
"""
from config import Config
from main import _build_tools


def _unpack(mode):
    """_build_tools 现在多返回一个 system_prompt_factory（二阶段路由接线）。"""
    tools, system_prompt, exec_mode, memory_manager, prompt_factory = (
        _build_tools(mode, Config())
    )
    return tools, system_prompt, exec_mode, memory_manager, prompt_factory


def test_normal_mode_registers_all_tools(tmp_path, monkeypatch):
    """normal 模式应注册全部 13 个工具"""
    monkeypatch.chdir(tmp_path)  # 避免在真实项目目录创建 .minicode
    tools, _, _, memory_manager, _ = _unpack("normal")
    try:
        assert {t.name for t in tools} == {
            "Read", "Write", "Edit", "Bash", "Grep", "Glob",
            "WebSearch", "WebFetch", "TodoWrite",
            "Skill", "RecallMemory", "Remember", "Agent",
        }
    finally:
        memory_manager.close()


def test_plan_mode_keeps_only_readonly_tools(tmp_path, monkeypatch):
    """plan 模式只保留只读工具 + TodoWrite（它只写 agent 自己的记账文件）"""
    monkeypatch.chdir(tmp_path)
    tools, _, _, memory_manager, _ = _unpack("plan")
    try:
        assert {t.name for t in tools} == {
            "Read", "Grep", "Glob", "WebSearch", "WebFetch",
            "Skill", "RecallMemory",
            "TodoWrite",  # 只读模式也要能写待办清单，否则「规划模式」写不了计划
        }
    finally:
        memory_manager.close()


def test_build_tools_returns_routing_prompt_factory(tmp_path, monkeypatch):
    """每任务重建 System Prompt 的工厂必须接出来，否则路由等于没接。"""
    monkeypatch.chdir(tmp_path)
    _, _, _, memory_manager, prompt_factory = _unpack("normal")
    try:
        assert callable(prompt_factory)
        relevant = prompt_factory("帮我审查一下这段代码有没有bug")
        irrelevant = prompt_factory("量子计算的哈密顿量怎么对角化")
        assert "## Task Routing" in relevant
        assert "code-review" in relevant
        assert "## Task Routing" not in irrelevant
    finally:
        memory_manager.close()
