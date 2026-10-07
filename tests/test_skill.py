"""Skill 二阶段路由测试 — 召回 + 精排"""
import pytest
from capabilities.skill import SkillSystem, Skill, _tokenize


@pytest.fixture
def skill_system():
    ss = SkillSystem()
    ss.register_builtin(Skill(
        name="code_review",
        description="Review code for bugs and security issues 审查代码缺陷和安全问题",
        full_instructions="You are a code reviewer.",
        tags=["review", "code", "security", "审查", "代码"],
        examples=["review this code for bugs", "审查代码找bug"],
    ))
    ss.register_builtin(Skill(
        name="run_tests",
        description="Run test suite and report results 运行测试套件",
        full_instructions="You run tests.",
        tags=["test", "run", "pytest", "测试"],
        examples=["run the tests", "运行测试"],
    ))
    ss.register_builtin(Skill(
        name="refactor",
        description="Refactor and clean up code structure 重构代码结构",
        full_instructions="You refactor code.",
        tags=["refactor", "cleanup", "重构"],
        examples=["refactor this function"],
    ))
    return ss


def test_tokenize_english():
    tokens = _tokenize("review the code")
    assert "review" in tokens
    assert "code" in tokens


def test_tokenize_chinese():
    tokens = _tokenize("审查代码")
    assert "审查" in tokens  # character bigram


def test_route_english(skill_system):
    results = skill_system.route("review the code for bugs", top_k=1)
    assert results[0][0].name == "code_review"
    assert results[0][1] > 0.5


def test_route_chinese(skill_system):
    results = skill_system.route("帮我审查一下这段代码有没有bug", top_k=1)
    assert results[0][0].name == "code_review"


def test_route_chinese_tests(skill_system):
    results = skill_system.route("运行测试套件看看结果", top_k=1)
    assert results[0][0].name == "run_tests"


def test_route_no_match(skill_system):
    """No matching skill → empty result"""
    results = skill_system.route("量子计算无关任务", top_k=3)
    assert results == []


def test_route_confidence_descending(skill_system):
    """Results are sorted by confidence descending"""
    results = skill_system.route("review code", top_k=3)
    confidences = [c for _, c in results]
    assert confidences == sorted(confidences, reverse=True)


def test_route_returns_skills_and_scores(skill_system):
    """Each result is a (Skill, float) tuple"""
    results = skill_system.route("review code", top_k=2)
    for skill, score in results:
        assert isinstance(skill, Skill)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0


def test_real_skills_load_and_route():
    """真实 skills/ 目录下的 skill 能被加载，且 route() 路由到正确技能"""
    import pathlib

    skills_dir = pathlib.Path(__file__).parent.parent / "skills"
    ss = SkillSystem()
    ss.register_from_source(skills_dir, priority=10)

    names = set(ss.list_skills())
    assert {
        "code-review", "write-tests", "debug",
        "refactor", "documentation", "security-audit",
    } <= names

    cases = [
        ("review this code for bugs", "code-review"),
        ("write unit tests for this function", "write-tests"),
        ("why is this test failing", "debug"),
        ("refactor this function to be cleaner", "refactor"),
        ("document this module with docstrings", "documentation"),
        ("find security vulnerabilities here", "security-audit"),
    ]
    for task, expected in cases:
        results = ss.route(task, top_k=1)
        assert results, f"route('{task}') 应命中 {expected}"
        assert results[0][0].name == expected, (
            f"route('{task}') 命中 {results[0][0].name}，期望 {expected}"
        )


def test_skill_tool_fuzzy_match():
    """Skill 名未精确命中时，SkillTool 用 route() 推荐最接近的 skill"""
    import asyncio
    import pathlib

    from core.tools.base import SkillTool

    skills_dir = pathlib.Path(__file__).parent.parent / "skills"
    ss = SkillSystem()
    ss.register_from_source(skills_dir, priority=10)
    tool = SkillTool(ss)

    # 空格写法 → 精确名 "code review" 未命中 → 走 route() 模糊匹配
    result = asyncio.run(tool.execute("code review"))
    assert "not found" in result
    assert "code-review" in result


# ── 同名覆盖：priority 必须真的说话 ──
# 回归背景：注册原来是直接 `registry[name] = skill`，于是 priority 只在没人调用的
# resolve() 里被读到 —— 等于装饰字段。"项目级覆盖内置"实际靠注册顺序碰巧成立。

def test_same_name_override_follows_priority():
    ss = SkillSystem()
    ss.register_builtin(Skill(name="x", description="内置版",
                              full_instructions="A", priority=10))
    ss.register_builtin(Skill(name="x", description="项目版",
                              full_instructions="B", priority=20))
    assert ss.get("x").description == "项目版"          # 高优先级覆盖

    ss.register_builtin(Skill(name="x", description="低优先级后注册",
                              full_instructions="C", priority=1))
    assert ss.get("x").description == "项目版"          # 低优先级不许覆盖

    ss.register_builtin(Skill(name="y", description="同优先级先来",
                              full_instructions="D", priority=5))
    ss.register_builtin(Skill(name="y", description="同优先级后来",
                              full_instructions="E", priority=5))
    assert ss.get("y").description == "同优先级后来"    # 同优先级后者赢


def test_real_skills_keep_priority_order_of_sources(tmp_path):
    """内置(10) 先注册、项目级(20) 后注册 → 项目级胜；顺序反过来结论不变。"""
    import pathlib

    skills_dir = pathlib.Path(__file__).parent.parent / "skills"
    for order in ("builtin_first", "project_first"):
        ss = SkillSystem()
        register = [
            (skills_dir, 10),
            (tmp_path, 20),
        ]
        if order == "project_first":
            register.reverse()
        for directory, priority in register:
            if directory is tmp_path:
                override = directory / "code_review.md"
                override.write_text(
                    "---\nname: code-review\ndescription: 项目级覆盖版\n---\n正文",
                    encoding="utf-8",
                )
            ss.register_from_source(directory, priority=priority)
        assert ss.get("code-review").description == "项目级覆盖版", order


def test_activated_skill_announces_declared_tools():
    """allowed-tools 是声明式的：要让模型看见，同时写明它不具强制力。"""
    import asyncio

    from core.tools.base import SkillTool

    ss = SkillSystem()
    ss.register_builtin(Skill(name="with-tools", description="d",
                              full_instructions="正文", allowed_tools=["Read", "Grep"]))
    ss.register_builtin(Skill(name="no-tools", description="d",
                              full_instructions="正文"))
    tool = SkillTool(ss)

    out = asyncio.run(tool.execute("with-tools"))
    assert "Read" in out and "Grep" in out
    assert "声明而非强制" in out

    plain = asyncio.run(tool.execute("no-tools"))
    assert plain == "正文"          # 没声明就不加这段噪音
