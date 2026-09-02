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
