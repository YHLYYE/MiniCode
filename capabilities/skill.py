"""Skill system — context transformer wrapped in tool protocol

Reference: Claude Code's Skill system (how-claude-code-works, ch09)
Core insight: Skills are NOT an independent subsystem. They are context
modifiers wrapped in the same tool protocol — the Skill tool's output
is injected as a tool_result at the tail of the message list, so the
System Prompt stays unchanged and the server-side prefix cache keeps hitting
（命中量从 provider 读回，见 model_adapter._usage_from；不再写"100% 命中"，
因为那是没测过的断言）。
"""

from dataclasses import dataclass, field
from pathlib import Path

from capabilities.tokenize import keyword_tokens


# 兼容旧引用（测试与外部都按 _tokenize 这个名字用）
_tokenize = keyword_tokens



@dataclass
class Skill:
    """A skill definition loaded from a SKILL.md file"""
    name: str
    description: str
    full_instructions: str
    # 声明式的"这个技能预期用哪些工具"。注意：它**不裁剪工具表** ——
    # 真正的硬约束是循环级的工具表 + 权限四层。此字段只在激活时提示给模型，
    # 让行为更贴合技能作者的意图（见 core/tools/base.py 的 SkillTool）。
    allowed_tools: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    boundary: str = ""
    examples: list[str] = field(default_factory=list)
    priority: int = 0


class SkillSystem:
    """Manages skill registration, discovery, and activation.

    Three sources (priority order):
    1. Project-level: skills/ directory in project root
    2. User-level: ~/.minicode/skills/
    3. Built-in: registered programmatically
    """

    def __init__(self):
        self._registry: dict[str, Skill] = {}

    # ── Registration ──

    def register_from_source(self, source_dir: Path, priority: int = 0):
        """Recursively load all .md files from a directory as skills.

        Each .md file should have YAML frontmatter:
        ---
        name: skill-name
        description: One-line summary
        allowed-tools: [ToolA, ToolB]
        ---
        ...skill instructions...
        """
        if not source_dir.exists():
            return
        for md_path in source_dir.rglob("*.md"):
            skill = self._parse_skill_md(md_path, priority)
            if skill:
                self._register(skill)

    def register_builtin(self, skill: Skill):
        """Register a skill programmatically (同样按 priority 裁决同名覆盖)"""
        self._register(skill)

    def _register(self, skill: Skill):
        """注册并裁决同名覆盖：priority 高的赢；相同则后注册的赢。

        为什么需要这一步：早先这里是直接 `self._registry[skill.name] = skill`，
        于是 priority 只在没人调用的 resolve() 里被读到 —— 等于装饰字段。
        "项目级覆盖内置"实际是靠**注册顺序**（main.py 先内置、后项目级）碰巧成立的，
        注册顺序一改，覆盖方向就反过来，而代码里没有任何保护。
        """
        existing = self._registry.get(skill.name)
        if existing is not None and skill.priority < existing.priority:
            # 优先级更低 → 不覆盖已注册的那份
            return
        self._registry[skill.name] = skill

    # ── Discovery (cheap index for System Prompt) ──

    def get_index_for_system_prompt(self) -> str:
        """Generate a one-line-per-skill index for the System Prompt.

        This is the "cheap ad" — model sees names + descriptions and
        decides which skill to activate. Full instructions are loaded
        on-demand via activate().
        """
        if not self._registry:
            return ""
        lines = ["Available skills:"]
        for skill in self._registry.values():
            lines.append(f"  - {skill.name}: {skill.description}")
        lines.append(
            '\nTo use a skill, call the Skill tool with name="<skill-name>" '
            "to load its full instructions."
        )
        return "\n".join(lines)

    # ── Activation (expensive — loaded on demand) ──

    def activate(self, skill_name: str) -> str | None:
        """Load full instructions for a skill.

        Returns None if skill not found. The caller injects the returned
        text as a tool_result message at the tail of the conversation.
        """
        skill = self._registry.get(skill_name)
        if skill is None:
            return None
        return skill.full_instructions

    # ── Lookup ──

    def get(self, name: str) -> Skill | None:
        """按名字取技能。同名覆盖已在注册时按 priority 裁决，这里只做查表。"""
        return self._registry.get(name)

    def list_skills(self) -> list[str]:
        """Return all registered skill names"""
        return list(self._registry.keys())

    # ── Two-stage routing (召回 → 精排) ──

    # rerank confidence floor. Calibrated against skills/: real matches land
    # at 0.43-1.00, incidental tag hits at 0.00-0.13. Below the floor the
    # signal is noise, so the hint is omitted entirely rather than shown.
    ROUTING_MIN_CONFIDENCE = 0.30

    def routing_hint(self, task: str, top_k: int = 3,
                     min_confidence: float | None = None) -> str:
        """Format the two-stage routing result for the System Prompt tail.

        Called once per task by the Agent Loop. Returns "" when nothing is
        relevant enough. The hint is a nudge, not a restriction — the model
        may still load a different skill, or none.
        """
        if not task or not self._registry:
            return ""

        floor = (self.ROUTING_MIN_CONFIDENCE if min_confidence is None
                 else min_confidence)
        ranked = [
            (skill, confidence)
            for skill, confidence in self.route(task, top_k=top_k)
            if confidence >= floor
        ]
        if not ranked:
            return ""

        lines = [
            "## Task Routing",
            "Two-stage routing (recall → rerank) suggests the following skills "
            "for the current task. This is a suggestion, not a restriction.",
        ]
        for skill, confidence in ranked:
            lines.append(
                f"- {skill.name} (confidence {confidence:.2f}): "
                f"{skill.description}"
            )
        lines.append(
            'Load one with the Skill tool: name="<skill-name>".'
        )
        return "\n".join(lines)

    def route(self, task: str, top_k: int = 3) -> list[tuple[Skill, float]]:
        """Two-stage skill routing for task → skill matching.

        Stage 1 (召回 Recall): keyword + tag overlap → Top-10 candidates
        Stage 2 (精排 Rerank): boundary + example similarity scoring → Top-K

        Returns list of (Skill, confidence_score) sorted by score descending.
        This complements the model's autonomous choice: when there are many
        skills (>15), routing pre-filters to reduce selection noise.
        """
        if not self._registry:
            return []

        # Stage 1: Recall — score every skill by keyword overlap
        recall_candidates = self._recall(task)

        if not recall_candidates:
            return []

        # Stage 2: Rerank — refine top candidates by boundary/examples
        reranked = self._rerank(task, recall_candidates)

        return reranked[:top_k]

    def _recall(self, task: str, top_k: int = 10) -> list[tuple[Skill, float]]:
        """Stage 1: keyword overlap scoring against description + tags + examples.

        Uses _tokenize() which handles both Chinese (character bigrams) and
        English (whitespace words).
        """
        task_tokens = _tokenize(task)

        scored = []
        for skill in self._registry.values():
            score = 0.0

            # Description token overlap
            desc_tokens = _tokenize(skill.description)
            score += len(task_tokens & desc_tokens) * 2.0

            # Tag matches are strong signals (substring or token match)
            for tag in skill.tags:
                if tag.lower() in task.lower() or _tokenize(tag) & task_tokens:
                    score += 3.0

            # Example similarity
            for ex in skill.examples:
                ex_tokens = _tokenize(ex)
                score += len(task_tokens & ex_tokens) * 0.5

            if score > 0:
                scored.append((skill, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def _rerank(self, task: str, candidates: list[tuple[Skill, float]]) -> list[tuple[Skill, float]]:
        """Stage 2: refine ranking using boundary fit and example similarity.

        Rules:
        - If task clearly falls outside skill.boundary, down-rank it
        - Boost skills whose examples closely match the task
        """
        task_tokens = _tokenize(task)
        reranked = []

        for skill, base_score in candidates:
            score = base_score

            # Boundary penalty: if boundary keywords contradict task, reduce score
            if skill.boundary:
                boundary_tokens = _tokenize(skill.boundary)
                if boundary_tokens & task_tokens:
                    score -= 2.0

            # Example boost: strong example overlap boosts confidence
            if skill.examples:
                best_example_overlap = 0
                for ex in skill.examples:
                    ex_tokens = _tokenize(ex)
                    best_example_overlap = max(
                        best_example_overlap, len(task_tokens & ex_tokens)
                    )
                score += best_example_overlap * 0.8

            # Normalize to 0-1 confidence
            confidence = max(0.0, min(1.0, score / 10.0))
            reranked.append((skill, confidence))

        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked

    # ── Parsing ──

    def _parse_skill_md(self, path: Path, priority: int) -> Skill | None:
        """Parse a SKILL.md file with YAML frontmatter."""
        try:
            content = path.read_text(encoding="utf-8")
        except Exception:
            return None

        lines = content.split("\n")
        name = ""
        description = ""
        allowed_tools: list[str] = []
        tags: list[str] = []
        boundary = ""
        examples: list[str] = []

        # Parse YAML frontmatter if present
        if lines and lines[0].strip() == "---":
            end = 1
            while end < len(lines) and lines[end].strip() != "---":
                line = lines[end]
                if ":" in line:
                    key, _, val = line.partition(":")
                    key = key.strip()
                    val = val.strip()
                    if key == "name":
                        name = val
                    elif key == "description":
                        description = val
                    elif key == "allowed-tools":
                        val_clean = val.strip("[]").strip()
                        allowed_tools = [
                            t.strip() for t in val_clean.split(",") if t.strip()
                        ]
                    elif key == "tags":
                        val_clean = val.strip("[]").strip()
                        tags = [t.strip() for t in val_clean.split(",") if t.strip()]
                    elif key == "boundary":
                        boundary = val
                    elif key == "examples":
                        # Format: [ex1, ex2] or multi-line
                        val_clean = val.strip("[]").strip()
                        examples = [e.strip() for e in val_clean.split(",") if e.strip()]
                end += 1
            body_start = end + 1
        else:
            body_start = 0

        if not name:
            return None

        full_instructions = "\n".join(lines[body_start:]).strip()

        return Skill(
            name=name,
            description=description,
            full_instructions=full_instructions,
            allowed_tools=allowed_tools,
            tags=tags,
            boundary=boundary,
            examples=examples,
            priority=priority,
        )
