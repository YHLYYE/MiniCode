"""Tool base class — fail-closed defaults for safety"""
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class ToolCall:
    """Represents a pending tool call"""
    name: str
    input: dict


class Tool(ABC):
    """Abstract base for all tools.

    Safety defaults (fail-closed):
    - is_readonly=False: Assumes tool modifies state → triggers permission check
    - is_concurrency_safe=False: Assumes side effects → serial execution
    - is_destructive=False: Safe default, override for destructive tools
    """

    name: str = ""
    description: str = ""
    input_schema: dict = {}

    # Safety declarations — fail-closed: safest is the default
    is_readonly: bool = False
    is_concurrency_safe: bool = False
    is_destructive: bool = False

    @abstractmethod
    async def execute(self, **kwargs) -> str:
        """Execute the tool with the given arguments. Returns result string."""
        ...

    def check_concurrency_safe(self, input: dict) -> bool:
        """Runtime check: can this tool run concurrently with others?
        Override for dynamic behavior based on input parameters."""
        return self.is_concurrency_safe

    def to_schema(self) -> dict:
        """Convert tool to Anthropic-compatible JSON Schema.

        Only parameters WITHOUT a default in execute() are marked required
        (带默认值的参数不应被强制要求).
        """
        import inspect
        required = []
        for name, param in inspect.signature(self.execute).parameters.items():
            if name == "self":
                continue
            if param.default is inspect.Parameter.empty:
                required.append(name)
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": self.input_schema,
                "required": required,
            },
        }


class SkillTool(Tool):
    """Load full instructions for a skill. Called by the model when it
    determines that a skill is relevant to the current task.

    This is a context transformer, not a traditional tool. Its "output"
    is the skill's full instructions, which are injected as a tool_result
    message. The System Prompt never changes → Prompt Cache stays hot.
    """

    name = "Skill"
    description = "Load full instructions for a skill. Call this before using a skill."
    input_schema = {
        "name": {
            "type": "string",
            "description": "Name of the skill to load (see Available skills list)",
        },
    }
    is_readonly = True
    is_concurrency_safe = True

    def __init__(self, skill_system=None):
        super().__init__()
        self._skills = skill_system

    async def execute(self, name: str) -> str:
        if self._skills is None:
            return "Skill system not initialized."
        instructions = self._skills.activate(name)
        if instructions is None:
            # 精确名未命中 → 用二阶段路由模糊匹配，推荐最接近的 skill
            suggestions = self._skills.route(name, top_k=3)
            if suggestions:
                lines = [f"Skill '{name}' not found. Closest matches:"]
                for skill, confidence in suggestions:
                    lines.append(
                        f"  - {skill.name} (置信度 {confidence:.2f}): "
                        f"{skill.description}"
                    )
                return "\n".join(lines)
            available = ", ".join(self._skills.list_skills())
            return (
                f"Skill '{name}' not found.\n"
                f"Available skills: {available}"
            )
        # 技能可以在 frontmatter 里声明 allowed-tools。这只是**声明**：
        # 工具表是循环级的，技能无法在运行时收窄它，硬约束来自工具表 + 权限四层。
        # 把它显式写进上下文，是让模型的行为更贴合技能作者的意图。
        skill = self._skills.get(name)
        declared = getattr(skill, "allowed_tools", None)
        if declared:
            return (
                f"[本技能声明使用的工具：{', '.join(declared)}"
                f" —— 声明而非强制，真正的工具边界由工具表与权限四层保证]\n\n"
                f"{instructions}"
            )
        return instructions


class RecallMemoryTool(Tool):
    """Search past session memories for relevant patterns.

    The model calls this when it encounters a problem that it suspects
    has been seen before (e.g., "this NPE looks familiar, let me check
    past fixes"). Results are injected as a tool_result — no Prompt
    Cache impact.
    """

    name = "RecallMemory"
    description = (
        "Search past sessions for similar issues, fix patterns, or "
        "project conventions. Use when encountering a problem you've "
        "seen before."
    )
    input_schema = {
        "query": {
            "type": "string",
            "description": "What to search for (e.g., 'NPE fix pattern')",
        },
        "top_k": {
            "type": "integer",
            "description": "Number of results (default 5, max 10)",
        },
    }
    is_readonly = True
    is_concurrency_safe = True

    def __init__(self, memory_manager=None):
        super().__init__()
        self._memory = memory_manager

    async def execute(self, query: str, top_k: int = 5) -> str:
        if self._memory is None:
            return "Memory system not initialized."
        top_k = min(top_k, 10)
        entries = await self._memory.search(query, top_k)
        if not entries:
            return "No relevant memories found."

        lines = [f"Found {len(entries)} relevant memories:\n"]
        for e in entries:
            files = ", ".join(e.file_paths) if e.file_paths else "none"
            lines.append(
                f"[{e.type}] {e.content}\n"
                f"  Files: {files}\n"
                f"  Tags: {', '.join(e.tags)}"
            )
        return "\n".join(lines)


class RememberTool(Tool):
    """Persist knowledge to long-term memory across sessions.

    Complements RecallMemoryTool (read) with an explicit write path.
    The model calls this to save a reusable pattern (procedural), an
    event (episodic), or a user preference (user_profile).
    """

    name = "Remember"
    description = (
        "Persist knowledge to long-term memory for future sessions. "
        "Pick memory_type from the parameter description (each lists examples "
        "and the boundary between the three)."
    )
    input_schema = {
        "memory_type": {
            "type": "string",
            "enum": ["procedural", "episodic", "user_profile"],
            # 为什么写这么长：早先只有一句 "Which memory category to write to"，
            # 模型只能靠英文名词猜，很容易把"可照做的规则"写成"发生过的日志"——
            # 而类型会连带决定检索算法（程序性走 BM25、情景走余弦），选错就捞不回来。
            # 所以这里给**本项目自己的例子**（中文 + 真实形态），而不是抽象定义。
            "description": (
                "Which category to write, with examples from this project:\n"
                "- procedural —— 以后要照做的规则/修法，例如："
                "「跑测试前必须设 MINICODE_OFFLINE=1」「改完索引要重建再跑测试」"
                "「Edit 报 NameError: name 're' → 补 import re」\n"
                "- episodic —— 发生过的事（事件、报错、任务轨迹），例如："
                "「Tool errors: - Bash: NameError re」「Task: 修 skill 优先级 | 4 turns」\n"
                "- user_profile —— 稳定的用户/项目属性，键值对，例如 "
                "key='naming_style' value='snake_case'；**必须带 key**\n"
                "边界：要「以后照做」→ procedural；要「复盘发生了什么」→ episodic；"
                "稳定属性/偏好 → user_profile。同一次失败可以写两条"
                "（一条 episodic 日志 + 一条 procedural 修法）。"
            ),
        },
        "content": {
            "type": "string",
            "description": "The knowledge / value to remember",
        },
        "key": {
            "type": "string",
            "description": "Required for user_profile: the preference key "
                           "(e.g. 'naming_style')",
        },
    }
    is_readonly = False
    is_concurrency_safe = True

    def __init__(self, memory_manager=None):
        super().__init__()
        self._memory = memory_manager

    async def execute(self, memory_type: str, content: str,
                      key: str | None = None) -> str:
        if self._memory is None:
            return "Memory system not initialized."
        try:
            if memory_type == "procedural":
                await self._memory.record_procedural(content)
            elif memory_type == "episodic":
                await self._memory.record_episodic(content)
            elif memory_type == "user_profile":
                if not key:
                    return "Error: 'key' is required for user_profile memory."
                await self._memory.record_user_profile(key, content)
            else:
                return f"Error: unknown memory_type '{memory_type}'"
        except Exception as e:
            return f"Failed to record memory: {e}"
        return f"Recorded [{memory_type}] memory."
