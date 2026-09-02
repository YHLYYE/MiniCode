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
        """Convert tool to Anthropic-compatible JSON Schema"""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": self.input_schema,
                "required": list(self.input_schema.keys()),
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
            available = ", ".join(self._skills.list_skills())
            return (
                f"Skill '{name}' not found.\n"
                f"Available skills: {available}"
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
