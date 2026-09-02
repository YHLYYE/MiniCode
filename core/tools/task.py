"""Task management tools"""
from pathlib import Path

from core.tools.base import Tool


class TodoWriteTool(Tool):
    """Create and update a structured task list"""

    name = "TodoWrite"
    description = "Create and update a task list to track progress"
    input_schema = {
        "todos": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "content": {"type": "string"},
                    "status": {
                        "type": "string",
                        "enum": ["pending", "in_progress", "completed"],
                    },
                },
            },
            "description": "List of todo items with status",
        },
    }
    is_readonly = False
    is_concurrency_safe = False

    async def execute(self, todos: list[dict]) -> str:
        icons = {
            "pending": "☐",
            "in_progress": "◉",
            "completed": "✓",
        }
        lines = ["## Task List"]
        for todo in todos:
            icon = icons.get(todo.get("status", "pending"), "☐")
            lines.append(f"{icon} {todo.get('content', '')}")

        result = "\n".join(lines)
        Path(".minicode/todos.md").parent.mkdir(parents=True, exist_ok=True)
        Path(".minicode/todos.md").write_text(result, encoding="utf-8")
        return result
