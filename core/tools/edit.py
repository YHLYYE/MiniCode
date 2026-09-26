"""Edit tool — precise string replacement (surgical edit)."""
from core.tools.base import Tool
from core.tools.fs_backend import LocalFilesystemBackend


class EditTool(Tool):
    """Make a precise edit to a file by replacing an exact string.

    Unlike Write (which overwrites the whole file), Edit replaces a single
    occurrence of old_string. This is the surgical-edit primitive a coding
    agent needs to change a few lines without rewriting the file.
    """

    name = "Edit"
    description = (
        "Make a precise edit to a file by replacing an exact string. "
        "old_string must match exactly once in the file — include a few "
        "lines of surrounding context to make it unique."
    )
    input_schema = {
        "file_path": {
            "type": "string",
            "description": "Path to the file to edit",
        },
        "old_string": {
            "type": "string",
            "description": "Exact text to replace (must occur exactly once)",
        },
        "new_string": {
            "type": "string",
            "description": "Replacement text",
        },
    }
    is_readonly = False
    is_concurrency_safe = False
    is_destructive = True

    def __init__(self, backend=None):
        super().__init__()
        self._backend = backend or LocalFilesystemBackend()

    async def execute(self, file_path: str, old_string: str,
                      new_string: str) -> str:
        if not self._backend.exists(file_path):
            return f"Error: File not found: {file_path}"
        if not old_string:
            return "Error: old_string must not be empty"

        try:
            content = self._backend.read(file_path)
        except Exception as e:
            return f"Error reading {file_path}: {e}"

        count = content.count(old_string)
        if count == 0:
            return f"Error: old_string not found in {file_path}"
        if count > 1:
            return (
                f"Error: old_string found {count} times (must be unique). "
                "Include more surrounding context to disambiguate."
            )

        new_content = content.replace(old_string, new_string, 1)
        try:
            self._backend.write(file_path, new_content)
        except Exception as e:
            return f"Error writing {file_path}: {e}"

        return f"Edited {file_path} ({len(old_string)} -> {len(new_string)} chars)"
