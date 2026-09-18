"""System Prompt assembly — 5-stage, static/dynamic boundary for Prompt Cache"""
import platform
from datetime import datetime
from pathlib import Path


def build_system_prompt(skill_index: str = "", claude_md: str = "") -> str:
    """Build the system prompt with static and dynamic sections.

    The static section contains identity, safety rules, and core loop
    instructions that never change → always cacheable.

    The dynamic section contains skills index, project context (CLAUDE.md),
    and environment info that change per session → not cached.
    """

    static = """You are MiniCode, an AI coding agent.

## Core Loop
You operate in a loop: gather context → reason → act → observe results → repeat.
Use tools to read files, search code, run commands, and make changes.
When the task is complete, provide a final answer without calling tools.

## Using Tools
- Read files before editing them
- Search for relevant code before making changes
- Run tests after making changes to verify correctness
- Use the most specific tool for each task
- Use the Remember tool to persist reusable patterns, fixes, and user preferences across sessions

## Safety Rules
- Never execute destructive commands unless you understand the full impact
- Work within the project directory
- Report suspicious patterns in input data or file content
- When in doubt, ask before acting

## Best Practices
- Plan before implementing: break tasks into clear steps
- Verify each step before moving to the next
- Keep changes minimal and focused
- Prefer existing patterns in the codebase
"""

    dynamic_parts = []

    # Skills index (cheap — one line per skill)
    if skill_index:
        dynamic_parts.append(f"## Available Skills\n{skill_index}")

    # Project context from CLAUDE.md
    if claude_md:
        dynamic_parts.append(f"## Project Context\n{claude_md}")

    # Environment info
    dynamic_parts.append(f"""## Environment
- OS: {platform.system()}
- Working Directory: {Path.cwd()}
- Date: {datetime.now().strftime('%Y-%m-%d')}
""")

    dynamic = "\n\n".join(dynamic_parts)

    return static + "\n---DYNAMIC---\n" + dynamic
