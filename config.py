"""MiniCode configuration management"""
import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    model: str = "claude-sonnet-4-6"
    max_turns: int = 20
    max_output_tokens: int = 8192
    max_cost_usd: float = 5.0
    project_root: Path = field(default_factory=Path.cwd)

    @classmethod
    def from_env(cls) -> "Config":
        from dotenv import load_dotenv
        load_dotenv()
        return cls(
            model=os.getenv("MINICODE_MODEL", "claude-sonnet-4-6"),
            max_cost_usd=float(os.getenv("MINICODE_MAX_COST", "5.0")),
        )
