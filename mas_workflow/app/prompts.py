"""读取 prompts/*.md 中的 agent system prompt。"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path


PROMPT_DIR = Path(__file__).resolve().parents[1] / "prompts"


@lru_cache(maxsize=32)
def load_prompt(name: str) -> str:
    """按文件名读取 prompt，例如 load_prompt('planner')。"""
    path = PROMPT_DIR / f"{name}.md"
    if not path.exists():
        raise FileNotFoundError(f"缺少 prompt 文件: {path}")
    return path.read_text(encoding="utf-8")

