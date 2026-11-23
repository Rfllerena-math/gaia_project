"""Tool for inspecting local files supplied with GAIA-style tasks."""

from __future__ import annotations

from pathlib import Path

from langchain_core.tools import tool

ROOT = Path(__file__).resolve().parents[1]


@tool
def read_file(path: str) -> str:
    """Read a UTF-8 text file within this project. Paths must be relative to the project root."""
    requested_path = (ROOT / path).resolve()
    if not requested_path.is_relative_to(ROOT):
        return "Only files within the project root can be read."
    if not requested_path.is_file():
        return f"File not found: {path}"
    try:
        return requested_path.read_text(encoding="utf-8")[:20000]
    except UnicodeDecodeError:
        return f"{path} is not a UTF-8 text file."