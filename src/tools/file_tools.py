"""Deterministic local data-file inspection tools for the file agent."""

from __future__ import annotations

import re

from langchain_core.tools import tool

from .file_reader import ROOT, read_file

_FILE_REFERENCE = re.compile(
    r"[\w./\\-]+\.(?:csv|xlsx|xls|json|txt|md|py|pdf|docx|pptx)\b", re.IGNORECASE
)
_TABULAR_EXTENSIONS = {".csv", ".xlsx", ".xls"}
_DOCUMENT_EXTENSIONS = {".pdf", ".docx", ".pptx"}


def find_referenced_file(text: str) -> str | None:
    """Return the first local file path mentioned in a request, if present."""
    match = _FILE_REFERENCE.search(text)
    return match.group(0) if match else None


@tool
def read_data_file(path: str) -> str:
    """Read a local project file. Uses pandas for CSV/Excel and plain text reading otherwise."""
    requested_path = (ROOT / path).resolve()
    if not requested_path.is_relative_to(ROOT):
        return "Only files within the project root can be read."
    if not requested_path.is_file():
        return f"File not found: {path}"

    suffix = requested_path.suffix.lower()
    if suffix in _TABULAR_EXTENSIONS:
        try:
            import pandas as pd
        except ImportError:
            return "Reading spreadsheet files requires pandas and openpyxl."
        try:
            frame = (
                pd.read_csv(requested_path)
                if suffix == ".csv"
                else pd.read_excel(requested_path)
            )
        except Exception as error:
            return f"Unable to read {path}: {error}"
        return (
            f"Shape: {frame.shape[0]} rows x {frame.shape[1]} columns\n"
            f"Columns: {list(frame.columns)}\n\n"
            f"{frame.head(20).to_string()}"
        )

    if suffix in _DOCUMENT_EXTENSIONS:
        try:
            from unstructured.partition.auto import partition
        except ImportError:
            return (
                f"Reading {suffix} files requires the optional 'unstructured' package. "
                "Install it to enable this file type."
            )
        try:
            elements = partition(filename=str(requested_path))
        except Exception as error:
            return f"Unable to read {path}: {error}"
        return "\n".join(str(element) for element in elements)[:12000]

    return read_file.invoke({"path": path})
