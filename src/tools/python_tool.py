"""Small sandbox-like Python execution tool for calculations and data analysis."""

from __future__ import annotations

import subprocess
import sys

from langchain_core.tools import tool


@tool
def run_python(code: str) -> str:
    """Run short Python code for calculations or data analysis.

    Use print() to return a value. Standard-library modules plus requests, BeautifulSoup,
    NumPy, and pandas are available when installed in the active project environment.
    """
    try:
        completed = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return "Python execution timed out after 15 seconds."

    output = completed.stdout.strip()
    errors = completed.stderr.strip()
    if completed.returncode:
        return f"Python failed:\n{errors}"
    return output or "Python completed with no output."