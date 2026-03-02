"""Sandboxed LangChain tools available to agent workers.

``run_python`` executes small, import-free snippets in a restricted
namespace.  ``get_current_utc_time`` returns the current UTC timestamp.
Both tools are safe to expose to LLM tool-calling without file-system or
network access.
"""

import ast
import contextlib
import io
import logging
from datetime import datetime, timezone
from typing import Any, Dict

from langchain_core.tools import tool

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Time tool
# ---------------------------------------------------------------------------


@tool
def get_current_utc_time() -> str:
    """Return the current UTC time in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Python sandbox tool
# ---------------------------------------------------------------------------


@tool
def run_python(code: str) -> str:
    """Execute a small Python snippet with no imports.

    Accepts only whitelisted built-in functions (print, range, len, sum,
    min, max, round).  Rejects any use of imports, dunder attributes,
    attribute access, ``open``, ``exec``, ``eval``, ``os``, or ``sys``.

    Args:
        code: Python source code to execute.

    Returns:
        Captured stdout, or ``"OK"`` if stdout is empty, or an error
        message prefixed with ``"Rejected:"`` for unsafe inputs.
    """
    _BLOCKED_TOKENS = ("import", "__", "open(", "exec(", "eval(", "os.", "sys.")
    if any(token in code for token in _BLOCKED_TOKENS):
        return "Rejected: unsafe code."

    tree = ast.parse(code)
    _ALLOWED_FUNCS = {"print", "range", "len", "sum", "min", "max", "round"}

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            return "Rejected: imports not allowed."
        if isinstance(node, ast.Attribute):
            return "Rejected: attribute access not allowed."
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id not in _ALLOWED_FUNCS:
                return f"Rejected: function {node.func.id!r} not allowed."

    safe_globals: Dict[str, Any] = {
        "__builtins__": {
            "print": print,
            "range": range,
            "len": len,
            "sum": sum,
            "min": min,
            "max": max,
            "round": round,
        }
    }
    safe_locals: Dict[str, Any] = {}
    stdout = io.StringIO()

    with contextlib.redirect_stdout(stdout):
        exec(  # noqa: S102
            compile(tree, filename="<exec>", mode="exec"),
            safe_globals,
            safe_locals,
        )

    output = stdout.getvalue().strip()
    return output or "OK"
