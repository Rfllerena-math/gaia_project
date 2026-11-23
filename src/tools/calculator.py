"""Safe arithmetic calculator tool for reasoning and verification agents."""

from __future__ import annotations

import ast
import operator

from langchain_core.tools import tool

_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _evaluate(node: ast.expr) -> float:
    """Evaluate one arithmetic AST node, rejecting anything beyond basic math."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
        return _OPERATORS[type(node.op)](_evaluate(node.left), _evaluate(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPERATORS:
        return _OPERATORS[type(node.op)](_evaluate(node.operand))
    raise ValueError("Unsupported expression")


@tool
def calculator(expression: str) -> str:
    """Evaluate one arithmetic expression using +, -, *, /, //, %, ** and parentheses only."""
    try:
        tree = ast.parse(expression, mode="eval")
        result = _evaluate(tree.body)
    except Exception as error:
        return f"Calculator failed: {error}"
    return str(result)
