"""Reviewer tests: INTERFACES.md code rules (file/function length, docstrings, return hints)."""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAX_FILE_LINES = 250
MAX_FUNC_LINES = 50


def _violations(folders: tuple[str, ...]) -> list[str]:
    """Rule violations in every .py file under the given repo folders."""
    out = []
    for folder in folders:
        for path in sorted((ROOT / folder).rglob("*.py")):
            source = path.read_text()
            rel = path.relative_to(ROOT)
            if len(source.splitlines()) > MAX_FILE_LINES:
                out.append(f"{rel}: more than {MAX_FILE_LINES} lines")
            for node in ast.walk(ast.parse(source)):
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    continue
                if not isinstance(node, ast.ClassDef) and node.end_lineno - node.lineno + 1 > MAX_FUNC_LINES:
                    out.append(f"{rel}:{node.lineno} {node.name} longer than {MAX_FUNC_LINES} lines")
                if node.name.startswith("_"):
                    continue
                if ast.get_docstring(node) is None:
                    out.append(f"{rel}:{node.lineno} {node.name} has no docstring")
                if not isinstance(node, ast.ClassDef) and node.returns is None:
                    out.append(f"{rel}:{node.lineno} {node.name} has no return type hint")
    return out


def test_library_code_follows_rules() -> None:
    """src/, gee/ and scripts/ respect the length, docstring and return-hint rules."""
    assert _violations(("src", "gee", "scripts")) == []


def test_test_code_follows_rules() -> None:
    """tests/ respects the same rules (every public function needs a docstring and a return hint)."""
    assert _violations(("tests",)) == []
