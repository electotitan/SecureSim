"""Static guard: all project source parses under the Python 3.10 grammar.

Limitation: this only detects 3.11+-only SYNTAX (e.g. `except*`). It cannot detect use of
3.11+-only standard-library APIs; those require actually running on Python 3.10.
"""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_all_python_files_parse_as_python_310():
    files = [p for d in ("backend", "tests") for p in (ROOT / d).rglob("*.py")]
    assert files
    for path in files:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path), feature_version=(3, 10))
