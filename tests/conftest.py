"""Pytest session guard: fail early with an actionable message when the environment is incomplete.

Without an HTTP client library, ``fastapi.testclient`` fails during collection with a message that
does not say which Python is running. That usually means pytest was started from the wrong
interpreter (e.g. the system Python instead of the project's virtual environment).
"""
import importlib.util
import sys

import pytest

if not (importlib.util.find_spec("httpx2") or importlib.util.find_spec("httpx")):
    pytest.exit(
        "\n\nThe API tests need an HTTP client package (httpx2 or httpx), but neither is installed for\n"
        f"the Python that is running pytest:\n    {sys.executable}\n\n"
        "This usually means pytest was started outside the project's virtual environment. Fix:\n"
        "    python3 -m venv .venv\n"
        "    source .venv/bin/activate\n"
        "    pip install -r requirements.txt\n"
        "    python -m pytest          # 'python -m' guarantees the venv's interpreter is used\n",
        returncode=4,
    )
