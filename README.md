# Interactive Mathematical Simulator for Secure Digital Communication

> Work in progress: the full README (architecture, API, usage) is a Checkpoint 16 deliverable.

## Python version note (known deviation from the specification)

The specification requires **Python 3.11 or newer**. During development the project was
also run on **Python 3.10.12**. Decision: all code is kept **compatible with Python 3.10**
(no 3.11-only syntax or standard-library features such as `tomllib`, `ExceptionGroup`,
`typing.Self`, or `datetime.UTC`). The project therefore runs on 3.10 and 3.11+, but
3.10 does not meet the specification's stated minimum. Use 3.11+ where possible.

Verification limits: the author's CI-style checks ran on Python 3.12 (automated tests) and the
user ran the Checkpoint 1 tests on Python 3.10.12. A static test (`tests/test_py310_compat.py`)
parses all source files with the 3.10 grammar; it cannot detect 3.11-only library calls.

## Security disclaimer

This is an educational simulator. AES, RSA and every other algorithm here are hand-written
for learning and must **not** be used to protect real sensitive data.

## Quick start

    python -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    uvicorn backend.main:app --reload --port 8000
    pytest
