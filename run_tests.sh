#!/usr/bin/env bash
# Run the test suite in a clean environment.
#
# Why: if your shell has other Python tooling on PYTHONPATH (for example ROS 2:  source /opt/ros/<distro>/setup.bash),
# pytest will auto-discover THAT tooling's pytest plugins even inside this project's virtual environment, and can crash
# before running any test (for example "ModuleNotFoundError: No module named 'lark'" from ROS's launch_testing).
#
# This script ignores PYTHONPATH and disables automatic loading of third-party pytest plugins. The project's tests need
# neither. Extra arguments are passed to pytest:   ./run_tests.sh -q -k rsa
set -euo pipefail
cd "$(dirname "$0")"
PY=python3
if [ -x ".venv/bin/python" ]; then PY=".venv/bin/python"; fi
exec env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "$PY" -m pytest "$@"
