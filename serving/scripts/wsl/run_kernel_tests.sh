#!/usr/bin/env bash
# run_kernel_tests.sh [pytest args...]: the Triton kernel tests, in the vllm venv (torch + triton on the GPU), from the Linux clone.
# Set SRC=win to run from the Windows checkout instead (uncommitted work while iterating).
if [ "${SRC:-clone}" = "win" ]; then cd "/mnt/c/Users/Sneha Basker/ui-click-rl" || exit 1; else cd "$HOME/work/ui_click_grpo" || exit 1; fi
PY="$HOME/venvs/vllm/bin/python"
# the venv was made by uv and has no pip: `uv pip install --python $PY pytest` once if this fails
"$PY" -c "import pytest" 2>/dev/null || { echo "pytest missing in the vllm venv: $HOME/.local/bin/uv pip install --python $PY pytest"; exit 1; }
exec "$PY" -m pytest serving/tests/test_rmsnorm_int8.py -x -q -p no:cacheprovider "$@"
