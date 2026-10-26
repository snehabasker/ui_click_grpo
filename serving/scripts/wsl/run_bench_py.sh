#!/bin/bash
# run_bench_py.sh <script.py> [args...]: run a repo script with the client venv from the Linux clone
cd "$HOME/work/ui_click_grpo" || exit 1
exec "$HOME/venvs/bench/bin/python" "$@"
