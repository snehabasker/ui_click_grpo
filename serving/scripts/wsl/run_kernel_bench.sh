#!/usr/bin/env bash
# run_kernel_bench.sh [bench_kernel.py args...]: the kernel benchmark in the vllm venv from the Linux clone (committed code only,
# so the raw result's repo_commit is the code that ran). SRC=win runs from the Windows checkout for smoke tests.
if [ "${SRC:-clone}" = "win" ]; then cd "/mnt/c/Users/Sneha Basker/ui-click-rl" || exit 1; else cd "$HOME/work/ui_click_grpo" || exit 1; fi
exec "$HOME/venvs/vllm/bin/python" -u serving/scripts/bench_kernel.py "$@"
