#!/bin/bash
# Step B again, queued behind step A: FlashInfer's JIT needs GCC <= 15 (see serve_sglang.sh). Waits for the running follow-up to
# finish (so it never changes the code under a running measurement), syncs the clone with verification, then runs B with a long
# startup timeout because FlashInfer compiles its kernels on first use.
LOG="$HOME/work/logs/followup1c.log"
until grep -q FOLLOWUP_1B_DONE "$LOG" 2>/dev/null; do sleep 20; done
bash "/mnt/c/Users/Sneha Basker/ui-click-rl/serving/scripts/wsl/sync_clone.sh" || { echo "SYNC_FAILED, not running"; exit 1; }
cd "$HOME/work/ui_click_grpo" || exit 1
PY="$HOME/venvs/bench/bin/python"
echo "step B retry started $(date) at commit $(git rev-parse --short HEAD)"
"$PY" -u serving/scripts/run_engine_bench.py --engine sglang --label sglang-flashinfer-bs32-prefix-on --prefix-caching 1 \
      --attn-backend flashinfer --cuda-graph-max-bs 32 --startup-timeout 1800
echo "FOLLOWUP_1D_DONE $(date)"
