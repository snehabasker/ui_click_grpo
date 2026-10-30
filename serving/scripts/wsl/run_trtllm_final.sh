#!/usr/bin/env bash
# Phase 5: sync the clone, copy the one-image trace's images (git-ignored, rendered on Windows), run run_trtllm_bench.py in MODE (accuracy|bench).
W="/mnt/c/Users/Sneha Basker/ui-click-rl"
sed 's/\r$//' "$W/serving/scripts/wsl/sync_clone.sh" > /tmp/sync_clone.sh; bash /tmp/sync_clone.sh | tail -1 || exit 1
D=~/work/ui_click_grpo; mkdir -p "$D/serving/traces/single"
[ -d "$D/serving/traces/single/images" ] || cp -r "$W/serving/traces/single/images" "$D/serving/traces/single/images"
sed 's/\r$//' "$W/serving/scripts/wsl/trtllm_env.sh" > /tmp/trtllm_env.sh
cd "$D" || exit 1; mkdir -p ~/work/logs
"$HOME/venvs/bench/bin/python" -u serving/scripts/run_trtllm_bench.py --mode "${MODE:-accuracy}" "$@" 2>&1 | tee ~/work/logs/trtllm_${MODE:-accuracy}.log | grep -E "acc=|ok=|wrote|Error|Traceback|error" | cut -c1-250
