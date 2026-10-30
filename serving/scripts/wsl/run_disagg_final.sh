#!/usr/bin/env bash
# Phase 4 timed benchmark with the workaround for the ExampleConnector / chunked-prefill problem on the PREFILL instance (see NOTES.md).
W="/mnt/c/Users/Sneha Basker/ui-click-rl/serving/scripts/wsl"
sed 's/\r$//' "$W/sync_clone.sh" > /tmp/sync_clone.sh; bash /tmp/sync_clone.sh || exit 1
mkdir -p ~/work/logs; cd ~/work/ui_click_grpo || exit 1
"$HOME/venvs/bench/bin/python" -u serving/scripts/run_disagg_bench.py --mode bench \
  --producer-extra "--no-enable-chunked-prefill --max-num-batched-tokens=4096" 2>&1 | tee ~/work/logs/disagg_final.log | grep -E "^\[|wrote|Error" | cut -c1-250
