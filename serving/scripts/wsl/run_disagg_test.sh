#!/usr/bin/env bash
# Diagnostic: split setup only, one round, c=4, prefill instance WITHOUT chunked prefill (hypothesis: the ExampleConnector stores a
# request's KV at its first scheduled step, so a prompt split over several steps is stored truncated). Tag explore-disagg: not a result.
W="/mnt/c/Users/Sneha Basker/ui-click-rl/serving/scripts/wsl"
sed 's/\r$//' "$W/sync_clone.sh" > /tmp/sync_clone.sh; bash /tmp/sync_clone.sh || exit 1
mkdir -p ~/work/logs; cd ~/work/ui_click_grpo || exit 1
"$HOME/venvs/bench/bin/python" -u serving/scripts/run_disagg_bench.py --mode bench --setups split --rounds 1 --concurrency 4 \
  --tag-prefix explore-disagg --producer-extra "--no-enable-chunked-prefill --max-num-batched-tokens=4096" 2>&1 | grep -E "^\[|wrote|Error" | cut -c1-250
