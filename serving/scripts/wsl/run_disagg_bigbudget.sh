#!/usr/bin/env bash
# Phase 4, split setup at c=16 with a 16384-token per-step budget on the prefill instance, 3 rounds (the configuration that stops the
# ExampleConnector from storing truncated KV, see NOTES.md). Tag disagg-bigbudget.
W="/mnt/c/Users/Sneha Basker/ui-click-rl/serving/scripts/wsl"
sed 's/\r$//' "$W/sync_clone.sh" > /tmp/sync_clone.sh; bash /tmp/sync_clone.sh || exit 1
cd ~/work/ui_click_grpo || exit 1
"$HOME/venvs/bench/bin/python" -u serving/scripts/run_disagg_bench.py --mode bench --setups split --rounds 3 --concurrency 16 \
  --tag-prefix disagg-bigbudget --producer-extra "--no-enable-chunked-prefill --max-num-batched-tokens=16384" 2>&1 | tee ~/work/logs/disagg_bigbudget.log | grep -E "^\[|wrote|Error" | cut -c1-250
