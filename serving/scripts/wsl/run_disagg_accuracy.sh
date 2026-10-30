#!/usr/bin/env bash
# Rule 3 for Phase 4: click accuracy through the split path (prefill+decode via ExampleConnector, 16384-token step budget on the prefill
# instance) and through the colocated instance, on both fixed sets.
W="/mnt/c/Users/Sneha Basker/ui-click-rl/serving/scripts/wsl"
sed 's/\r$//' "$W/sync_clone.sh" > /tmp/sync_clone.sh; bash /tmp/sync_clone.sh || exit 1
cd ~/work/ui_click_grpo || exit 1
"$HOME/venvs/bench/bin/python" -u serving/scripts/run_disagg_bench.py --mode accuracy \
  --producer-extra "--no-enable-chunked-prefill --max-num-batched-tokens=16384" 2>&1 | tee ~/work/logs/disagg_accuracy.log | grep -E "acc=|wrote|Error" | cut -c1-250
