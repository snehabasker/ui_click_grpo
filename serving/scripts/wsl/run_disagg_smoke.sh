#!/usr/bin/env bash
# sync the clone to HEAD, then run the disaggregation smoke test with the client venv (log: ~/work/logs/disagg_smoke.log)
W="/mnt/c/Users/Sneha Basker/ui-click-rl/serving/scripts/wsl"
sed 's/\r$//' "$W/sync_clone.sh" > /tmp/sync_clone.sh; bash /tmp/sync_clone.sh || exit 1
mkdir -p ~/work/logs; cd ~/work/ui_click_grpo || exit 1
"$HOME/venvs/bench/bin/python" -u serving/scripts/run_disagg_bench.py --mode "${MODE:-smoke}" "$@" 2>&1 | tee ~/work/logs/disagg_${MODE:-smoke}.log | tail -30
