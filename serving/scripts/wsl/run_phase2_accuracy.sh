#!/bin/bash
# Click accuracy on both fixed sets (synthetic n=300, ScreenSpot n=400) through each router policy, after the router change.
# Driver output: ~/work/logs/router_phase2_accuracy.log
cd "$HOME/work/ui_click_grpo" || exit 1
{
  echo "=== accuracy through the router $(date +%T)  commit $(git rev-parse --short HEAD)"
  "$HOME/venvs/bench/bin/python" -u serving/scripts/run_router_bench.py --mode accuracy
  echo "=== all done $(date +%T)"
} > "$HOME/work/logs/router_phase2_accuracy.log" 2>&1
