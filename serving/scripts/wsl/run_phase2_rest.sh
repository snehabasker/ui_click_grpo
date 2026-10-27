#!/bin/bash
# Phase 2, remaining runs: (1) the policy comparison again with the order reversed (round-robin first) to separate policy from run
# order / GPU thermal state; (2) the kill-a-replica failure test. Driver output goes to ~/work/logs/router_phase2_rest.log.
cd "$HOME/work/ui_click_grpo" || exit 1
PY="$HOME/venvs/bench/bin/python"
LOG="$HOME/work/logs/router_phase2_rest.log"
{
  echo "=== reversed-order bench $(date +%T)"
  "$PY" -u serving/scripts/run_router_bench.py --mode bench --policies round-robin prefix --skip-accuracy
  echo "=== failure test $(date +%T)"
  "$PY" -u serving/scripts/run_router_bench.py --mode failure
  echo "=== all done $(date +%T)"
} > "$LOG" 2>&1
