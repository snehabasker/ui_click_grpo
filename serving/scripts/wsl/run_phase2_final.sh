#!/bin/bash
# Phase 2 final runs with the router that retries a replica that dies before its first body byte:
#   (1) the kill-a-replica test again, with the load outlasting the restart so the router can be seen routing to the restarted replica;
#   (2) the policy comparison with the policy order alternating on every repeat (run order was a confound in the first two runs).
# Driver output: ~/work/logs/router_phase2_final.log
export PATH="$HOME/.cargo/bin:$HOME/toolchain/bin:$PATH" CC=gcc
cd "$HOME/work/ui_click_grpo" || exit 1
PY="$HOME/venvs/bench/bin/python"
LOG="$HOME/work/logs/router_phase2_final.log"
{
  echo "=== build $(date +%T)  commit $(git rev-parse --short HEAD)"
  ROUTER_SRC="$PWD/serving/router" bash serving/scripts/wsl/cargo_router.sh build --release 2>&1 | tail -2
  echo "=== failure test $(date +%T)"
  "$PY" -u serving/scripts/run_router_bench.py --mode failure --kill-after 25 --restart-after 2
  echo "=== interleaved bench $(date +%T)"
  "$PY" -u serving/scripts/run_router_bench.py --mode bench --interleave
  echo "=== all done $(date +%T)"
} > "$LOG" 2>&1
