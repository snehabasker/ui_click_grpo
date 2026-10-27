#!/bin/bash
# Emit one line per milestone/failure of run_router_bench.py (reads its driver log). Exits when the driver finishes or fails.
LOG="$HOME/work/logs/router_bench_driver.log"
seen=0
while true; do
  n=$(wc -l < "$LOG" 2>/dev/null || echo 0)
  if [ "$n" -gt "$seen" ]; then
    tail -n +$((seen + 1)) "$LOG" | grep -E --line-buffered "ready|Traceback|Error|error|wrote|FAILED|accuracy|policy|conc|done" | cut -c1-220
    seen=$n
  fi
  if grep -q -E "Traceback|^wrote router_bench" "$LOG" 2>/dev/null; then break; fi
  if ! pgrep -f run_router_bench.py >/dev/null; then echo "driver process gone"; break; fi
  sleep 10
done
