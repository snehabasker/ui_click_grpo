#!/bin/bash
# One-shot status of the router benchmark (read-only).
L="$HOME/work/logs"
echo "--- processes"; pgrep -af "run_router_bench|vllm serve|prefix-router" | cut -c1-140
echo "--- GPU"; nvidia-smi --query-gpu=memory.used,temperature.gpu,power.draw --format=csv,noheader
echo "--- driver log"; tail -n 15 "$L/router_bench_driver.log"
echo "--- newest logs"; ls -t "$L" | head -6
for f in $(ls -t "$L"/router-bench-replica*.log 2>/dev/null | head -2); do
  echo "--- $(basename "$f")"; grep -E "Available KV|GPU KV cache size|Maximum concurrency|Error|ERROR|Traceback" "$f" | cut -c1-200 | tail -5
done
