#!/bin/bash
# The full memory-accounting lines of two replicas (read-only). usage: replica_mem_full.sh <log0> <log1>
for f in "$1" "$2"; do
  echo "=================== $(basename "$f")"
  grep -E "Free memory on device|Actual usage|Available KV|memory profiling|non-torch" "$f" | sed -E 's/^\(EngineCore pid=[0-9]+\) //' | cut -c1-900
done
