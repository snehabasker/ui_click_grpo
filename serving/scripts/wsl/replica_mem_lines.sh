#!/bin/bash
# Every memory-accounting line a vLLM replica logged, for two replicas side by side (read-only). usage: replica_mem_lines.sh <log0> <log1>
for f in "$1" "$2"; do
  echo "=================== $(basename "$f")"
  grep -i -E "memory|GiB|MiB|budget|utilization" "$f" | grep -v -E "Warning|warn|Route:|Allocating 0.0" | sed -E 's/^\(EngineCore pid=[0-9]+\) //' | cut -c1-260 | head -16
done
echo "=================== GPU now: $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader)"
