#!/bin/bash
# Stop any inference server this project started (inside WSL only) and report whether the GPU memory came back.
for pat in "vllm serve" "sglang.launch_server" "sglang serve" "vllm.entrypoints"; do
  pkill -TERM -f "$pat" 2>/dev/null && echo "sent SIGTERM to: $pat"
done
for i in $(seq 1 20); do
  n=$(pgrep -f -c "vllm serve|sglang|vllm.entrypoints|EngineCore" 2>/dev/null || true)
  [ "${n:-0}" = "0" ] && break
  sleep 1
done
pkill -KILL -f "EngineCore|vllm serve|sglang.launch_server" 2>/dev/null && echo "forced remaining processes"
sleep 2
echo "gpu memory now: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
