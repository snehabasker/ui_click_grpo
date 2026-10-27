#!/bin/bash
# Print vLLM's KV-cache budget formula and request_memory (read-only).
V=$(dirname "$(find "$HOME" -path "*vllm/v1/worker/gpu_worker.py" 2>/dev/null | head -1)")/../..
F="$V/v1/worker/gpu_worker.py"
sed -n 705,765p "$F"
echo "================ request_memory"
M=$(grep -rl "def request_memory" "$V" 2>/dev/null | head -1)
echo "file: $M"
grep -n "def request_memory" -A40 "$M"
