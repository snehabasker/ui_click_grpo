#!/bin/bash
# Where vLLM computes the per-process KV-cache budget (read-only). Finds gpu_worker.py wherever the vLLM install lives.
F=$(find "$HOME" -path "*vllm/v1/worker/gpu_worker.py" 2>/dev/null | head -1)
echo "file: $F"
[ -z "$F" ] && exit 1
grep -n -E "requested_memory|non_torch|non_kv_cache|available_kv_cache|init_snapshot|free_memory|baseline|Free memory on device" "$F" | head -70
