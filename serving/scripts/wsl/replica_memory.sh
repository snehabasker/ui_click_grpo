#!/bin/bash
# Memory facts of a running vLLM replica from its log and nvidia-smi (read-only). usage: replica_memory.sh <logfile>
LOG="$1"
grep -o -E "Model loading took [0-9.]+ GiB[^,]*|Available KV cache memory: [0-9.]+ GiB|GPU KV cache size: [0-9,]+ tokens|Maximum concurrency for [0-9,]+ tokens per request: [0-9.]+x|Graph capturing finished in [0-9]+ secs, took [0-9.]+ GiB|quantization=[a-z0-9_]+" "$LOG" | awk '!s[$0]++'
echo "GPU memory used now: $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader)"
echo "free for a second replica: $(nvidia-smi --query-gpu=memory.free --format=csv,noheader)"
