#!/bin/bash
# Print the explicit-kv-cache-memory branch of vLLM's profiling (read-only).
V=/home/sneha_basker/.cache/uv/archive-v0/9rOG_flsRczlCTnF/vllm
sed -n 640,700p "$V/v1/worker/gpu_worker.py"
