#!/usr/bin/env bash
# What Phases 4 and 5 can build on: free disk, vLLM's KV-transfer connectors, whether TensorRT-LLM wheels are reachable.
df -h "$HOME" | tail -1
V="$HOME/venvs/vllm/lib/python3.12/site-packages/vllm"
echo "--- vllm kv connectors:"; ls "$V/distributed/kv_transfer/kv_connector/v1/" 2>/dev/null | head -30
echo "--- registered names:"; grep -n "register_connector" -A3 "$V/distributed/kv_transfer/kv_connector/factory.py" 2>/dev/null | grep -o '"[A-Za-z0-9_]*Connector[A-Za-z0-9_]*"' | sort -u
echo "--- examples shipped?"; find "$V/.." -iname "*disagg*" 2>/dev/null | head -5
echo "--- trtllm on pip index?"
"$HOME/.local/bin/uv" pip install --dry-run --python "$HOME/venvs/vllm/bin/python" --extra-index-url https://pypi.nvidia.com tensorrt_llm 2>&1 | tail -6
echo "--- python versions:"; ls "$HOME/.local/share/uv/python" 2>/dev/null | head; free -g | head -2
