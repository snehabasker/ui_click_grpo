#!/bin/bash
# grep_envs.sh <pattern>: vLLM environment variables whose name matches a pattern, with their defaults/comments
E="$HOME/venvs/vllm/lib/python3.12/site-packages/vllm/envs.py"
grep -n -i -E "$1" "$E" | head -20 | cut -c1-220
