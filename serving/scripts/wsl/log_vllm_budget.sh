#!/bin/bash
# vLLM's per-step token budget as it logged it (read-only)
V=$(ls -t "$HOME/work/logs"/vllm-prefix-on-*.log | head -1)
echo "log: $(basename "$V")"
grep -i -o -E ".{0,60}(batched|max_num_seqs|chunked).{0,90}" "$V" | head -8
echo "--- vLLM default for max_num_batched_tokens in this version (source):"
grep -n -E "DEFAULT_MAX_NUM_BATCHED_TOKENS|default_max_num_batched_tokens|max_num_batched_tokens\s*[:=]" "$HOME/venvs/vllm/lib/python3.12/site-packages/vllm/engine/arg_utils.py" | head -6 | cut -c1-200
grep -n -A6 "_set_default_args" "$HOME/venvs/vllm/lib/python3.12/site-packages/vllm/engine/arg_utils.py" | grep -i -E "batched|usage_context|device" | head -6 | cut -c1-200
