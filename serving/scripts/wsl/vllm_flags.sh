#!/bin/bash
# which flags does this vLLM version accept? (read-only)
V="$HOME/venvs/vllm/bin/vllm"
"$V" --version 2>&1 | tail -1
"$V" serve --help=all 2>&1 | tr -d '\r' > "$HOME/work/vllm_serve_help.txt"
for f in gpu-memory-utilization max-model-len max-num-seqs limit-mm-per-prompt mm-processor-kwargs enable-prefix-caching no-enable-prefix-caching served-model-name enforce-eager dtype attention-backend max-num-batched-tokens enable-prompt-tokens-details kv-cache-dtype; do
  printf "%-32s %s\n" "--$f" "$(grep -c -- "--$f" "$HOME/work/vllm_serve_help.txt")"
done
echo "--- help lines for the multimodal limit and prompt-token details:"
grep -A3 -- "--limit-mm-per-prompt" "$HOME/work/vllm_serve_help.txt" | head -8
grep -A2 -- "--enable-prompt-tokens-details" "$HOME/work/vllm_serve_help.txt" | head -4
