#!/bin/bash
# which flags does this SGLang version accept? (read-only)
PY="$HOME/venvs/sglang/bin/python"
command -v gcc >/dev/null || export PATH="$HOME/toolchain/bin:$PATH"
"$PY" -m sglang.launch_server --help 2>&1 | tr -d '\r' > "$HOME/work/sglang_help.txt"
echo "help lines: $(wc -l < "$HOME/work/sglang_help.txt")"
for f in model-path served-model-name dtype context-length max-running-requests mem-fraction-static attention-backend sampling-backend mm-attention-backend disable-radix-cache enable-cache-report enable-metrics mm-process-config chunked-prefill-size disable-cuda-graph cuda-graph-max-bs max-total-tokens mm-max-concurrent-calls grammar-backend skip-server-warmup; do
  printf "%-28s %s\n" "--$f" "$(grep -c -- "--$f" "$HOME/work/sglang_help.txt")"
done
echo "--- attention backend choices:"
grep -A3 -- "--attention-backend" "$HOME/work/sglang_help.txt" | head -6 | cut -c1-230
echo "--- sampling backend:"
grep -A2 -- "--sampling-backend" "$HOME/work/sglang_help.txt" | head -4 | cut -c1-200
echo "--- multimodal processing / mm options:"
grep -n -i -- "--mm-" "$HOME/work/sglang_help.txt" | head -8 | cut -c1-200
