#!/bin/bash
# why did the two SGLang follow-up launches fail? (read-only)
L="$HOME/work/logs"
for label in sglang-flashinfer-bs32-prefix-on sglang-triton-bs32-prefix-on; do
  f=$(ls -t "$L"/${label}-2*.log | head -1)
  echo "=================== $label ($(basename "$f"), $(wc -l < "$f") lines)"
  echo "--- command line:"; grep -m1 -E '^\+ exec' "$f" | cut -c1-600 | sed -E "s#$HOME#~#g"
  echo "--- last 14 non-blank lines:"
  grep -v -E '^\s*$|Warning|warn' "$f" | tail -14 | cut -c1-260 | sed -E "s#$HOME#~#g"
done
echo "=================== the flag names SGLang accepts for CUDA graph batch size:"
grep -n -E -- "--cuda-graph-max-bs|--cuda-graph-bs" "$HOME/work/sglang_help.txt" | head -8 | cut -c1-200
