#!/bin/bash
# help text of the SGLang flags that must be matched against vLLM (read-only)
F="$HOME/work/sglang_help.txt"
for flag in "--mm-process-config" "--mem-fraction-static" "--max-running-requests" "--enable-cache-report" "--context-length"; do
  echo "=== $flag"
  grep -n -A5 -- "^  $flag" "$F" | head -8 | cut -c1-240
done
