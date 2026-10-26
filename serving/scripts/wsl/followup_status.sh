#!/bin/bash
# Status of the newest follow-up attempt (read-only). Looks at the runner log and the NEWEST server log by modification time.
L="$HOME/work/logs"
date
echo "--- runner log (followup1c.log):"
grep -v -E "Warning|warn|^\s*$" "$L/followup1c.log" 2>/dev/null | tail -n 8 | cut -c1-200
echo "--- server logs of the follow-up, newest first (name, size, age in seconds):"
now=$(date +%s)
for f in $(ls -t "$L"/sglang-flashinfer-bs32-prefix-on-2*.log "$L"/sglang-triton-bs32-prefix-on-2*.log 2>/dev/null | head -4); do
  echo "$(basename "$f")  $(wc -c < "$f") bytes  age $(( now - $(stat -c %Y "$f") ))s"
done
NEW=$(ls -t "$L"/sglang-flashinfer-bs32-prefix-on-2*.log 2>/dev/null | head -1)
echo "--- newest FlashInfer log: tail:"
grep -v -E '^\s*$|Warning|warn' "$NEW" | tail -n 10 | cut -c1-230 | sed -E "s#$HOME#~#g"
echo "--- its command line:"; grep -m1 -o -E -- "--cuda-graph-max-bs[a-z-]* [0-9]+|--attention-backend [a-z]+" "$NEW" | tr '\n' ' '; echo
echo "--- sglang alive: $(pgrep -f -c 'sglang.launch_server')"
