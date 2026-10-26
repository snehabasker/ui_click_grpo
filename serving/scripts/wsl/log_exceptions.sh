#!/bin/bash
# log_exceptions.sh <logfile>: the exception lines (type: message) in a server log, plus what the engine did before it failed
LOG="$1"
echo "== lines: $(wc -l < "$LOG")"
echo "== exception messages (first 12, deduplicated):"
sed -E 's/^\(EngineCore pid=[0-9]+\) ERROR [0-9-]+ [0-9:]+ \[[a-z_.]+:[0-9]+\] //' "$LOG" \
  | grep -E "^[A-Za-z_.]*(Error|Exception|Failure)[A-Za-z_]*:|^[a-z_.]+\.[A-Za-z]*(Error|Exception)" | awk '!seen[$0]++' | head -12 | cut -c1-400
echo
echo "== INFO/WARNING lines just before the first failure:"
N=$(grep -n -E "EngineCore failed to start|Traceback" "$LOG" | head -1 | cut -d: -f1)
S=$(( N > 22 ? N - 22 : 1 ))
sed -n "${S},$((N - 1))p" "$LOG" | grep -v -E "^\s*$" | cut -c1-260
