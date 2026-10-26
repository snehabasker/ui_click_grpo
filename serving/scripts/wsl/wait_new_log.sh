#!/bin/bash
# wait_new_log.sh <label> <max_seconds> [lines]: wait for a server log of this run label created AFTER this script started, then
# follow it with wait_log.sh. Avoids the mistake of reading the previous attempt's log.
LABEL="$1"; MAX="${2:-900}"; N="${3:-24}"
L="$HOME/work/logs"; start=$(date +%s); marker=$(mktemp)
touch "$marker"
for i in $(seq 1 120); do
  f=$(find "$L" -name "${LABEL}-2*.log" -newer "$marker" 2>/dev/null | head -1)
  [ -n "$f" ] && break
  sleep 2
done
[ -z "$f" ] && { echo "no new log for $LABEL after 240 s (did the sync fail?)"; tail -n 5 "$L/followup1e.log"; exit 3; }
echo "following $(basename "$f")"
exec bash "$(dirname "$0")/wait_log.sh" "$f" "$MAX" "$N"
