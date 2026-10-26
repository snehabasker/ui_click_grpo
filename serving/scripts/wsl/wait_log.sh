#!/bin/bash
# wait_log.sh <logfile> <max_seconds> [lines_to_show]
# Waits until the log shows readiness or a failure, then prints the tail. Exit 0 = ready, 1 = failure, 2 = timeout.
LOG="$1"; MAX="${2:-300}"; N="${3:-45}"
READY='Application startup complete|The server is fired up|Uvicorn running on'
FAIL='Traceback|RuntimeError|Error:|error:|Killed|CUDA out of memory|No such file|Engine core initialization failed'
start=$(date +%s); rc=2
while [ $(( $(date +%s) - start )) -lt "$MAX" ]; do
  if [ -f "$LOG" ] && grep -q -E "$READY" "$LOG"; then rc=0; break; fi
  if [ -f "$LOG" ] && grep -q -E "$FAIL" "$LOG"; then rc=1; sleep 3; break; fi
  sleep 2
done
echo "== result: $([ $rc = 0 ] && echo READY || ([ $rc = 1 ] && echo FAILURE || echo TIMEOUT)) after $(( $(date +%s) - start ))s; log lines: $(wc -l < "$LOG" 2>/dev/null)"
grep -v -E '^\+|^$' "$LOG" | tail -n "$N" | cut -c1-240
exit $rc
