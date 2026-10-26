#!/bin/bash
# log_root_cause.sh <logfile>: the first error in a server log, with context, plus the environment lines around startup
LOG="$1"
echo "== total lines: $(wc -l < "$LOG")"
echo "== first 'Error' / 'error' / 'Exception' lines (with line numbers):"
grep -n -i -E "error|exception|failed|cannot|not found|unsupported|no kernel|invalid" "$LOG" | grep -v -E "^\s*[0-9]+:\s*(File|\^)" | head -25 | cut -c1-260
echo
FIRST=$(grep -n -i -E "error|exception|failed" "$LOG" | head -1 | cut -d: -f1)
if [ -n "$FIRST" ]; then
  START=$(( FIRST > 12 ? FIRST - 12 : 1 ))
  echo "== context around the first error (lines $START..$((FIRST + 25))):"
  sed -n "${START},$((FIRST + 25))p" "$LOG" | cut -c1-260
fi
