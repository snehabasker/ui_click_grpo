#!/bin/bash
# Progress of the Phase 1 run (read-only): what the runner printed and which raw files exist so far.
LOG="$HOME/work/logs/phase1.log"
RAW="$HOME/work/ui_click_grpo/serving/results/raw"
date
echo "--- runner log (last 14 lines):"
grep -v -E "Warning|warn|^\s*$" "$LOG" | tail -n 14 | cut -c1-220
echo "--- raw files so far (excluding explore):"
ls -1 "$RAW" | grep -v explore | sed -E 's/_[0-9]{8}-[0-9]{6}\.json$//' | sort | uniq -c
echo "--- GPU now: $(nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw,clocks.sm --format=csv,noheader)"
