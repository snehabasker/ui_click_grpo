#!/bin/bash
# Progress of the follow-up run + what else is using the GPU/CPU right now (read-only).
LOG="$HOME/work/logs/followup1.log"
RAW="$HOME/work/ui_click_grpo/serving/results/raw"
date
echo "--- runner log (last 12 lines):"
grep -v -E "Warning|warn|^\s*$" "$LOG" | tail -n 12 | cut -c1-210
echo "--- follow-up raw files so far:"
ls -1 "$RAW" | grep -E "flashinfer|triton-bs32|prefix-on-rerun" | sed -E 's/_[0-9]{8}-[0-9]{6}\.json$//' | sort | uniq -c
echo "--- GPU now (util, mem, temp, power, sm clock):"
nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw,clocks.sm --format=csv,noheader
echo "--- GPU processes visible from Linux:"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>&1 | head -6
echo "--- Linux CPU load (1/5/15 min) and cores: $(cut -d' ' -f1-3 /proc/loadavg) / $(nproc)"
