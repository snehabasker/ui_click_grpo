#!/bin/bash
# Stop a running follow-up (script, runner, servers) inside WSL and discard the partial results of the aborted run.
pkill -TERM -f run_followup1.sh 2>/dev/null && echo "stopped run_followup1.sh"
pkill -TERM -f run_engine_bench.py 2>/dev/null && echo "stopped run_engine_bench.py"
pkill -TERM -f "serving/scripts/loadtest.py" 2>/dev/null && echo "stopped loadtest.py"
sleep 3
bash "$(dirname "$0")/stop_servers.sh"
RAW="$HOME/work/ui_click_grpo/serving/results/raw"
n=$(ls "$RAW" 2>/dev/null | grep -c -E "rerun|flashinfer|triton-bs32")
rm -f "$RAW"/*rerun* "$RAW"/*flashinfer* "$RAW"/*triton-bs32* 2>/dev/null
echo "discarded $n partial result files from the aborted follow-up"
echo "remaining processes: $(pgrep -f -c 'run_engine_bench|loadtest.py|vllm serve|sglang.launch_server' 2>/dev/null || echo 0)"
