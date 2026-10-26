#!/bin/bash
# why did the runner reject --cuda-graph-max-bs in the clone? (read-only)
C="$HOME/work/ui_click_grpo"
echo "clone HEAD: $(git -C "$C" rev-parse --short HEAD)"
echo "--- args defined in the clone's runner:"
grep -n -E 'add_argument\("--(attn-backend|cuda-graph-max-bs|extra|sets)"' "$C/serving/scripts/run_engine_bench.py" | cut -c1-150
echo "--- first 12 lines of the follow-up log:"
grep -v -E "Warning|warn" "$HOME/work/logs/followup1.log" | head -12 | cut -c1-200
echo "--- is that commit's runner the same as Windows'? (hash of file in git at HEAD)"
git -C "$C" log -1 --format='%h %s' -- serving/scripts/run_engine_bench.py
