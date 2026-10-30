#!/bin/bash
# Copy raw results from the Linux clone into the Windows checkout (exploratory files excluded; existing files never overwritten).
SRC="$HOME/work/ui_click_grpo/serving/results/raw"
DST="${WIN_REPO:-/mnt/c/Users/Sneha Basker/ui-click-rl}/serving/results/raw"
mkdir -p "$DST"
n=0
for f in "$SRC"/loadtest_*.json "$SRC"/engine_*.json "$SRC"/accuracy_*.json "$SRC"/router_bench_*.json "$SRC"/router_failure_*.json "$SRC"/kernel_bench_*.json "$SRC"/kernel_accuracy_*.json "$SRC"/disagg*_bench_2*.json "$SRC"/disagg_accuracy_*.json "$SRC"/trtllm_bench_2*.json "$SRC"/trtllm_accuracy_2*.json "$SRC"/prompt_tokens_check_*.json; do
  [ -e "$f" ] || continue
  case "$(basename "$f")" in *explore*) continue;; esac
  if [ ! -e "$DST/$(basename "$f")" ]; then cp "$f" "$DST/"; n=$((n+1)); fi
done
echo "copied $n new raw files"
echo "raw files now in the Windows repo: loadtest=$(ls "$DST" | grep -c '^loadtest_') engine=$(ls "$DST" | grep -c '^engine_') accuracy=$(ls "$DST" | grep -c '^accuracy_')"
echo "--- any home-directory path left in the copied files? $(grep -l "$HOME" "$DST"/loadtest_*.json "$DST"/engine_*.json "$DST"/accuracy_*.json "$DST"/router_*.json 2>/dev/null | wc -l) files"
