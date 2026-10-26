#!/bin/bash
# Phase 1: vLLM vs SGLang on the same merged checkpoint, with prefix caching on and off.
#   run from WSL after sync_clone.sh (so the code that runs is the committed code recorded in every result)
# prefix-cache-on runs get both accuracy checks; the off runs get the synthetic one (they change scheduling/caching, not the weights).
cd "$HOME/work/ui_click_grpo" || exit 1
PY="$HOME/venvs/bench/bin/python"
BENCH=serving/scripts/run_engine_bench.py
echo "phase 1 started $(date) at commit $(git rev-parse --short HEAD)"
"$PY" -u $BENCH --engine vllm   --label vllm-prefix-on   --prefix-caching 1
"$PY" -u $BENCH --engine sglang --label sglang-prefix-on --prefix-caching 1
"$PY" -u $BENCH --engine vllm   --label vllm-prefix-off  --prefix-caching 0 --sets synth300
"$PY" -u $BENCH --engine sglang --label sglang-prefix-off --prefix-caching 0 --sets synth300
"$PY" serving/scripts/summarize.py --out serving/results/tables/phase1
echo "PHASE1_RUNS_DONE $(date)"
