#!/bin/bash
# Phase 1 follow-up: is the vLLM-vs-SGLang gap a configuration effect?
#   B  SGLang, FlashInfer attention + CUDA graphs up to batch 32 (best effort), full run, both accuracy sets
#   A  SGLang, Triton attention (as in Phase 1) + CUDA graphs up to batch 32: isolates the CUDA-graph effect (c=16, 32)
#   C  vLLM prefix-on rerun at c=16 with the extended GPU monitor: does Phase 1 reproduce, and what were clocks/temperature?
# Prefix caching is ON in all three (the Phase 1 result that matters). Run after sync_clone.sh.
cd "$HOME/work/ui_click_grpo" || exit 1
PY="$HOME/venvs/bench/bin/python"
BENCH=serving/scripts/run_engine_bench.py
echo "follow-up started $(date) at commit $(git rev-parse --short HEAD)"
"$PY" -u $BENCH --engine sglang --label sglang-flashinfer-bs32-prefix-on --prefix-caching 1 \
      --attn-backend flashinfer --cuda-graph-max-bs 32
"$PY" -u $BENCH --engine sglang --label sglang-triton-bs32-prefix-on --prefix-caching 1 \
      --cuda-graph-max-bs 32 --concurrency 16 32 --sets synth300
"$PY" -u $BENCH --engine vllm --label vllm-prefix-on-rerun --prefix-caching 1 \
      --concurrency 16 --skip-accuracy
echo "FOLLOWUP_RUNS_DONE $(date)"
