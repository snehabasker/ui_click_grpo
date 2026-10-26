#!/bin/bash
# Phase 1 follow-up, second attempt: steps B and A only. Step C (vLLM rerun) already ran in the first attempt, and both SGLang
# steps there failed immediately because SGLang 0.5.22 renamed --cuda-graph-max-bs to --cuda-graph-max-bs-decode.
#   B  SGLang, FlashInfer attention + decode CUDA graphs up to batch 32 (best effort), both accuracy sets, c=1/4/16/32
#   A  SGLang, Triton attention (as in Phase 1) + graphs up to batch 32: isolates the CUDA-graph effect (c=16, 32)
cd "$HOME/work/ui_click_grpo" || exit 1
PY="$HOME/venvs/bench/bin/python"
BENCH=serving/scripts/run_engine_bench.py
echo "follow-up 1b started $(date) at commit $(git rev-parse --short HEAD)"
"$PY" -u $BENCH --engine sglang --label sglang-flashinfer-bs32-prefix-on --prefix-caching 1 \
      --attn-backend flashinfer --cuda-graph-max-bs 32
"$PY" -u $BENCH --engine sglang --label sglang-triton-bs32-prefix-on --prefix-caching 1 \
      --cuda-graph-max-bs 32 --concurrency 16 32 --sets synth300
echo "FOLLOWUP_1B_DONE $(date)"
