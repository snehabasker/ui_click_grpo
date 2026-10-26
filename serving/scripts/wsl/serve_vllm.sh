#!/bin/bash
# Serve the merged checkpoint with vLLM (OpenAI-compatible API on 127.0.0.1).
#   env: PORT (8000)  PREFIX_CACHING (1|0)  GPU_UTIL (0.85)  MAX_SEQS (32)  MAX_LEN (4096)  EXTRA (additional flags)
# Settings that must match the SGLang launch for a fair comparison: model, dtype, context length, max concurrent sequences,
# greedy decoding (set per request), image pixel budget, prefix cache on/off.
M="${MODEL_DIR:-$HOME/work/ui_click_grpo/serving/models/merged-clean}"
export PATH="$HOME/venvs/vllm/bin:$PATH"
# Triton JIT-compiles a small C helper on first use and needs a C compiler. Use the system one if present, else the
# user-space toolchain from 04_install_toolchain.sh.
command -v gcc >/dev/null || export PATH="$HOME/toolchain/bin:$PATH"
export CC="${CC:-gcc}"
# vLLM's FlashInfer top-k/top-p sampler JIT-compiles with nvcc, which is not installed here (the attention backend is
# FlashAttention 2 and needs no nvcc). All benchmark requests are greedy (temperature 0), so the sampler choice does not change
# any output; this is recorded in the run metadata as a deviation from vLLM's defaults.
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
# dev endpoints (/reset_prefix_cache, /reset_mm_cache) are used to make repeats cold; they are local-only (127.0.0.1)
export VLLM_SERVER_DEV_MODE=1
PC=--enable-prefix-caching
[ "${PREFIX_CACHING:-1}" = "0" ] && PC=--no-enable-prefix-caching
set -x
exec vllm serve "$M" \
  --served-model-name merged-clean --dtype bfloat16 \
  --max-model-len "${MAX_LEN:-4096}" --max-num-seqs "${MAX_SEQS:-32}" \
  --gpu-memory-utilization "${GPU_UTIL:-0.85}" \
  --limit-mm-per-prompt '{"image":4}' --mm-processor-kwargs '{"max_pixels":1003520}' \
  $PC --enable-prompt-tokens-details \
  --host 127.0.0.1 --port "${PORT:-8000}" ${EXTRA:-}
