#!/bin/bash
# Serve the merged checkpoint with SGLang (OpenAI-compatible API on 127.0.0.1).
#   env: PORT (8000)  PREFIX_CACHING (1|0)  GPU_UTIL (0.85)  MAX_SEQS (32)  MAX_LEN (4096)  EXTRA (additional flags)
# Matched with serve_vllm.sh: model, dtype, context length, max concurrent sequences, image pixel budget, greedy requests,
# prefix cache on/off (RadixAttention vs automatic prefix caching), cached-token reporting.
# NOT identical, and recorded as such: attention/sampling kernels (each engine's nvcc-free choice), memory-fraction semantics
# (SGLang's covers weights + KV pool; vLLM's the whole budget incl. CUDA-graph memory), so the resulting KV capacity differs.
M="${MODEL_DIR:-$HOME/work/ui_click_grpo/serving/models/merged-clean}"
command -v gcc >/dev/null || export PATH="$HOME/toolchain/bin:$PATH"
export CC="${CC:-gcc}"
# SGLang's deep_gemm asserts a CUDA home exists at import on sm_120, and its JIT paths need nvcc: use the user-space toolkit from
# 06_install_cuda_toolkit.sh (13.2, the CUDA the SGLang torch wheel was built with) and its bundled host compiler.
export CUDA_HOME="$HOME/cuda132"
export PATH="$CUDA_HOME/bin:$PATH"
export NVCC_PREPEND_FLAGS="-ccbin $CUDA_HOME/bin/x86_64-conda-linux-gnu-g++"
# FlashInfer's JIT calls nvcc with its own -ccbin (plain `c++`, which here is GCC 16) and CUDA 13.2 rejects GCC > 15
# ("unsupported GNU version"), overriding the flag above. Point every compiler variable at the GCC 15.3 that ships with the toolkit.
export CC="$CUDA_HOME/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$CUDA_HOME/bin/x86_64-conda-linux-gnu-g++"
export CUDAHOSTCXX="$CXX"
# FlashInfer's kernel build links with -lcuda. Under WSL2 the driver library lives in /usr/lib/wsl/lib, which the conda linker does
# not search ("cannot find -lcuda"); LIBRARY_PATH is honoured by GCC at link time.
export LIBRARY_PATH="/usr/lib/wsl/lib:$CUDA_HOME/lib64${LIBRARY_PATH:+:$LIBRARY_PATH}"
PC=""
[ "${PREFIX_CACHING:-1}" = "0" ] && PC="--disable-radix-cache"
# Phase 1 follow-up knobs (defaults reproduce the Phase 1 runs exactly): attention backend, and the largest decode batch that gets a
# CUDA graph (unset = SGLang's own choice, which on this 8 GB GPU came out as 8).
ATTN="${ATTN_BACKEND:-triton}"
CG=""
# SGLang 0.5.22 renamed this flag: plain --cuda-graph-max-bs is now ambiguous (-decode / -prefill) and the server refuses to start.
[ -n "${CUDA_GRAPH_MAX_BS:-}" ] && CG="--cuda-graph-max-bs-decode $CUDA_GRAPH_MAX_BS"
set -x
exec "$HOME/venvs/sglang/bin/python" -m sglang.launch_server \
  --model-path "$M" --served-model-name merged-clean --dtype bfloat16 \
  --context-length "${MAX_LEN:-4096}" --max-running-requests "${MAX_SEQS:-32}" \
  --mem-fraction-static "${GPU_UTIL:-0.85}" \
  --attention-backend "$ATTN" --sampling-backend pytorch --mm-attention-backend sdpa $CG \
  --mm-process-config '{"image":{"max_pixels":1003520}}' \
  --enable-cache-report --enable-metrics $PC \
  --host 127.0.0.1 --port "${PORT:-8000}" ${EXTRA:-}
