#!/usr/bin/env bash
# Phase 5: TensorRT-LLM's PyTorch backend JIT-compiles FlashInfer kernels with nvcc 13.2, which rejects the user-space gcc 16 ("gcc versions
# later than 15 are not supported"). Install gcc 14 from conda-forge into ~/gcc14 (no sudo); trtllm_env.sh puts it first on PATH.
[ -x "$HOME/gcc14/bin/gcc" ] || "$HOME/.local/bin/micromamba" create -y -q -p "$HOME/gcc14" -c conda-forge "gcc=14" "gxx=14" 2>&1 | tail -3
"$HOME/gcc14/bin/gcc" --version | head -1
