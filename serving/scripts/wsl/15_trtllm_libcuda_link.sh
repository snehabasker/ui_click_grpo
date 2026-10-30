#!/usr/bin/env bash
# Phase 5: FlashInfer's JIT links with -lcuda; the user-space CUDA toolkit has no stub libcuda.so. WSL exposes the real driver library as
# /usr/lib/wsl/lib/libcuda.so.1, so expose it under the name the linker looks for (trtllm_env.sh adds ~/cuda_stubs to LIBRARY_PATH).
mkdir -p "$HOME/cuda_stubs" && ln -sf /usr/lib/wsl/lib/libcuda.so.1 "$HOME/cuda_stubs/libcuda.so" && ls -l "$HOME/cuda_stubs/libcuda.so"
