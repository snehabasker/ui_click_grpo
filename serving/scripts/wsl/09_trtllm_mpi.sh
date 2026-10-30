#!/usr/bin/env bash
# Phase 5: TensorRT-LLM needs libmpi at import time; without sudo (no apt) take OpenMPI from conda-forge into ~/mpi and put it on LD_LIBRARY_PATH.
[ -d "$HOME/mpi/lib" ] || "$HOME/.local/bin/micromamba" create -y -q -p "$HOME/mpi" -c conda-forge openmpi 2>&1 | tail -3
ls "$HOME/mpi/lib" | grep -E "^libmpi\.so" | head -3
export LD_LIBRARY_PATH="$HOME/mpi/lib:${LD_LIBRARY_PATH:-}"
command -v gcc >/dev/null || export PATH="$HOME/toolchain/bin:$PATH"
timeout 300 "$HOME/venvs/trtllm/bin/python" -c "import tensorrt_llm; print('tensorrt_llm', tensorrt_llm.__version__)" 2>&1 | tail -8 | cut -c1-300
