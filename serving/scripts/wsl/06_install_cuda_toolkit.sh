#!/bin/bash
# A CUDA toolkit (nvcc + headers) WITHOUT sudo, for SGLang. Its deep_gemm module asserts that a CUDA home exists at import time
# on sm_120 GPUs, and its JIT paths need nvcc. Version 13.2 matches the CUDA the SGLang torch wheel was built with.
# Kept in its own prefix and only put on PATH by serve_sglang.sh, so the vLLM launch is unaffected.
set -euo pipefail
MM="$HOME/.local/bin/micromamba"
if [ ! -x "$HOME/cuda132/bin/nvcc" ]; then
  "$MM" create -y -q -p "$HOME/cuda132" -c conda-forge "cuda-nvcc=13.2.*" "cuda-cudart-dev=13.2.*" "cuda-cccl=13.2.*" "cuda-version=13.2" ninja
fi
export CUDA_HOME="$HOME/cuda132"
export PATH="$CUDA_HOME/bin:$PATH"
# The conda layout keeps libraries in lib/, but SGLang's JIT links with -L$CUDA_HOME/lib64 (found when its first kernel build
# failed with "cannot find -lcudart"). A symlink inside this private prefix fixes it.
[ -e "$CUDA_HOME/lib64" ] || ln -s lib "$CUDA_HOME/lib64"
nvcc --version | tail -2
echo "CUDA_HOME=$CUDA_HOME"
ls "$CUDA_HOME" | head -12 | tr '\n' ' '; echo
ls "$CUDA_HOME/targets/x86_64-linux/include" 2>/dev/null | head -4 | tr '\n' ' '; echo
echo "host compilers in that prefix: $(ls "$CUDA_HOME/bin" | grep -E 'g\+\+|gcc' | head -4 | tr '\n' ' ')"
# does nvcc compile for this GPU (sm_120)? tiny kernel, no run needed
printf '__global__ void k(){}\nint main(){k<<<1,1>>>();return 0;}\n' > /tmp/k.cu
nvcc -arch=sm_120 /tmp/k.cu -o /tmp/k 2>&1 | head -5 && echo "nvcc sm_120 compile: OK" || echo "nvcc sm_120 compile: FAILED"
"$HOME/venvs/sglang/bin/python" -c "import deep_gemm; print('deep_gemm import: OK')" 2>&1 | tail -2
echo "CUDA_TOOLKIT_DONE"
