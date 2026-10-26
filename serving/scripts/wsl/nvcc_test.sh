#!/bin/bash
# Does nvcc compile and RUN a kernel on this GPU, using the host compiler bundled in the CUDA prefix?
export CUDA_HOME="$HOME/cuda132"
export PATH="$CUDA_HOME/bin:$PATH"
CCBIN="$CUDA_HOME/bin/x86_64-conda-linux-gnu-g++"
echo "host compiler: $("$CCBIN" --version | head -1)"
cat > /tmp/k.cu <<'EOF'
#include <cstdio>
__global__ void k(int *x){ *x = 42; }
int main(){ int *d, h = 0; cudaMalloc(&d, 4); k<<<1,1>>>(d); cudaMemcpy(&h, d, 4, cudaMemcpyDeviceToHost);
  printf("kernel result: %d (arch sm_120)\n", h); return h == 42 ? 0 : 1; }
EOF
nvcc -ccbin "$CCBIN" -arch=sm_120 /tmp/k.cu -o /tmp/k -L"$CUDA_HOME/lib" -L"$CUDA_HOME/targets/x86_64-linux/lib" -L/usr/lib/wsl/lib 2>&1 | head -6
echo "compile exit: ${PIPESTATUS[0]}"
LD_LIBRARY_PATH="/usr/lib/wsl/lib:$CUDA_HOME/lib:$CUDA_HOME/targets/x86_64-linux/lib" /tmp/k 2>&1 | head -3
