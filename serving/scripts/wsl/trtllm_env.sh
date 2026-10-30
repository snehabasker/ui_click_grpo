# source me: environment for the TensorRT-LLM venv (no sudo, so libmpi and the CUDA 13 runtime libraries are found through LD_LIBRARY_PATH)
V="$HOME/venvs/trtllm/lib/python3.12/site-packages"
L="$HOME/mpi/lib"
for d in "$V"/nvidia/*/lib "$V"/tensorrt_libs "$V"/torch/lib; do [ -d "$d" ] && L="$L:$d"; done
export LD_LIBRARY_PATH="$L:${LD_LIBRARY_PATH:-}"
command -v gcc >/dev/null || export PATH="$HOME/toolchain/bin:$PATH"
export CC="${CC:-gcc}"
TRT_PY="$HOME/venvs/trtllm/bin/python"
export CUDA_HOME="${CUDA_HOME:-$HOME/cuda132}"
export PATH="$CUDA_HOME/bin:$PATH"
