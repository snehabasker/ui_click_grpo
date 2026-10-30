#!/usr/bin/env bash
# Phase 5: tensorrt_llm 1.2.1 links CUDA 13 libraries, but the default PyPI torch 2.9.1 is built for CUDA 12.8 (import failed with
# "libcublasLt.so.13: cannot open shared object file"). Put torch 2.9.1 built for CUDA 13.0 from the PyTorch index into the same venv.
export LD_LIBRARY_PATH="$HOME/mpi/lib:${LD_LIBRARY_PATH:-}"
command -v gcc >/dev/null || export PATH="$HOME/toolchain/bin:$PATH"
P="$HOME/venvs/trtllm/bin/python"
"$HOME/.local/bin/uv" pip install --python "$P" --index-url https://download.pytorch.org/whl/cu130 --extra-index-url https://pypi.org/simple \
  --index-strategy unsafe-best-match "torch==2.9.1+cu130" torchvision 2>&1 | tail -12 | cut -c1-200
echo "--- torch:"; "$P" -c "import torch;print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_capability())" 2>&1 | tail -2
echo "--- import:"; timeout 300 "$P" -c "import tensorrt_llm; print('tensorrt_llm', tensorrt_llm.__version__)" 2>&1 | tail -6 | cut -c1-300
