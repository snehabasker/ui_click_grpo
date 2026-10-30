#!/usr/bin/env bash
# torchvision must match torch's CUDA major version (10_trtllm_torch_cu130.sh left the PyPI cu128 build in place); then retry the import.
source /tmp/trtllm_env.sh
"$HOME/.local/bin/uv" pip install --python "$TRT_PY" --index-url https://download.pytorch.org/whl/cu130 --extra-index-url https://pypi.org/simple \
  --index-strategy unsafe-best-match "torchvision==0.24.1+cu130" 2>&1 | tail -4 | cut -c1-200
bash /tmp/trtllm_import.sh
