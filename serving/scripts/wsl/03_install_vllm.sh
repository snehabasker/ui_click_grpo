#!/bin/bash
# Install vLLM into its own venv. --torch-backend=auto lets uv pick the torch build matching the driver (CUDA 13.1 here).
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
LOG="$HOME/work/install_vllm.log"
{
  date
  uv pip install --python "$HOME/venvs/vllm/bin/python" vllm --torch-backend=auto
  echo "exit code: $?"
  "$HOME/venvs/vllm/bin/python" - <<'PYEOF'
import importlib.metadata as md
for p in ("vllm", "torch", "transformers", "triton", "flashinfer-python", "xformers", "flash-attn"):
    try:
        print(p, md.version(p))
    except md.PackageNotFoundError:
        print(p, "-")
import torch
print("torch.cuda.is_available:", torch.cuda.is_available(), "| torch cuda:", torch.version.cuda)
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0), "| capability:", torch.cuda.get_device_capability(0), "| arch list:", torch.cuda.get_arch_list())
PYEOF
  date
} 2>&1 | tee "$LOG"
