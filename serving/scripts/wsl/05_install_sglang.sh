#!/bin/bash
# Install SGLang into its own venv (separate from vLLM so their torch/transformers pins cannot conflict).
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
command -v gcc >/dev/null || export PATH="$HOME/toolchain/bin:$PATH"
LOG="$HOME/work/install_sglang.log"
{
  date
  uv pip install --python "$HOME/venvs/sglang/bin/python" "sglang" --torch-backend=auto
  echo "exit code: $?"
  "$HOME/venvs/sglang/bin/python" - <<'PYEOF'
import importlib.metadata as md
for p in ("sglang", "sgl-kernel", "sglang-kernel", "torch", "transformers", "triton", "flashinfer-python", "flashinfer-cubin", "flash-attn", "xgrammar"):
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
