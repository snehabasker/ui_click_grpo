#!/usr/bin/env bash
# Print torch / triton / CUDA capability for every venv, to decide where the Triton kernel can run.
for v in "$HOME"/venvs/*; do
  echo "== $v"
  "$v/bin/python" - <<'EOF' 2>&1 | tail -4
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("capability", torch.cuda.get_device_capability(), torch.cuda.get_device_name())
try:
    import triton
    print("triton", triton.__version__)
except Exception as e:
    print("triton import failed:", e)
EOF
done
