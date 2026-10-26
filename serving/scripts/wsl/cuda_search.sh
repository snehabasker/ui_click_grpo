#!/bin/bash
# What CUDA toolkit packages does conda-forge offer? (read-only; nothing is installed)
MM="$HOME/.local/bin/micromamba"
for pkg in cuda-nvcc cuda-cudart-dev cuda-version; do
  echo "== $pkg (newest 6 versions):"
  "$MM" search -c conda-forge "$pkg" --json 2>/dev/null | "$HOME/venvs/bench/bin/python" -c "
import json, sys
d = json.load(sys.stdin)
vs = sorted({r['version'] for r in d.get('result', {}).get('pkgs', [])}, key=lambda v: [int(x) if x.isdigit() else 0 for x in v.replace('-', '.').split('.')])
print(', '.join(vs[-6:]))"
done
echo "== torch in the SGLang venv was built with CUDA: $("$HOME/venvs/sglang/bin/python" -c 'import torch; print(torch.version.cuda)')"
echo "== deep_gemm cuda_helpers (what it looks for):"
sed -n 1,30p "$HOME/venvs/sglang/lib/python3.12/site-packages/deep_gemm/cuda_helpers.py" | cut -c1-160
