#!/usr/bin/env bash
# run_kernel_accuracy.sh [kernel_accuracy.py args...]: rule-3 accuracy check of the INT8 RMSNorm numerics, vllm venv, from the clone.
cd "$HOME/work/ui_click_grpo" || exit 1
command -v gcc >/dev/null || export PATH="$HOME/toolchain/bin:$PATH"
export CC="${CC:-gcc}"
exec "$HOME/venvs/vllm/bin/python" -u serving/scripts/kernel_accuracy.py "$@"
