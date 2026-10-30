#!/usr/bin/env bash
# Phase 5: TensorRT-LLM into its own venv (never into the vllm venv: it pins different torch / xgrammar). Log: ~/work/logs/install_trtllm.log
set -u
mkdir -p "$HOME/work/logs"; cd "$HOME" || exit 1
command -v gcc >/dev/null || export PATH="$HOME/toolchain/bin:$PATH"
[ -d "$HOME/venvs/trtllm" ] || "$HOME/.local/bin/uv" venv "$HOME/venvs/trtllm" --python 3.12
"$HOME/.local/bin/uv" pip install --python "$HOME/venvs/trtllm/bin/python" --extra-index-url https://pypi.nvidia.com tensorrt_llm 2>&1 | tee "$HOME/work/logs/install_trtllm.log" | tail -25
echo "INSTALL_EXIT=${PIPESTATUS[0]}"
