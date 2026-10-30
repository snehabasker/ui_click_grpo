#!/usr/bin/env bash
# Dry run into a throwaway target: which tensorrt_llm / torch / CUDA does the NVIDIA index resolve to? (nothing is installed)
"$HOME/.local/bin/uv" pip install --dry-run --python 3.12 --extra-index-url https://pypi.nvidia.com tensorrt_llm 2>&1 | grep -E "tensorrt|torch|nvidia-cu|cuda|Resolved|Would|error" | head -20
