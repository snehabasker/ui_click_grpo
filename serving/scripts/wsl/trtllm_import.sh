#!/usr/bin/env bash
source "$(dirname "$0")/trtllm_env.sh" 2>/dev/null || source /tmp/trtllm_env.sh
timeout 300 "$TRT_PY" -c "import tensorrt_llm; print('tensorrt_llm', tensorrt_llm.__version__)" 2>&1 | tail -6 | cut -c1-300
