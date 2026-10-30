#!/usr/bin/env bash
# tensorrt_llm pins numpy 1.26.x; the newest pyarrow needs numpy >= 2. Use an older pyarrow that still supports numpy 1.x, then retry the import.
source /tmp/trtllm_env.sh
"$HOME/.local/bin/uv" pip list --python "$TRT_PY" 2>/dev/null | grep -i -E "^(pyarrow|numpy|datasets) "
"$HOME/.local/bin/uv" pip install --python "$TRT_PY" "pyarrow<21" 2>&1 | tail -3 | cut -c1-200
bash /tmp/trtllm_import.sh
