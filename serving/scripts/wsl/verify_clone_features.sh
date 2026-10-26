#!/bin/bash
# Does the clone really contain the code the follow-up needs? (read-only)
cd "$HOME/work/ui_click_grpo" || exit 1
PY="$HOME/venvs/bench/bin/python"
echo "HEAD: $(git rev-parse --short HEAD)"
echo "runner accepts --cuda-graph-max-bs: $("$PY" serving/scripts/run_engine_bench.py --help 2>&1 | grep -c -- '--cuda-graph-max-bs')   --attn-backend: $("$PY" serving/scripts/run_engine_bench.py --help 2>&1 | grep -c -- '--attn-backend')"
echo "GPU monitor samples clock+temperature: $(grep -c 'temperature.gpu,clocks.sm' serving/servebench/client.py)"
echo "serve_sglang.sh reads ATTN_BACKEND / CUDA_GRAPH_MAX_BS: $(grep -c 'ATTN_BACKEND' serving/scripts/wsl/serve_sglang.sh) / $(grep -c 'CUDA_GRAPH_MAX_BS' serving/scripts/wsl/serve_sglang.sh)"
"$PY" -c "
import sys; sys.path.insert(0, 'serving')
from servebench.client import parse_gpu_line
print('parse_gpu_line works:', parse_gpu_line('7000, 99, 71, 1500, 30.2'))"
