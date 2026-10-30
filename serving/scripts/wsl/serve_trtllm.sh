#!/bin/bash
# Serve the merged checkpoint with TensorRT-LLM (PyTorch backend, OpenAI-compatible API on 127.0.0.1).
#   env: PORT (8003)  REUSE (true; false = no prefix reuse)  MAX_SEQS (32)  MAX_LEN (4096)  FRAC (0.80, fraction of the memory that is free after loading the weights, used for KV)
# Limits match the Phase 1 vLLM runs (bf16, max_num_seqs 32, max_model_len 4096) so the comparison is like for like; block reuse
# (prefix caching) is on, as in the "prefix-on" vLLM runs.
source /tmp/trtllm_env.sh 2>/dev/null || source "$(dirname "$0")/trtllm_env.sh"
M="${MODEL_DIR:-$HOME/work/ui_click_grpo/serving/models/merged-clean}"
CFG="${TRTLLM_CFG:-/tmp/trtllm_cfg_$$.yaml}"
cat > "$CFG" <<YAML
kv_cache_config:
  enable_block_reuse: ${REUSE:-true}
  free_gpu_memory_fraction: ${FRAC:-0.80}
YAML
set -x
exec "$HOME/venvs/trtllm/bin/trtllm-serve" serve "$M" --backend pytorch --host 127.0.0.1 --port "${PORT:-8003}" \
  --max_batch_size "${MAX_SEQS:-32}" --max_seq_len "${MAX_LEN:-4096}" --max_num_tokens "${MAX_NUM_TOKENS:-4096}" \
  --config "$CFG" ${EXTRA:-}
