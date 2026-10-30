#!/usr/bin/env bash
# Phase 5 local patch to the INSTALLED tensorrt_llm 1.2.1 (a throwaway venv, not the repo): its Qwen2-VL vision tower hard-codes
# attn_implementation = 'flash_attention_2' (modeling_qwen2vl.py line 388, comment: "to avoid OOM issue") and the flash_attn package is not
# installable here (no wheel for torch 2.9.1 + CUDA 13; a source build was not attempted). Switch that one line to 'sdpa'. Idempotent; the
# original is kept as modeling_qwen2vl.py.orig. Any result obtained with it is labelled as such in serving/README.md.
F="$HOME/venvs/trtllm/lib/python3.12/site-packages/tensorrt_llm/_torch/models/modeling_qwen2vl.py"
[ -f "$F.orig" ] || cp "$F" "$F.orig"
sed -i "s/self.config._attn_implementation = 'flash_attention_2'/self.config._attn_implementation = 'sdpa'/" "$F"
diff "$F.orig" "$F" && echo "no change needed" || echo "patched"
