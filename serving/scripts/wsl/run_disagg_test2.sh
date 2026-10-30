#!/usr/bin/env bash
# Diagnostic 2: split only, one round, c=16, prefill instance with a larger per-step budget (--max-num-batched-tokens=16384, no chunked
# prefill). Hypothesis: at c=16 the multimodal encoder budget defers a prompt's last image to a later step, and the ExampleConnector stores
# KV at the first step only. Tag explore-disagg2: not a result.
cd ~/work/ui_click_grpo || exit 1
"$HOME/venvs/bench/bin/python" -u serving/scripts/run_disagg_bench.py --mode bench --setups split --rounds 1 --concurrency 16 \
  --tag-prefix explore-disagg2 --producer-extra "--no-enable-chunked-prefill --max-num-batched-tokens=16384" 2>&1 | grep -E "^\[|wrote|Error" | cut -c1-250
