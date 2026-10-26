#!/bin/bash
# Scheduler settings and batch statistics the engines logged during the Phase 1 runs (read-only).
L="$HOME/work/logs"
V=$(ls -t "$L"/vllm-prefix-on-*.log | head -1)
S=$(ls -t "$L"/sglang-prefix-on-*.log | head -1)
echo "=== vLLM scheduler settings (log: $(basename "$V")):"
grep -o -E "max_num_batched_tokens=[0-9]+|max_num_seqs=[0-9]+|enable_chunked_prefill=[A-Za-z]+|Chunked prefill is enabled[^,]*|long_prefill_token_threshold=[0-9]+" "$V" | sort -u | head -8
echo "=== SGLang scheduler settings:"
grep -o -E "chunked_prefill_size=[0-9]+|max_prefill_tokens=[0-9]+|max_running_requests=[0-9]+|schedule_policy='?[a-z-]+'?|max_total_num_tokens=[0-9]+" "$S" | sort -u | head -8
echo "=== SGLang decode-batch lines under load (sample of 6 from the middle of the run):"
grep -E "Decode batch" "$S" | awk 'NR%40==0' | sed -E 's/^\[[0-9-]+ [0-9:]+\] //' | cut -c1-210 | head -6
echo "=== SGLang: retractions / evictions mentioned?"
grep -i -c -E "retract|evict|preempt" "$S"
echo "=== vLLM: preemptions mentioned?"
grep -i -c -E "preempt" "$V"
echo "=== SGLang log: prefill batch lines (queue length, token usage) - 6 samples:"
grep -E "Prefill batch" "$S" | awk 'NR%60==0' | sed -E 's/^\[[0-9-]+ [0-9:]+\] //' | cut -c1-230 | head -6
