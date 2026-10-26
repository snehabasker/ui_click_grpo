#!/bin/bash
# Targeted scheduler facts (read-only).
L="$HOME/work/logs"
V=$(ls -t "$L"/vllm-prefix-on-*.log | head -1)
S=$(ls -t "$L"/sglang-prefix-on-*.log | head -1)
SOFF=$(ls -t "$L"/sglang-prefix-off-*.log | head -1)
echo "=== vLLM: token budget per scheduler step"
grep -o -E "max_num_batched_tokens[^,)]{0,20}" "$V" | sort -u | head -4
grep -o -E "Chunked prefill is enabled with max_num_batched_tokens=[0-9]+" "$V" | head -1
echo "=== SGLang (cache ON): decode batches with >= 10 running requests"
grep -E "Decode batch, #running-req: (1[0-9]|2[0-9]|3[0-9])," "$S" | sed -E 's/^\[[0-9-]+ [0-9:]+\] //' | awk 'NR%25==1' | cut -c1-190 | head -5
echo "count of decode batches with >=10 running (ON):  $(grep -c -E 'Decode batch, #running-req: (1[0-9]|2[0-9]|3[0-9]),' "$S")"
echo "count of decode batches with >=10 running (OFF): $(grep -c -E 'Decode batch, #running-req: (1[0-9]|2[0-9]|3[0-9]),' "$SOFF")"
echo "=== SGLang (cache ON): max token usage seen: $(grep -o -E 'token usage: [0-9.]+' "$S" | awk '{print $3}' | sort -n | tail -1)    (OFF): $(grep -o -E 'token usage: [0-9.]+' "$SOFF" | awk '{print $3}' | sort -n | tail -1)"
echo "=== SGLang (cache ON): the retract/evict line:"
grep -i -E "retract|evict|preempt" "$S" | cut -c1-200 | head -2
echo "=== SGLang (cache ON): prefill batches during load with large cached-token counts (cache really hit)"
grep -E "Prefill batch" "$S" | grep -E "#cached-token: ([1-9][0-9]{3,})" | sed -E 's/^\[[0-9-]+ [0-9:]+\] //' | awk 'NR%80==1' | cut -c1-170 | head -4
echo "count of prefill batches with >=1000 cached tokens: $(grep -E 'Prefill batch' "$S" | grep -c -E '#cached-token: [1-9][0-9]{3,}')   total prefill batches: $(grep -c 'Prefill batch' "$S")"
