#!/bin/bash
# log_nvcc_trace.sh <logfile>: the call chain that ended in the nvcc error (library frames only, deduplicated)
LOG="$1"
N=$(grep -n "Could not find nvcc" "$LOG" | head -1 | cut -d: -f1)
echo "error at line $N"
sed -n "$((N - 60)),$((N + 1))p" "$LOG" | sed -E 's/^\(EngineCore pid=[0-9]+\) ERROR [0-9-]+ [0-9:]+ \[[a-z_.]+:[0-9]+\] //' \
  | grep -E 'File ".*(flashinfer|vllm/(v1|model_executor|utils)|triton|torch/utils/cpp_extension)' | sed -E "s#File \"$HOME/venvs/vllm/lib/python3.12/site-packages/##" | cut -c1-200 | tail -14
echo "--- attention backend / sampler lines in the log:"
grep -i -E "attention backend|Using .* backend|flashinfer|flash.attn|FLASH_ATTN|sampler" "$LOG" | sed -E 's/^\(EngineCore pid=[0-9]+\) //' | awk '!seen[$0]++' | head -10 | cut -c1-220
