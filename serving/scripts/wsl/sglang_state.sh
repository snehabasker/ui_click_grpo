#!/bin/bash
# Is the SGLang server alive, and what are the last log lines? Where does libcudart live in the CUDA prefix?
LOG="${1:-$HOME/work/logs/sglang_explore2.log}"
echo "sglang processes alive: $(pgrep -f 'sglang.launch_server' | wc -l)"
echo "log lines: $(wc -l < "$LOG")"
echo "--- last 12 log lines:"
tail -n 12 "$LOG" | cut -c1-260
echo "--- libcudart in the CUDA prefix:"
find "$HOME/cuda132" -name 'libcudart*' 2>/dev/null | head -6
echo "--- lib64 present? $(ls -d "$HOME/cuda132/lib64" 2>&1 | head -1)"
