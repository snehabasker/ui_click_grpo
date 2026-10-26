#!/bin/bash
# GPU state (temperature, clock, power) the extended monitor recorded during the vLLM rerun, next to its throughput (read-only)
cd "$HOME/work/ui_click_grpo" || exit 1
"$HOME/venvs/bench/bin/python" - <<'PYEOF'
import glob, json
print(f"{'run':34s} {'req/s':>6s} {'ttft_p50':>9s} {'temp_max':>8s} {'sm_clock_busy_med':>18s} {'sm_clock_busy_min':>18s} {'power_med_W':>11s} {'mem_max':>8s}")
for f in sorted(glob.glob("serving/results/raw/loadtest_vllm-prefix-on-rerun_*.json")):
    d = json.load(open(f)); g = d["gpu"]; s = d["summary"]
    print(f"{f.split('/')[-1][9:50]:34s} {s['req_per_s']:6.2f} {s['ttft']['p50']:9.2f} {g.get('temp_c_max'):8} "
          f"{g.get('sm_clock_mhz_busy_median'):18} {g.get('sm_clock_mhz_busy_min'):18} {g.get('power_w_median'):11} {g.get('mem_used_mib_max'):8}")
print("\nGPU max SM clock seen while running:", max(json.load(open(f))["gpu"]["sm_clock_mhz_max"] for f in glob.glob("serving/results/raw/loadtest_vllm-prefix-on-rerun_*.json")), "MHz")
PYEOF
echo; echo "Phase 1 (same cell, vllm-prefix-on, c=16): req/s 3.48 / 3.55 / ~3.52, ttft p50 2.11-2.22 s (no clock/temperature recorded then)"
echo "GPU clocks by P-state right now: $(nvidia-smi --query-gpu=clocks.max.sm,clocks.sm,power.limit --format=csv,noheader)"
