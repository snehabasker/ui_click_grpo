"""Per-run GPU power, temperature and throughput for every run recorded with the extended monitor, in time order.
Used to check whether throughput tracks the GPU's power state. Read-only; prints the table (no statistics: 3 repeats per group is too few
for a correlation to mean anything)."""
import glob
import json
import statistics
from pathlib import Path

root = Path(__file__).resolve().parents[1] / "results" / "raw"
rows = []
for f in sorted(glob.glob(str(root / "loadtest_*.json"))):
    d = json.load(open(f))
    g = d.get("gpu") or {}
    if g.get("power_w_median") is None:
        continue
    rows.append((Path(f).name.split("_")[-1][:-5], d["tag"], d["concurrency"], d["repeat"], d["summary"]["req_per_s"],
                 g["power_w_median"], g["power_w_max"], g["temp_c_max"], g["temp_c_median"]))
rows.sort()
print(f"{'time':15s} {'run':34s} {'c':>3s} {'r':>2s} {'req/s':>6s} {'pwr_med':>7s} {'pwr_max':>7s} {'T_max':>5s} {'T_med':>5s}")
for r in rows:
    print(f"{r[0]:15s} {r[1]:34s} {r[2]:3d} {r[3]:2d} {r[4]:6.2f} {r[5]:7.1f} {r[6]:7.1f} {r[7]:5.0f} {r[8]:5.0f}")
