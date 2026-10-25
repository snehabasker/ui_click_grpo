"""How different are the merged model's clicks from the adapter model's, beyond the accuracy comparison?

Reads a merge_verify_*.json (which stores every raw prediction) and measures the coordinate shift on items where both models
returned a valid point. Writes results/raw/merge_diff_<ts>.json.   python serving/scripts/merge_diff.py <merge_verify.json>
"""
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "serving"))

from servebench import envinfo
from servebench.data import load_manifest
from servebench.stats import percentile
from uiclick.reward import in_box, parse_click

src = Path(sys.argv[1])
d = json.loads(src.read_text())
out = {}
for s in ("synth300", "screenspot400"):
    _, items = load_manifest(ROOT / "serving" / "data" / s)
    A, M = d["predictions"][f"adapter/{s}"], d["predictions"][f"merged/{s}"]
    shifts, flips_to_miss, flips_to_hit, both = [], 0, 0, 0
    for it, a, m in zip(items, A, M):
        pa, pm = parse_click(a), parse_click(m)
        if pa is None or pm is None:
            continue
        both += 1
        shifts.append(max(abs(pa[0] - pm[0]), abs(pa[1] - pm[1])))     # worst axis, in 0-1000 units
        ha, hm = in_box(pa, it.box), in_box(pm, it.box)
        flips_to_miss += ha and not hm
        flips_to_hit += hm and not ha
    out[s] = {"n_both_valid": both, "identical_point": sum(x == 0 for x in shifts) / both,
              "shift_p50": percentile(shifts, 50), "shift_p90": percentile(shifts, 90), "shift_max": max(shifts),
              "shift_mean": statistics.fmean(shifts), "hit_to_miss": flips_to_miss, "miss_to_hit": flips_to_hit,
              "unit": "max(|dx|,|dy|) in the model's 0-1000 coordinate space (1 unit = 0.1% of the image side)"}
    print(s, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in out[s].items() if k != "unit"}))
path = ROOT / "serving" / "results" / "raw" / f"merge_diff_{time.strftime('%Y%m%d-%H%M%S')}.json"
path.write_text(json.dumps({"kind": "merge_diff", "source": src.name, "env": envinfo.capture(), "results": out}, indent=1))
print("wrote", path.name)
