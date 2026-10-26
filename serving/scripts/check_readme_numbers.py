"""Guard against typos: every "median (min–max)" cell in serving/README.md must exist in the generated results/tables/phase1.md,
and the ratio tables must match ratios recomputed from phase1.csv.   python serving/scripts/check_readme_numbers.py"""
import csv
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
readme = (ROOT / "serving" / "README.md").read_text(encoding="utf-8")
table = (ROOT / "serving" / "results" / "tables" / "phase1.md").read_text(encoding="utf-8")
section = readme.split("## Phase 1 results", 1)[1].split("## Rules this folder follows", 1)[0]

cells = set(re.findall(r"\d+(?:\.\d+)?%? \(\d+(?:\.\d+)?%?–\d+(?:\.\d+)?%?\)", section))
missing = sorted(c for c in cells if c not in table)
print(f"{len(cells)} 'median (min-max)' cells in the README Phase 1 section; missing from the generated table: {len(missing)}")
for m in missing:
    print("  MISSING:", m)

med = {}
with open(ROOT / "serving" / "results" / "tables" / "phase1.csv") as f:
    for r in csv.DictReader(f):
        if r["metric"] == "req_per_s":
            med[(r["run"], int(r["concurrency"]))] = float(r["median"])
bad = 0


def claimed(row_label):
    m = re.search(rf"\| {row_label} \| ([\d.]+)x \| ([\d.]+)x \| ([\d.]+)x \| ([\d.]+)x \|", section)
    return [float(x) for x in m.groups()]


for label, num, den in [("vLLM", "vllm-prefix-on", "vllm-prefix-off"), ("SGLang", "sglang-prefix-on", "sglang-prefix-off")]:
    want = [round(med[(num, c)] / med[(den, c)], 2) for c in (1, 4, 16, 32)]
    got = claimed(label)
    ok = all(abs(a - b) < 0.006 for a, b in zip(want, got))
    bad += not ok
    print(f"cache on/off throughput ratio {label}: README {got} vs recomputed {want} -> {'OK' if ok else 'MISMATCH'}")
want = [round(med[("vllm-prefix-on", c)] / med[("sglang-prefix-on", c)], 2) for c in (1, 4, 16, 32)]
m = re.search(r"\| ratio \| ([\d.]+)x \| ([\d.]+)x \| ([\d.]+)x \| ([\d.]+)x \|", section)
got = [float(x) for x in m.groups()]
ok = all(abs(a - b) < 0.006 for a, b in zip(want, got))
bad += not ok
print(f"vLLM/SGLang throughput ratio: README {got} vs recomputed {want} -> {'OK' if ok else 'MISMATCH'}")
sys.exit(1 if missing or bad else 0)
