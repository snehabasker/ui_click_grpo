"""Guard against typos: every "median (min–max)" cell in serving/README.md must exist in the generated results/tables/phase1.md,
and the ratio tables must match ratios recomputed from phase1.csv.   python serving/scripts/check_readme_numbers.py"""
import csv
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
readme = (ROOT / "serving" / "README.md").read_text(encoding="utf-8")
table = (ROOT / "serving" / "results" / "tables" / "phase1.md").read_text(encoding="utf-8")
section = readme.split("## Phase 1 results", 1)[1].split("## Phase 2 results", 1)[0]
section2 = readme.split("## Phase 2 results", 1)[1].split("## Phase 3 results", 1)[0]
section3 = readme.split("## Phase 3 results", 1)[1].split("## Phase 4 results", 1)[0]
section4 = readme.split("## Phase 4 results", 1)[1].split("## Phase 5 results" if "## Phase 5 results" in readme else "## Rules this folder follows", 1)[0]

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
bad += len(missing)


# ---- prose claims in the follow-up section, recomputed from the CSV medians -------------------------------------------------
def metric(name, run, c):
    with open(ROOT / "serving" / "results" / "tables" / "phase1.csv") as f:
        for r in csv.DictReader(f):
            if r["metric"] == name and r["run"] == run and int(r["concurrency"]) == c:
                return float(r["median"])
    raise KeyError((name, run, c))


bad2 = 0
vs = [metric("req_per_s", "vllm-prefix-on", c) / metric("req_per_s", "sglang-flashinfer-bs32-prefix-on", c) for c in (1, 4, 16, 32)]
claim = f"about {vs[0]:.1f}x at c=1, {vs[1]:.1f}x at c=4, and {vs[2]:.1f}x and {vs[3]:.1f}x at c=16 and c=32 against B"
ok = claim in " ".join(section.split())
bad2 += not ok
print(f"vLLM vs SGLang FlashInfer ratios in prose: expected '{claim}' -> {'OK' if ok else 'NOT FOUND / MISMATCH'}")
thr = 1 - metric("req_per_s", "vllm-prefix-on-rerun", 16) / metric("req_per_s", "vllm-prefix-on", 16)
ttft = metric("ttft_p50", "vllm-prefix-on-rerun", 16) / metric("ttft_p50", "vllm-prefix-on", 16) - 1
for label, text in ((f"{100 * thr:.0f}% lower", "rerun throughput"), (f"{100 * ttft:.0f}% higher", "rerun time to first token")):
    ok = label in section
    bad2 += not ok
    print(f"{text}: expected '{label}' in README -> {'OK' if ok else 'MISSING'}")

# ---- Phase 2: cells must exist in phase2.md; per-replica request counts are recomputed from the raw bench record ---------------
import json

table2 = (ROOT / "serving" / "results" / "tables" / "phase2.md").read_text(encoding="utf-8")
cells2 = set(re.findall(r"\d+(?:\.\d+)?%? \(\d+(?:\.\d+)?%?–\d+(?:\.\d+)?%?\)", section2))
missing2 = sorted(c for c in cells2 if c not in table2)
print(f"{len(cells2)} 'median (min-max)' cells in the README Phase 2 section; missing from phase2.md: {len(missing2)}")
for m in missing2:
    print("  MISSING:", m)
bad3 = len(missing2)

bench = [f for f in sorted((ROOT / "serving" / "results" / "raw").glob("router_bench_*.json"))
         if json.loads(f.read_text())["env"]["args"].get("interleave")][-1]
tot = {}
for s in json.loads(bench.read_text())["router_metric_snapshots"]:
    ok = [s["after"][k] - s["before"].get(k, 0) for k in s["after"] if k.startswith("router_backend_ok_total")]
    t = tot.setdefault((s["policy"], s["concurrency"]), [0, 0])
    t[0] += ok[0]
    t[1] += ok[1]
fmt = lambda p, cs: ", ".join(f"{int(tot[(p, c)][0])} / {int(tot[(p, c)][1])}" for c in cs)
claim = (f"served {fmt('prefix', (4,))} requests at c=4, {fmt('prefix', (16,))} at c=16 and {fmt('prefix', (32,))} at c=32 "
         f"(round-robin: {fmt('round-robin', (4,))}, {fmt('round-robin', (16,))}, {fmt('round-robin', (32,))})")
ok = claim in " ".join(section2.split())
bad3 += not ok
print(f"per-replica request counts ({bench.name}): expected '{claim}' -> {'OK' if ok else 'NOT FOUND / MISMATCH'}")

# ---- Phase 3: table rows are verbatim from phase3.md; prose ranges are recomputed from the raw benchmark and accuracy files -----------
table3 = (ROOT / "serving" / "results" / "tables" / "phase3.md").read_text(encoding="utf-8")
rows3 = [l.strip() for l in section3.splitlines() if l.startswith("| ") and any(ch.isdigit() for ch in l) and not set(l) <= set("|- ")]
missing3 = [r for r in rows3 if r not in table3]
print(f"{len(rows3)} table rows in the README Phase 3 section; not verbatim in phase3.md: {len(missing3)}")
for r in missing3:
    print("  MISSING:", r[:120])
bad4 = len(missing3)

kb = json.loads(sorted((ROOT / "serving" / "results" / "raw").glob("kernel_bench_*.json"))[-1].read_text())
runs = [r for r in kb["runs"] if "impl" in r]
cellmed = {}
for key, name in (("ms", "ms"), ("ms_cudagraph", "graph")):
    for r in runs:
        cellmed.setdefault((r["dtype"], r["hidden"], r["rows"], r["impl"], name), []).append(r[key])
import statistics
cellmed = {k: statistics.median(v) for k, v in cellmed.items()}
cells = sorted({k[:3] for k in cellmed})
claims = []
for name, label in (("ms", "do_bench"), ("graph", "graph")):
    ve = [cellmed[c + ("torch_eager", name)] / cellmed[c + ("triton_fused", name)] for c in cells]
    vc = {c: cellmed[c + ("torch_compile", name)] / cellmed[c + ("triton_fused", name)] for c in cells}
    if name == "ms":
        tied = [c for c, v in vc.items() if v < 1.05]
        rest = [v for c, v in vc.items() if v >= 1.05]
        big = [v for c, v in vc.items() if c[2] >= 1024]
        claims += [f"{min(ve):.1f}x to {max(ve):.1f}x faster than eager", f"{min(vc.values()):.2f}x to {max(vc.values()):.2f}x",
                   f"tied ({min(vc[c] for c in tied):.2f}x-{max(vc[c] for c in tied):.2f}x) in {len(tied)} of the {len(cells)} cells",
                   f"{min(rest):.2f}x to {max(rest):.2f}x faster, and {min(big):.2f}x to {max(big):.2f}x at 1024 rows"]
    else:
        claims += [f"{min(ve):.1f}x to {max(ve):.1f}x against eager and {min(vc.values()):.2f}x to {max(vc.values()):.2f}x"]
# roofline: best repeat of the fused kernel in the cells that move >= 64 MiB
ceil_rows = [r["ceilings"] for r in kb["runs"] if "ceilings" in r]
best_copy = max(c["copy_256MiB_GBps"] for c in ceil_rows)
best_read = max(c["read_only_256MiB_GBps"] for c in ceil_rows)
big = {}
for r in runs:
    if r["impl"] == "triton_fused" and r["min_bytes"] >= 64 * 2**20:
        big.setdefault((r["dtype"], r["hidden"], r["rows"]), []).append(r["gbps"])
best = [max(v) for v in big.values()]
meds = [statistics.median(v) for v in big.values()]
claims += [f"{round(min(best))}-{round(max(best))} GB/s", f"{round(100 * min(best) / 384)}-{round(100 * max(best) / 384)}% of the reported spec peak",
           f"({round(100 * min(best) / best_copy)}-{round(100 * max(best) / best_copy)}%)", f"{round(100 * min(best) / best_read)}-{round(100 * max(best) / best_read)}% of the best read-only",
           f"The medians are lower ({round(min(meds))}-{round(max(meds))} GB/s)"]
ka = json.loads(sorted((ROOT / "serving" / "results" / "raw").glob("kernel_accuracy_*.json"))[-1].read_text())
mv = json.loads(sorted((ROOT / "serving" / "results" / "raw").glob("merge_verify_*.json"))[-1].read_text())
R = ka["results"]
idk = [100 * R[x]["kernel_vs_baseline"]["identical_output_text"] for x in R]
idr = [100 * R[x]["kernel_vs_reference_identical_output_text"] for x in R]
dis = [R[x]["kernel_vs_baseline"]["only_baseline_correct"] + R[x]["kernel_vs_baseline"]["only_kernel_correct"] for x in R]
gap = [100 * abs(R[x]["kernel"]["acc"] - R[x]["reference"]["acc"]) for x in R]
mid = [100 * mv["results"][x]["identical_output_text"] for x in mv["results"]]
ae = ka["activation_error"]
claims += [f"only {round(min(idk))}-{round(max(idk))}% of the kernel's outputs", f"only {round(min(idr))}-{round(max(idr))}% of their outputs are identical",
           f"with {dis[0]} and {dis[1]} discordant items", f"{gap[0]:.1f} points on both sets" if round(gap[0], 1) == round(gap[1], 1) else "GAP-DIFFERS",
           f"{mid[0]:.1f}% and {mid[1]:.1f}% identical for adapter vs merged", f"relative L2 error {ae['relative_l2_error_mean']:.3f} on average",
           f"{ae['relative_l2_error_max']:.3f} at worst"]
text3 = " ".join(section3.split())
for c in claims:
    ok = c in text3
    bad4 += not ok
    print(f"Phase 3 prose: expected '{c}' -> {'OK' if ok else 'NOT FOUND / MISMATCH'}")

# ---- Phase 4: table rows verbatim from phase4.md / phase4_estimate.md; prose numbers must appear in the generated tables ---------------
table4 = (ROOT / "serving" / "results" / "tables" / "phase4.md").read_text(encoding="utf-8") + (ROOT / "serving" / "results" / "tables" / "phase4_estimate.md").read_text(encoding="utf-8")
rows4 = [l.strip() for l in section4.splitlines() if l.startswith("| ") and any(ch.isdigit() for ch in l) and not set(l) <= set("|- ")]
missing4 = [r for r in rows4 if r not in table4]
print(f"{len(rows4)} table rows in the README Phase 4 section; not verbatim in phase4*.md: {len(missing4)}")
for r in missing4:
    print("  MISSING:", r[:120])
bad5 = len(missing4)
text4 = " ".join(section4.split())
claims4 = [  # (sentence in the README, cells that must exist in the generated tables)
    ("median time to first token 0.26 s against 0.15 s at 1 client and 0.95 s against 0.54 s at 4 clients", ["0.26 (0.25–0.26)", "0.15 (0.14–0.15)", "0.95 (0.95–1.00)", "0.54 (0.54–0.54)"]),
    ("and 2.68 against 4.07 requests/s at 1 client and 3.55 against 5.63 at 4", ["2.68 (2.67–2.69)", "4.07 (4.05–4.11)", "3.55 (3.45–3.57)", "5.63 (5.58–5.65)"]),
    ("At 4 clients 99.2% of requests failed and at 16 clients 100%", ["99.2% (99.2%–99.2%)", "100.0% (100.0%–100.0%)"]),
    ("At 16 clients it still failed (99.1% of requests)", ["99.1% (99.1%–99.1%)"]),
    ("median time to first token of 29.60 s and 0.61 requests/s against 1.53 s and 6.38 requests/s", ["29.60 (21.48–30.32)", "0.61 (0.53–0.81)", "1.53 (1.52–1.53)", "6.38 (6.37–6.39)"]),
    ("(0.2% of requests)", ["0.2% (0.0%–0.4%)"]),
    ("the prefill step alone (the proxy's own timing) is about 0.20 s of the 0.26 s", ["0.201 (0.199–0.205)"]),
    ("about 49.5 MiB per request", ["49.5 (49.5–49.5)"]),
    ("| 5.4% |" if False else "over 10 GbE 5.4%", ["38.2", "5.4%"]),
    ("No change was detected: 1 and 1 discordant items on the synthetic set, 5 and 4 on ScreenSpot", ["| 1 / 1 | 1.000 |", "| 5 / 4 | 1.000 |"]),
]
for sent, cells in claims4:
    ok = sent in text4 and all(c in table4 for c in cells)
    bad5 += not ok
    print(f"Phase 4 prose: '{sent[:70]}...' -> {'OK' if ok else 'NOT FOUND / MISMATCH'}")
import math
kv = 2 * 28 * 2 * 128 * 2
ok = "28,672 bytes" in text4 and kv == 28672 and f"{round(49.5 * 2**20 / kv):,}" == "1,810" and "about 1,810 tokens" in text4
bad5 += not ok
print(f"Phase 4 arithmetic (28,672 bytes per token; 49.5 MiB / 28,672 = 1,810 tokens) -> {'OK' if ok else 'MISMATCH'}")

# ---- Phase 5: rows verbatim from phase5.md; ratios, start-up times and power recomputed from the raw benchmark -----------------------
section5 = readme.split("## Phase 5 results", 1)[1].split("## Rules this folder follows", 1)[0]
table5 = (ROOT / "serving" / "results" / "tables" / "phase5.md").read_text(encoding="utf-8")
rows5 = [l.strip() for l in section5.splitlines() if l.startswith("| ") and any(ch.isdigit() for ch in l) and not set(l) <= set("|- ")]
missing5 = [r for r in rows5 if r not in table5]
print(f"{len(rows5)} table rows in the README Phase 5 section; not verbatim in phase5.md: {len(missing5)}")
for r in missing5:
    print("  MISSING:", r[:120])
bad6 = len(missing5)
tb = json.loads(sorted((ROOT / "serving" / "results" / "raw").glob("trtllm_bench_2*.json"))[-1].read_text())
agg = {}
for rnd in tb["rounds"]:
    for run in rnd["runs"]:
        d = json.loads((ROOT / "serving" / "results" / "raw" / [x for x in run["loadtest_files"] if x.startswith("loadtest_")][0]).read_text())
        agg.setdefault((rnd["engine"], run["concurrency"]), []).append((d["summary"]["ttft"]["p50"], d["summary"]["e2e"]["p50"], d["summary"]["req_per_s"], d["gpu"]["power_w_median"]))
m = lambda e, c, i: statistics.median(x[i] for x in agg[(e, c)])
ttft = [m("trtllm", c, 0) / m("vllm", c, 0) for c in (1, 4, 16)]
e2e = [m("trtllm", c, 1) / m("vllm", c, 1) for c in (1, 4, 16)]
thr = [m("trtllm", c, 2) / m("vllm", c, 2) for c in (1, 4, 16)]
st = {e: [r["server"]["startup_s"] for r in tb["rounds"] if r["engine"] == e] for e in ("vllm", "trtllm")}
pw = {e: [x[3] for c in (1, 4, 16) for x in agg[(e, c)]] for e in ("vllm", "trtllm")}
text5 = " ".join(section5.split())
claims5 = [f"**{ttft[0]:.1f}x, {ttft[1]:.1f}x and {ttft[2]:.1f}x** vLLM's at 1, 4 and 16 clients",
           f"end-to-end latency about {min(e2e):.1f}x to {max(e2e):.1f}x", f"about **{statistics.mean(thr):.1f}x** ({m('trtllm', 1, 2):.2f} against {m('vllm', 1, 2):.2f} requests/s at 1 client, {m('trtllm', 16, 2):.2f} against {m('vllm', 16, 2):.2f} at 16)",
           f"{round(min(st['vllm']))} to {round(max(st['vllm']))} s for vLLM and {round(min(st['trtllm']))} to {round(max(st['trtllm']))} s for TensorRT-LLM",
           f"({round(min(pw['trtllm']))} to {round(max(pw['trtllm']))} W median against {round(min(pw['vllm']))} to {round(max(pw['vllm']))} W)",
           "On the synthetic set no difference was detected (-1.67 points, p=0.33)", "TensorRT-LLM scored 4.75 points higher** (31 items only it got right, 12 only vLLM did; p=0.0054)"]
for c_ in claims5:
    ok = c_ in text5
    bad6 += not ok
    print(f"Phase 5 prose: expected '{c_[:90]}' -> {'OK' if ok else 'NOT FOUND / MISMATCH'}")
sys.exit(1 if (bad or bad2 or bad3 or bad4 or bad5 or bad6) else 0)
