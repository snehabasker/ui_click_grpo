"""Aggregate raw load-test files into tables (median and min-max over repeats). The only source of numbers for the README.

    python serving/scripts/summarize.py --out serving/results/tables/phase1
Excluded on purpose: MOCK_* files (mock server) and anything tagged 'explore' (exploratory first looks).
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "serving"))

from servebench.stats import median_range

METRICS = [  # key, title, scale, format
    ("ttft_p50", "Time to first token, p50 (s)", 1, "{:.2f}"),
    ("ttft_p99", "Time to first token, p99 (s)", 1, "{:.2f}"),
    ("itl_p50", "Inter-token latency, p50 (ms)", 1000, "{:.1f}"),
    ("e2e_p50", "End-to-end latency, p50 (s)", 1, "{:.2f}"),
    ("e2e_p99", "End-to-end latency, p99 (s)", 1, "{:.2f}"),
    ("req_per_s", "Throughput (requests/s)", 1, "{:.2f}"),
    ("cached_frac", "Prompt tokens served from the prefix cache (engine-reported)", 100, "{:.0f}%"),
    ("trace_acc", "Click accuracy on the trace requests", 100, "{:.1f}%"),
    ("error_rate", "Request error rate", 100, "{:.1f}%"),
    ("gpu_mem_max_mib", "Peak GPU memory (MiB)", 1, "{:.0f}"),
]


def extract(run: dict) -> dict:
    s = run["summary"]
    return {
        "ttft_p50": s["ttft"]["p50"], "ttft_p99": s["ttft"]["p99"], "itl_p50": s["itl"]["p50"],
        "e2e_p50": s["e2e"]["p50"], "e2e_p99": s["e2e"]["p99"], "req_per_s": s["req_per_s"],
        "cached_frac": s.get("cached_token_fraction"),
        "trace_acc": (s.get("trace_click_accuracy") or {}).get("acc"), "error_rate": s["error_rate"],
        "gpu_mem_max_mib": (run.get("gpu") or {}).get("mem_used_mib_max"),
    }


def load_runs(raw: Path) -> list[dict]:
    runs = []
    for f in sorted(raw.glob("loadtest_*.json")):
        if f.name.startswith("MOCK_") or "explore" in f.name:
            continue
        d = json.loads(f.read_text())
        if d.get("mock"):
            continue
        d["_file"] = f.name
        runs.append(d)
    return runs


def aggregate(runs: list[dict]) -> dict:
    groups = defaultdict(list)
    for r in runs:
        groups[(r["tag"], r["concurrency"])].append(extract(r))
    return {k: {m: median_range([x[m] for x in v]) | {"repeats": len(v)} for m, *_ in METRICS} for k, v in groups.items()}


def cell(mr: dict, scale: float, fmt: str) -> str:
    if mr["median"] is None:
        return "–"
    f = lambda x: fmt.format(x * scale)
    return f"{f(mr['median'])} ({f(mr['min'])}–{f(mr['max'])})" if mr["n"] > 1 else f(mr["median"])


def markdown(agg: dict) -> str:
    tags = sorted({t for t, _ in agg})
    concs = sorted({c for _, c in agg})
    reps = sorted({v["ttft_p50"]["repeats"] for v in agg.values()})
    out = [f"Median (min–max) over repeats; repeats per cell: {', '.join(map(str, reps))}. Columns are client concurrency.\n"]
    for key, title, scale, fmt in METRICS:
        out += [f"### {title}\n", "| run | " + " | ".join(f"c={c}" for c in concs) + " |", "|---|" + "---|" * len(concs)]
        for t in tags:
            out.append(f"| {t} | " + " | ".join(cell(agg[(t, c)][key], scale, fmt) if (t, c) in agg else "" for c in concs) + " |")
        out.append("")
    return "\n".join(out)


def write_csv(agg: dict, path: Path) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "concurrency", "metric", "median", "min", "max", "repeats"])
        for (t, c), ms in sorted(agg.items()):
            for m, mr in ms.items():
                w.writerow([t, c, m, mr["median"], mr["min"], mr["max"], mr["repeats"]])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(ROOT / "serving" / "results" / "raw"))
    ap.add_argument("--out", default=str(ROOT / "serving" / "results" / "tables" / "phase1"))
    a = ap.parse_args()
    runs = load_runs(Path(a.raw))
    agg = aggregate(runs)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".md").write_text(markdown(agg), encoding="utf-8")
    write_csv(agg, out.with_suffix(".csv"))
    print(f"{len(runs)} runs -> {len(agg)} cells -> {out.with_suffix('.md')}")
