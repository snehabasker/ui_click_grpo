"""Phase 2 tables from the raw router benchmark files. The only source of Phase 2 numbers in the README.

    python serving/scripts/summarize_router.py --out serving/results/tables/phase2

Each `loadtest_router-<policy>_c<N>_r1_<ts>.json` is attributed to the benchmark invocation that produced it: the latest
`router_bench_<ts>.json` whose timestamp is not after the loadtest's. Invocations are labelled by the policy order they ran in
(its `args.policies`), because run order is a confound on this laptop GPU (power/thermal state drifts over a long run).
"""
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "serving"))

from servebench.stats import median_range

NL = chr(10)
TS = re.compile(r"_(\d{8}-\d{6})\.json$")
POLICIES = ("prefix", "round-robin")


def ts_of(name: str) -> str:
    m = TS.search(name)
    return m.group(1) if m else ""


def load_benches(raw: Path) -> list[dict]:
    out = []
    for f in sorted(raw.glob("router_bench_*.json")):
        d = json.loads(f.read_text())
        d["_ts"], d["_file"] = ts_of(f.name), f.name
        args = d.get("env", {}).get("args", {})
        d["_order"] = ("interleaved, " + " / ".join(args["policies"]) + " alternating first" if args.get("interleave")
                       else " then ".join(args["policies"]) if args else "?")
        out.append(d)
    return out


def attribute(raw: Path, benches: list[dict]) -> dict:
    """{(bench file, policy, concurrency): [run summary dicts]} for every router loadtest."""
    groups = defaultdict(list)
    for f in sorted(raw.glob("loadtest_router-*_c*_r1_*.json")):
        if "router-failure" in f.name:   # the failure test's own load run, reported with the failure record
            continue
        d = json.loads(f.read_text())
        owner = [b for b in benches if b["_ts"] <= ts_of(f.name)]
        if not owner:
            continue
        b = owner[-1]
        pol = d["tag"].removeprefix("router-")
        groups[(b["_file"], pol, d["concurrency"])].append(d)
    return groups


def metric(runs: list[dict], getter) -> dict:
    return median_range([getter(r) for r in runs])


def cell(mr: dict, scale: float, fmt: str) -> str:
    if mr["median"] is None:
        return "–"
    f = lambda x: fmt.format(x * scale)
    return f"{f(mr['median'])} ({f(mr['min'])}–{f(mr['max'])})" if mr["n"] > 1 else f(mr["median"])


COLS = [  # title, getter, scale, fmt
    ("TTFT p50 (s)", lambda r: r["summary"]["ttft"]["p50"], 1, "{:.2f}"),
    ("TTFT p99 (s)", lambda r: r["summary"]["ttft"]["p99"], 1, "{:.2f}"),
    ("e2e p50 (s)", lambda r: r["summary"]["e2e"]["p50"], 1, "{:.2f}"),
    ("req/s", lambda r: r["summary"]["req_per_s"], 1, "{:.2f}"),
    ("prompt tokens from cache", lambda r: r["summary"].get("cached_token_fraction"), 100, "{:.0f}%"),
    ("errors", lambda r: r["summary"]["error_rate"], 100, "{:.1f}%"),
    ("peak temp (C)", lambda r: (r.get("gpu") or {}).get("temp_c_max"), 1, "{:.0f}"),
    ("median power (W)", lambda r: (r.get("gpu") or {}).get("power_w_median"), 1, "{:.0f}"),
]


def decisions(bench: dict, policy: str, conc: int) -> dict:
    """Router decision counts summed over the repeats of one (policy, concurrency) cell."""
    tot = defaultdict(float)
    for s in bench["router_metric_snapshots"]:
        if s["policy"] == policy and s["concurrency"] == conc:
            for k, v in s["after"].items():
                if k.startswith("router_route_decisions_total"):
                    tot[re.search(r'decision="(\w+)"', k).group(1)] += v - s["before"].get(k, 0)
    return dict(tot)


def tables(raw: Path) -> str:
    benches = load_benches(raw)
    groups = attribute(raw, benches)
    out = []
    for b in benches:
        cells = {(p, c): runs for (bf, p, c), runs in groups.items() if bf == b["_file"]}
        if not cells:
            continue
        concs = sorted({c for _, c in cells})
        reps = sorted({len(v) for v in cells.values()})
        rs, e = b["replica_settings"], b["env"]
        out += [f"### Run order: {b['_order']}  (`{b['_file']}`)", "",
                f"Two {rs['quantization'].upper()} replicas, {rs['kv_cache_gib_each']} GiB KV cache each "
                f"({b['replicas'][0]['kv_cache_tokens']:,} tokens), max_num_seqs {rs['max_seqs']}, max_model_len {rs['max_len']}. "
                f"Median (min–max) over {', '.join(map(str, reps))} repeats; caches flushed before every repeat.", ""]
        for c in concs:
            out += [f"**c={c}**", "", "| policy | " + " | ".join(t for t, *_ in COLS) + " |", "|---|" + "---|" * len(COLS)]
            for p in POLICIES:
                if (p, c) in cells:
                    runs = cells[(p, c)]
                    out.append(f"| {p} | " + " | ".join(cell(metric(runs, g), s, f) for _, g, s, f in COLS) + " |")
            out.append("")
        out += ["Router decisions (all repeats of the cell):", "", "| policy | c | affinity | spill | round-robin |", "|---|---|---|---|---|"]
        for p in POLICIES:
            for c in concs:
                if (p, c) in cells:
                    d = decisions(b, p, c)
                    out.append(f"| {p} | {c} | {int(d.get('affinity', 0))} | {int(d.get('spill', 0))} | {int(d.get('round_robin', 0))} |")
        out.append("")
    return NL.join(out)


def failure_section(raw: Path) -> str:
    out = []
    for f in sorted(raw.glob("router_failure_*.json")):
        d = json.loads(f.read_text())
        out += [f"### `{f.name}`", "", "```", json.dumps(d.get("summary", {k: d[k] for k in d if k not in ("env", "timeline")}),
                                                      indent=1)[:3000], "```", ""]
    return NL.join(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(ROOT / "serving" / "results" / "raw"))
    ap.add_argument("--out", default=str(ROOT / "serving" / "results" / "tables" / "phase2"))
    a = ap.parse_args()
    raw = Path(a.raw)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    body = "## Prefix-aware vs round-robin routing" + NL * 2 + tables(raw) + NL + "## Failure test" + NL * 2 + failure_section(raw)
    out.with_suffix(".md").write_text(body, encoding="utf-8")
    print(f"-> {out.with_suffix('.md')}")
