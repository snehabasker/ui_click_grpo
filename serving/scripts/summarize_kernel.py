"""Phase 3 tables from the raw kernel benchmark files. The only source of Phase 3 numbers in the README.

    python serving/scripts/summarize_kernel.py --out serving/results/tables/phase3

Reads every serving/results/raw/kernel_bench_*.json (written by bench_kernel.py). Cells are "median (min–max)" over the repeats.
Speed-ups are ratios of medians. "% of copy" divides the fused kernel's effective GB/s by the measured device-to-device copy
bandwidth on the same run (a ceiling measured on this GPU, not a spec-sheet number).
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

NL = chr(10)
IMPLS = ("triton_fused", "torch_eager", "torch_compile")
CEILING = "copy_256MiB_GBps"
SPEC_GBPS_REPORTED = 384.0   # RTX 5050 Laptop, 128-bit GDDR7 at 24 Gbps; secondary-source figure, see serving/README.md


def load(raw: Path) -> list[dict]:
    out = []
    for f in sorted(raw.glob("kernel_bench_*.json")):
        d = json.loads(f.read_text())
        d["_file"] = f.name
        out.append(d)
    return out


def group(d: dict) -> dict:
    g = defaultdict(list)
    for r in d["runs"]:
        if "impl" in r:
            g[(r["dtype"], r["hidden"], r["rows"], r["impl"])].append(r)
    return g


def fmt(mr: dict, spec: str, scale: float = 1.0) -> str:
    if mr["median"] is None:
        return "–"
    f = lambda x: spec.format(x * scale)
    return f"{f(mr['median'])} ({f(mr['min'])}–{f(mr['max'])})" if mr["n"] > 1 else f(mr["median"])


def stat(runs: list[dict], key: str) -> dict:
    return median_range([r[key] for r in runs])


def ceilings(d: dict) -> dict:
    keys = sorted({k for r in d["runs"] if "ceilings" in r for k in r["ceilings"]})
    return {k: median_range([r["ceilings"][k] for r in d["runs"] if "ceilings" in r]) for k in keys}


def spec_peak_from_clocks(d: dict):
    """Memory clock x bus width x 2 transfers per clock, from what the driver reports (None if not exposed)."""
    p = d["env"].get("device_props", {})
    if p.get("mem_clock_rate") and p.get("mem_bus_width"):
        return p["mem_clock_rate"] * 1e3 * 2 * p["mem_bus_width"] / 8 / 1e9
    return None


def speed_table(d: dict, g: dict, dtype: str, key: str, gbps_key: str, with_ceiling: bool) -> list[str]:
    cells = sorted({(h, m) for (dt, h, m, _) in g if dt == dtype})
    cmed = ceilings(d)[CEILING]["median"]
    head = ["hidden", "rows", "fused (ms)", "eager (ms)", "compile (ms)", "fused vs eager", "fused vs compile", "fused (GB/s)"]
    if with_ceiling:
        head.append("% of copy")
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for h, m in cells:
        runs = {i: g[(dtype, h, m, i)] for i in IMPLS}
        if any(not v for v in runs.values()) or any(r[key] is None for v in runs.values() for r in v):
            continue
        med = {i: stat(runs[i], key)["median"] for i in IMPLS}
        row = [str(h), str(m)] + [fmt(stat(runs[i], key), "{:.4f}") for i in IMPLS]
        row += [f"{med['torch_eager'] / med['triton_fused']:.1f}x", f"{med['torch_compile'] / med['triton_fused']:.2f}x"]
        gb = stat(runs["triton_fused"], gbps_key)
        row.append(fmt(gb, "{:.0f}"))
        if with_ceiling:
            row.append(f"{100 * gb['median'] / cmed:.0f}%")
        out.append("| " + " | ".join(row) + " |")
    return out


def tables(raw: Path) -> str:
    out = []
    for d in load(raw):
        g = group(d)
        e = d["env"]
        props = e.get("device_props", {})
        reps = sorted({len(v) for v in g.values()})
        out += [f"### `{d['_file']}`", "",
                f"{e['gpu']['gpus'][0]['name']}, driver {e['gpu']['gpus'][0]['driver']}, torch {e['torch']}, triton {e['triton']}, "
                f"CUDA runtime {e['cuda_runtime']}, commit `{(e.get('repo_commit') or '?')[:7]}`; L2 cache "
                f"{(props.get('l2_cache_bytes') or 0) / 2**20:.0f} MiB. Median (min–max) over {', '.join(map(str, reps))} repeats; "
                "within a repeat the implementation order alternates and odd repeats walk the grid backwards.", ""]
        c = ceilings(d)
        peak = spec_peak_from_clocks(d)
        out += ["Memory-bandwidth ceilings (GB/s):", "", "| ceiling | GB/s |", "|---|---|"]
        for k, v in c.items():
            out.append(f"| measured {k.replace('_GBps', '').replace('_', ' ')} | {fmt(v, '{:.0f}')} |")
        out.append(f"| reported spec peak (secondary source) | {SPEC_GBPS_REPORTED:.0f} |")
        if peak:
            out.append(f"| memory clock x bus width x 2, as the driver reports them | {peak:.0f} |")
        out.append("")
        for dtype in sorted({k[0] for k in g}):
            out += [f"**{dtype}, `do_bench` (L2 flushed before each launch, includes launch cost)**", ""]
            out += speed_table(d, g, dtype, "ms", "gbps", True) + [""]
            out += [f"**{dtype}, `do_bench_cudagraph` (launch cost removed; L2 is not flushed, so tensors smaller than L2 are cache-warm "
                    "and GB/s can exceed DRAM bandwidth)**", ""]
            out += speed_table(d, g, dtype, "ms_cudagraph", "gbps_cudagraph", False) + [""]
        corr = d["correctness"].values()
        out += ["Correctness measured in the benchmark (fused kernel vs fp32 reference, every cell):", "",
                f"* largest integer-code difference: {max(v['max_code_diff'] for v in corr)}",
                f"* largest fraction of codes that differ: {max(v['frac_codes_differing'] for v in corr):.2e}",
                f"* largest relative error of the scales: {max(v['max_rel_scale_err'] for v in corr):.2e}",
                f"* torch.compile reference vs eager reference, largest code difference: "
                f"{max(v['compiled_vs_eager_max_code_diff'] for v in corr)}", ""]
        out += ["GPU state while each repeat ran (nvidia-smi, 1 Hz):", "",
                "| repeat | median power (W) | median SM clock (MHz) | min SM clock (MHz) | max temp (C) | throttle reasons seen |",
                "|---|---|---|---|---|---|"]
        for r in d["runs"]:
            if "gpu_during_rep" in r:
                s = r["gpu_during_rep"]
                out.append(f"| {r['rep'] + 1} | {s['power_w_median']:.0f} | {s['sm_mhz_median']:.0f} | {s['sm_mhz_min']:.0f} | "
                           f"{s['temp_c_max']:.0f} | {', '.join(s['throttle_reasons_seen'])} |")
        out.append("")
    return NL.join(out)


def csv_rows(raw: Path):
    for d in load(raw):
        for (dt, h, m, impl), runs in sorted(group(d).items()):
            for key in ("ms", "ms_cudagraph", "gbps", "gbps_cudagraph"):
                if any(r[key] is None for r in runs):
                    continue
                mr = stat(runs, key)
                yield [d["_file"], dt, h, m, impl, key, mr["median"], mr["min"], mr["max"], mr["n"]]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(ROOT / "serving" / "results" / "raw"))
    ap.add_argument("--out", default=str(ROOT / "serving" / "results" / "tables" / "phase3"))
    a = ap.parse_args()
    raw, out = Path(a.raw), Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".md").write_text(tables(raw), encoding="utf-8")
    with open(out.with_suffix(".csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "dtype", "hidden", "rows", "impl", "metric", "median", "min", "max", "n"])
        w.writerows(csv_rows(raw))
    print("wrote", out.with_suffix(".md"), "and", out.with_suffix(".csv"))
