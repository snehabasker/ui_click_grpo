"""Phase 4 tables from the raw disaggregation benchmark. The only source of Phase 4 measured numbers in the README.

    python serving/scripts/summarize_disagg.py --out serving/results/tables/phase4

Reads the final ``disagg_bench_<ts>.json`` (it lists, for every round, the setup, the concurrency and the load-test files it produced)
and those load-test files. Each cell is median (min–max) over the rounds. The order of the two setups alternates between rounds, so
the rounds are labelled with it. ``prefill step`` is the proxy's own timing of the prefill instance's reply (a one-token request); the
proxy also saw the discarded warm-up requests, so that column covers them too.
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "serving"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from servebench.stats import median_range
from servebench.stats import sign_test_p
from summarize_router import COLS, cell

NL = chr(10)
SETUPS = ("colocated", "split")


def load_runs(raw: Path, bench: dict) -> dict:
    """{(setup, concurrency): [{'loadtest': dict, 'prefill_s': [..] | None, 'kv_bytes': int | None, 'round': int}]}"""
    out = {}
    prev = {}
    for rnd in bench["rounds"]:
        prev_len = 0
        for run in rnd["runs"]:
            files = [f for f in run["loadtest_files"] if f.startswith("loadtest_")]
            if len(files) != 1:
                continue
            d = json.loads((raw / files[0]).read_text())
            cum = run.get("proxy_prefill_s_cumulative")
            sl = cum[prev_len:] if cum is not None else None
            prev_len = len(cum) if cum is not None else 0
            out.setdefault((rnd["setup"], run["concurrency"]), []).append(
                {"loadtest": d, "prefill_s": sl, "kv_bytes": run.get("kv_store_bytes_after_run"), "round": rnd["round"]})
    return out


def tables(raw: Path) -> str:
    out = []
    for f in sorted(p for p in raw.glob("*_bench_2*.json") if p.name.startswith("disagg")):   # not explore_*, not router_bench_*
        b = json.loads(f.read_text())
        runs = load_runs(raw, b)
        s = b["settings"]
        orders = {r["round"]: " then ".join(r["order"]) for r in b["rounds"]}
        concs = sorted({c for _, c in runs})
        pe = b.get("env", {}).get("args", {}).get("producer_extra") or "none (vLLM defaults, chunked prefill on)"
        out += [f"### `{f.name}`", "", f"Prefill-instance extra flags: `{pe}`.", "",
                f"One GPU. Setups: **colocated** = one vLLM instance, FP8 weights, {s['kv_gib_colocated']} GiB KV cache; **split** = a prefill "
                f"instance and a decode instance (FP8, {s['kv_gib_split_each']} GiB KV each) connected by `{s['connector']}`, behind the "
                f"prefill/decode proxy. max_num_seqs {s['max_seqs']}, max_model_len {s['max_len']}. Median (min–max) over "
                f"{len({r['round'] for rs in runs.values() for r in rs})} rounds; order of the setups per round: "
                + "; ".join(f"round {k + 1}: {v}" for k, v in sorted(orders.items())) + ".", ""]
        for c in concs:
            out += [f"**c={c}**", "", "| setup | " + " | ".join(t for t, *_ in COLS) + " | prefill step p50 (s) | KV store per request (MiB) |",
                    "|---|" + "---|" * (len(COLS) + 2)]
            for setup in SETUPS:
                rs = runs.get((setup, c))
                if not rs:
                    continue
                cells = [cell(median_range([g(r["loadtest"]) for r in rs]), sc, fm) for _, g, sc, fm in COLS]
                pf = median_range([statistics.median(r["prefill_s"]) for r in rs if r["prefill_s"]])
                kv = median_range([r["kv_bytes"] / len(r["prefill_s"]) / 2**20 for r in rs if r["kv_bytes"] and r["prefill_s"]])
                out.append(f"| {setup} | " + " | ".join(cells) + f" | {cell(pf, 1, '{:.3f}')} | {cell(kv, 1, '{:.1f}')} |")
            out.append("")
    return NL.join(out)


def accuracy_table(raw: Path, tag_a: str = "disagg-colocated", tag_b: str = "disagg-split") -> str:
    """Paired comparison of two tags' click accuracy on each fixed set (rule 3): latest run of each tag, sign test on the discordant items."""
    out = ["| set | n | colocated acc | split acc | delta (split - colocated) | only colocated right / only split right | sign-test p |", "|---|---|---|---|---|---|---|"]
    for s in ("synth300", "screenspot400"):
        fa = sorted(raw.glob(f"accuracy_{tag_a}_{s}_*.json"))
        fb = sorted(raw.glob(f"accuracy_{tag_b}_{s}_*.json"))
        if not fa or not fb:
            continue
        a, b = (json.loads(f[-1].read_text()) for f in (fa, fb))
        ha, hb = a["score"]["per_item_hit"], b["score"]["per_item_hit"]
        oa, ob = sum(x and not y for x, y in zip(ha, hb)), sum(y and not x for x, y in zip(ha, hb))
        out.append(f"| {s} | {a['score']['n']} | {100 * a['score']['acc']:.2f}% | {100 * b['score']['acc']:.2f}% | "
                   f"{100 * (b['score']['acc'] - a['score']['acc']):+.2f} pts | {oa} / {ob} | {sign_test_p(oa, ob):.3f} |")
    return NL.join(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(ROOT / "serving" / "results" / "raw"))
    ap.add_argument("--out", default=str(ROOT / "serving" / "results" / "tables" / "phase4"))
    a = ap.parse_args()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    acc = accuracy_table(Path(a.raw))
    out.with_suffix(".md").write_text(tables(Path(a.raw)) + NL + "## Click accuracy through the split path vs one colocated instance" + NL + NL + acc + NL, encoding="utf-8")
    print("wrote", out.with_suffix(".md"))
