"""Phase 5 tables from the raw TensorRT-LLM benchmark. The only source of Phase 5 numbers in the README.

    python serving/scripts/summarize_trtllm.py --out serving/results/tables/phase5

Reads ``trtllm_bench_<ts>.json`` (rounds -> engine -> per-concurrency load-test files), the accuracy runs of TensorRT-LLM and of the Phase 1 vLLM bf16
run (paired sign test), and the ``prompt_tokens_check_*`` files. Latency cells are median (min–max) over the rounds; the order of the engines alternates.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "serving"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from servebench.stats import median_range, sign_test_p
from summarize_router import COLS, cell

NL = chr(10)
ENGINES = ("vllm", "trtllm")


def load_runs(raw: Path, bench: dict) -> dict:
    out = {}
    for rnd in bench["rounds"]:
        for run in rnd["runs"]:
            files = [f for f in run["loadtest_files"] if f.startswith("loadtest_")]
            if len(files) != 1:
                continue
            out.setdefault((rnd["engine"], run["concurrency"]), []).append(json.loads((raw / files[0]).read_text()))
    return out


def tables(raw: Path) -> str:
    out = []
    for f in sorted(raw.glob("trtllm_bench_2*.json")):
        b = json.loads(f.read_text())
        runs = load_runs(raw, b)
        s, v = b["settings"], b["trtllm_env"]
        orders = {r["round"]: " then ".join(r["order"]) for r in b["rounds"]}
        out += [f"### `{f.name}`", "",
                f"One GPU, bf16, {s['trace']} (one image per request, 120 requests, no shared prefix), prefix caching/reuse {s['prefix_caching']}, max_num_seqs "
                f"{s['max_seqs']}, max_model_len {s['max_len']}. TensorRT-LLM {v.get('tensorrt_llm')} ({s['trtllm_backend']} backend) with torch {v.get('torch')}, "
                f"transformers {v.get('transformers')}; vLLM {b['env']['packages'].get('vllm', '?')}. Median (min–max) over "
                f"{len({r['round'] for r in b['rounds']})} rounds; engine order per round: " + "; ".join(f"round {k + 1}: {x}" for k, x in sorted(orders.items())) + ".", ""]
        diff = b.get("trtllm_local_patch_diff") or "(none)"
        out += ["TensorRT-LLM local patch to the installed package (vision-tower attention):", "", "```", diff, "```", ""]
        for c in sorted({c for _, c in runs}):
            out += [f"**c={c}**", "", "| engine | " + " | ".join(t for t, *_ in COLS) + " |", "|---|" + "---|" * len(COLS)]
            for e in ENGINES:
                rs = runs.get((e, c))
                if rs:
                    out.append(f"| {e} | " + " | ".join(cell(median_range([g(r) for r in rs]), sc, fm) for _, g, sc, fm in COLS) + " |")
            out.append("")
    return NL.join(out)


def accuracy_table(raw: Path) -> str:
    out = ["| set | n | vLLM bf16 (Phase 1) | TensorRT-LLM | delta (TensorRT-LLM - vLLM) | only vLLM right / only TensorRT-LLM right | sign-test p |", "|---|---|---|---|---|---|---|"]
    for s in ("synth300", "screenspot400"):
        fa = sorted(raw.glob(f"accuracy_vllm-prefix-on_{s}_*.json"))
        fb = sorted(raw.glob(f"accuracy_trtllm_{s}_*.json"))
        if not fa or not fb:
            continue
        a, b = (json.loads(f[-1].read_text()) for f in (fa, fb))
        ha, hb = a["score"]["per_item_hit"], b["score"]["per_item_hit"]
        oa, ob = sum(x and not y for x, y in zip(ha, hb)), sum(y and not x for x, y in zip(ha, hb))
        out.append(f"| {s} | {a['score']['n']} | {100 * a['score']['acc']:.2f}% | {100 * b['score']['acc']:.2f}% | "
                   f"{100 * (b['score']['acc'] - a['score']['acc']):+.2f} pts | {oa} / {ob} | {sign_test_p(oa, ob):.4f} |")
    return NL.join(out)


def tokens_table(raw: Path) -> str:
    rows = ["| engine | set | items | prompt tokens equal to the Hugging Face reference | smallest difference | largest difference |", "|---|---|---|---|---|---|"]
    for f in sorted(raw.glob("prompt_tokens_check_*.json")):
        d = json.loads(f.read_text())
        sm = d["summary"]
        rows.append(f"| {d['tag']} | {d['set']} | {sm['n']} | {sm['equal']} | {sm['min_diff']:+d} | {sm['max_diff']:+d} |")
    return NL.join(rows) if len(rows) > 2 else "(no prompt-token check files)"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(ROOT / "serving" / "results" / "raw"))
    ap.add_argument("--out", default=str(ROOT / "serving" / "results" / "tables" / "phase5"))
    a = ap.parse_args()
    raw, out = Path(a.raw), Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".md").write_text(tables(raw) + NL + "## Click accuracy: TensorRT-LLM vs vLLM bf16" + NL + NL + accuracy_table(raw) + NL + NL
                                      + "## Prompt tokens each engine reports vs the Hugging Face reference" + NL + NL + tokens_table(raw) + NL, encoding="utf-8")
    print("wrote", out.with_suffix(".md"))
