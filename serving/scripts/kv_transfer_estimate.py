"""ANALYTICAL ESTIMATE (not a measurement) of what it costs to move one request's KV cache from a prefill GPU to a decode GPU.

    python serving/scripts/kv_transfer_estimate.py

Inputs: the merged model's config (layers, KV heads, head size), the token counts of the default trace as counted by the real Qwen2-VL
processor (serving/traces/default/token_stats.json), and the *theoretical* line rate of each link from its specification. Output:
bytes per request and the time those bytes need at line rate. That is a lower bound: it ignores protocol overhead, copies, kernel
launches, synchronisation and the fact that real links do not reach line rate, and it assumes the whole KV is sent after prefill rather
than layer by layer overlapped with compute. Nothing here was run on a link. The file written to results/raw/ is labelled
"estimate" and the README repeats that label next to every number from it.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "serving"))

from servebench import envinfo

# theoretical per-direction rates, GB/s (1 GB = 1e9 bytes)
LINKS = {
    "PCIe 3.0 x16": (8e9 * 16 * 128 / 130 / 8 / 1e9, "8 GT/s x 16 lanes x 128b/130b encoding, per direction (a Kaggle T4 pair sits on PCIe 3.0)"),
    "PCIe 4.0 x16": (16e9 * 16 * 128 / 130 / 8 / 1e9, "16 GT/s x 16 lanes x 128b/130b encoding, per direction"),
    "PCIe 5.0 x16": (32e9 * 16 * 128 / 130 / 8 / 1e9, "32 GT/s x 16 lanes x 128b/130b encoding, per direction"),
    "10 GbE": (10 / 8, "10 Gbit/s Ethernet line rate"),
    "25 GbE": (25 / 8, "25 Gbit/s Ethernet line rate"),
    "100 GbE": (100 / 8, "100 Gbit/s Ethernet line rate"),
}
DTYPES = {"bf16": 2, "fp8": 1}


def kv_bytes_per_token(layers: int, kv_heads: int, head_dim: int, dtype_bytes: int) -> int:
    """K and V for every layer: 2 x layers x kv_heads x head_dim x bytes."""
    return 2 * layers * kv_heads * head_dim * dtype_bytes


def transfer_ms(n_bytes: float, gb_per_s: float) -> float:
    return n_bytes / (gb_per_s * 1e9) * 1e3


def build(cfg: dict, stats: dict, measured_ttft_ms: float) -> dict:
    t = cfg["text_config"]
    head_dim = t.get("head_dim") or t["hidden_size"] // t["num_attention_heads"]
    prompts = {"shared prefix (system + 2 few-shot screenshots)": stats["shared_prefix_tokens_approx"],
               "first turn of a session": stats["tokens_first_turn_with_prefix"],
               "last measured turn": stats["tokens_last_turn"]}
    out = {"model": {"layers": t["num_hidden_layers"], "kv_heads": t["num_key_value_heads"], "head_dim": head_dim},
           "bytes_per_token": {d: kv_bytes_per_token(t["num_hidden_layers"], t["num_key_value_heads"], head_dim, b) for d, b in DTYPES.items()},
           "prompt_tokens": prompts, "links_gbps": {k: v[0] for k, v in LINKS.items()}, "link_notes": {k: v[1] for k, v in LINKS.items()},
           "measured_reference_ttft_ms": measured_ttft_ms, "rows": []}
    for pname, tokens in prompts.items():
        for dtype in DTYPES:
            n = tokens * out["bytes_per_token"][dtype]
            for link, (gbps, _) in LINKS.items():
                ms = transfer_ms(n, gbps)
                out["rows"].append({"prompt": pname, "tokens": tokens, "kv_dtype": dtype, "kv_bytes": n, "kv_mib": n / 2**20, "link": link,
                                    "ms_at_line_rate": ms, "fraction_of_reference_ttft": ms / measured_ttft_ms})
    return out


def markdown(rec: dict) -> str:
    m = rec["model"]
    nl = chr(10)
    lines = ["**ESTIMATE, not a measurement.** Lower bounds at each link's theoretical line rate.", "",
             f"KV cache per token: 2 (K, V) x {m['layers']} layers x {m['kv_heads']} KV heads x {m['head_dim']} dims x dtype bytes = "
             + ", ".join(f"{v:,} bytes ({k})" for k, v in rec["bytes_per_token"].items()) + ".", "",
             f"Reference for scale: the measured median time to first token of vLLM with prefix caching off at 1 client in Phase 1 was "
             f"{rec['measured_reference_ttft_ms']:.0f} ms (all turns of the trace, so it mixes prompt sizes).", "",
             "| prompt | tokens | KV dtype | KV size (MiB) | link | line rate (GB/s) | time at line rate (ms) | as a share of the reference TTFT |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rec["rows"]:
        lines.append(f"| {r['prompt']} | {r['tokens']} | {r['kv_dtype']} | {r['kv_mib']:.1f} | {r['link']} | {rec['links_gbps'][r['link']]:.2f} | "
                     f"{r['ms_at_line_rate']:.1f} | {100 * r['fraction_of_reference_ttft']:.1f}% |")
    return nl.join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=str(ROOT / "serving" / "models" / "merged-clean"))
    ap.add_argument("--stats", default=str(ROOT / "serving" / "traces" / "default" / "token_stats.json"))
    ap.add_argument("--phase1-csv", default=str(ROOT / "serving" / "results" / "tables" / "phase1.csv"))
    a = ap.parse_args()
    cfg = json.loads((Path(a.model_dir) / "config.json").read_text())
    stats = json.loads(Path(a.stats).read_text())
    import csv
    ttft = next(float(r["median"]) for r in csv.DictReader(open(a.phase1_csv))
                if r["run"] == "vllm-prefix-off" and r["concurrency"] == "1" and r["metric"] == "ttft_p50") * 1e3
    rec = build(cfg, stats, ttft)
    out = ROOT / "serving" / "results" / "raw" / f"kv_estimate_{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(envinfo.dumps_scrubbed({"kind": "estimate", "label": "ANALYTICAL ESTIMATE, NOT A MEASUREMENT",
                                           "env": envinfo.capture(a.model_dir, {"args": vars(a)}), **rec}, indent=1))
    tab = ROOT / "serving" / "results" / "tables" / "phase4_estimate.md"
    tab.parent.mkdir(parents=True, exist_ok=True)
    tab.write_text(markdown(rec) + chr(10), encoding="utf-8")
    print("wrote", out.name, "and", tab.name)
