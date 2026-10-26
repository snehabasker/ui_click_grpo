"""Extract the few facts the README cites from the engines' server logs into a small committed JSON.
The logs themselves stay out of the repo (they carry local paths); this file keeps those numbers traceable.

    python serving/scripts/extract_server_facts.py --logs ~/work/logs --out serving/results/raw/server_facts_phase1.json
"""
import argparse
import json
import re
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--logs", default=str(Path.home() / "work" / "logs"))
ap.add_argument("--out", required=True)
a = ap.parse_args()
logs = Path(a.logs)


def newest(label: str) -> Path | None:
    # exact label + timestamp, so "vllm-prefix-on" never picks up the log of "vllm-prefix-on-rerun"
    pat = re.compile(rf"^{re.escape(label)}-\d{{8}}-\d{{6}}\.log$")
    xs = sorted((p for p in logs.glob("*.log") if pat.match(p.name)), key=lambda p: p.stat().st_mtime)
    return xs[-1] if xs else None


def sglang_facts(path: Path) -> dict:
    t = path.read_text(errors="ignore")
    out = {"log": path.name}
    m = re.search(r"Capture target decode CUDA graph end\. bs=(\[[\d, ]+\])", t)
    out["decode_cuda_graph_batch_sizes"] = json.loads(m.group(1)) if m else None
    m = re.search(r"max_total_num_tokens=(\d+), chunked_prefill_size=(\d+), max_prefill_tokens=(\d+)", t)
    if m:
        out.update(kv_pool_tokens=int(m.group(1)), chunked_prefill_size=int(m.group(2)), max_prefill_tokens=int(m.group(3)))
    usage = [float(x) for x in re.findall(r"token usage: ([0-9.]+)", t)]
    out["max_token_usage"] = max(usage) if usage else None
    dec = re.findall(r"Decode batch, #running-req: (\d+),.*?cuda graph: (True|False)", t)
    big = [(int(n), g) for n, g in dec if int(n) > 8]
    small = [(int(n), g) for n, g in dec if int(n) <= 8 and int(n) > 0]
    out["decode_batches_logged"] = {"running_le_8": {"n": len(small), "cuda_graph_true": sum(g == "True" for _, g in small)},
                                    "running_gt_8": {"n": len(big), "cuda_graph_true": sum(g == "True" for _, g in big)}}
    pre = re.findall(r"Prefill batch, #new-seq: \d+, #new-token: \d+, #cached-token: (\d+)", t)
    out["prefill_batches"] = {"total": len(pre), "with_ge_1000_cached_tokens": sum(int(c) >= 1000 for c in pre)}
    return out


def vllm_facts(path: Path) -> dict:
    t = path.read_text(errors="ignore")
    out = {"log": path.name}
    m = re.search(r"GPU KV cache size: ([\d,]+) tokens", t)
    out["kv_cache_tokens"] = int(m.group(1).replace(",", "")) if m else None
    m = re.search(r"Using (\w+) attention backend", t)
    out["attention_backend"] = m.group(1) if m else None
    out["preemption_lines"] = len(re.findall(r"preempt", t, re.I))
    m = re.search(r"Graph capturing finished in (\d+) secs", t)
    out["cuda_graph_capture_s"] = int(m.group(1)) if m else None
    return out


facts = {"kind": "server_facts", "date": time.strftime("%Y-%m-%dT%H:%M:%S"),
         "note": "extracted from engine server logs of the Phase 1 runs; logs themselves are not committed"}
LABELS = ["sglang-prefix-on", "sglang-prefix-off", "vllm-prefix-on", "vllm-prefix-off",
          "sglang-flashinfer-bs32-prefix-on", "sglang-triton-bs32-prefix-on", "vllm-prefix-on-rerun"]
for label in LABELS:
    p = newest(label)
    if p:
        facts[label] = sglang_facts(p) if label.startswith("sglang") else vllm_facts(p)
Path(a.out).write_text(json.dumps(facts, indent=1))
print(json.dumps({k: v for k, v in facts.items() if k not in ("note",)}, indent=1)[:2500])
