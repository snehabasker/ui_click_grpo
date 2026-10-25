"""Replay a trace against an OpenAI-compatible endpoint and write raw JSON (one file per repeat).

    python serving/scripts/loadtest.py --base-url http://localhost:8000 --model merged-clean \
        --trace serving/traces/default --concurrency 4 --repeats 3 --engine vllm --tag vllm-c4 \
        --flush-url http://localhost:8000/reset_prefix_cache --model-dir serving/models/merged-clean

Each repeat: optional cache flush, a discarded warm-up (unique prefixes, so it does not prime the shared ones), then the
measured replay. Median and spread over repeats are computed later by scripts/summarize.py from these raw files.
"""
import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import aiohttp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "serving"))

from servebench import envinfo
from servebench.client import run_load
from servebench.traces import load_trace

ap = argparse.ArgumentParser()
ap.add_argument("--base-url", default="http://localhost:8000")
ap.add_argument("--model", required=True)
ap.add_argument("--trace", required=True)
ap.add_argument("--concurrency", type=int, required=True)
ap.add_argument("--repeats", type=int, default=3)
ap.add_argument("--max-sessions", type=int, default=None)
ap.add_argument("--max-tokens", type=int, default=24)
ap.add_argument("--engine", required=True, help="label recorded in the result, e.g. vllm / sglang / router-prefix")
ap.add_argument("--tag", required=True)
ap.add_argument("--flush-url", default=None, help="POST here before each repeat to empty the prefix cache")
ap.add_argument("--model-dir", default=None)
ap.add_argument("--server-cmd", default=None, help="the command that launched the server (recorded verbatim)")
ap.add_argument("--gpu-index", type=int, default=None)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--mock", action="store_true", help="marks the output as coming from the mock server: NOT a result")
a = ap.parse_args()

trace_dir = Path(a.trace)
sessions = load_trace(trace_dir)[: a.max_sessions]
warm = load_trace(trace_dir, "warmup.jsonl")
out_dir = ROOT / "serving" / "results" / "raw"
out_dir.mkdir(parents=True, exist_ok=True)


async def flush():
    if a.flush_url:
        async with aiohttp.ClientSession() as s:
            async with s.post(a.flush_url) as r:
                return r.status


for rep in range(1, a.repeats + 1):
    fs = asyncio.run(flush()) if a.flush_url else None
    asyncio.run(run_load(a.base_url, a.model, trace_dir, warm, concurrency=1, max_tokens=a.max_tokens))   # discarded
    res = asyncio.run(run_load(a.base_url, a.model, trace_dir, sessions, a.concurrency, a.max_tokens, gpu_index=a.gpu_index))
    rec = {"kind": "loadtest", "mock": a.mock, "engine": a.engine, "tag": a.tag, "repeat": rep, "concurrency": a.concurrency,
           "seed": a.seed, "flush_status": fs, "server_cmd": a.server_cmd,
           "trace": {"dir": str(trace_dir), "n_sessions": len(sessions),
                     "meta": json.loads((trace_dir / "trace_meta.json").read_text())},
           "env": envinfo.capture(a.model_dir, {"args": vars(a)}), **res}
    name = f"{'MOCK_' if a.mock else ''}loadtest_{a.tag}_c{a.concurrency}_r{rep}_{time.strftime('%Y%m%d-%H%M%S')}.json"
    (out_dir / name).write_text(json.dumps(rec, indent=1))
    s = res["summary"]
    print(f"[{a.tag} c={a.concurrency} r{rep}] ok={s['ok']}/{s['requests']} ttft p50={s['ttft']['p50']} e2e p50={s['e2e']['p50']} "
          f"req/s={s['req_per_s']:.2f} acc={s.get('trace_click_accuracy', {}).get('acc')} -> {name}")
