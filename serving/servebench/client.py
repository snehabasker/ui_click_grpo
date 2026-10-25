"""Async load client for any OpenAI-compatible server (vLLM, SGLang, the Rust router).

Closed loop: ``concurrency`` workers each take a session and play its turns one after another (an agent waits for its own
answer before the next screenshot), then take the next session. Every request streams, so time-to-first-token and the
inter-token gaps are measured at the client, which is what a caller experiences.

Replays are identical across engines: the assistant history in each request is the ground-truth click, not the model's own
answer, so two engines see byte-identical request bodies.
"""
from __future__ import annotations

import asyncio
import json
import re
import statistics
import subprocess
import threading
import time
from pathlib import Path

import aiohttp

from .prompts import build_messages
from .stats import percentile, wilson

try:  # scoring reuses the training reward's parser; optional so the client also runs without the repo on the path
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from uiclick.reward import in_box, parse_click
except Exception:  # pragma: no cover
    parse_click = in_box = None


class GpuMonitor:
    """Samples nvidia-smi memory/utilisation in a background thread."""

    def __init__(self, interval: float = 0.5, gpu_index: int | None = None):
        self.interval, self.gpu_index, self.samples, self._stop = interval, gpu_index, [], threading.Event()
        self._t = threading.Thread(target=self._loop, daemon=True)

    def _loop(self):
        cmd = ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"]
        if self.gpu_index is not None:
            cmd += ["-i", str(self.gpu_index)]
        while not self._stop.is_set():
            try:
                out = subprocess.run(cmd, capture_output=True, text=True, timeout=5).stdout.strip().splitlines()
                if out:
                    mem, util = (float(x) for x in out[0].split(","))
                    self.samples.append((time.time(), mem, util))
            except Exception:
                pass
            self._stop.wait(self.interval)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join(timeout=5)

    def summary(self) -> dict:
        if not self.samples:
            return {"available": False}
        mem = [s[1] for s in self.samples]
        util = [s[2] for s in self.samples]
        return {"available": True, "n": len(mem), "mem_used_mib_max": max(mem), "mem_used_mib_median": statistics.median(mem),
                "gpu_util_pct_median": statistics.median(util), "gpu_util_pct_max": max(util)}


_METRIC_RE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+([-+0-9.eE]+|NaN|\+Inf|-Inf)\s*$")


async def scrape_metrics(sess: aiohttp.ClientSession, base_url: str) -> dict | None:
    """Prometheus text -> {name{labels}: value}, keeping only cache / queue related series. None if no /metrics."""
    try:
        async with sess.get(f"{base_url}/metrics", timeout=aiohttp.ClientTimeout(total=5)) as r:
            if r.status != 200:
                return None
            text = await r.text()
    except Exception:
        return None
    out = {}
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        m = _METRIC_RE.match(line.strip())
        if m and re.search(r"cache|prefix|num_requests|queue", m.group(1), re.I):
            try:
                out[m.group(1) + (m.group(2) or "")] = float(m.group(3))
            except ValueError:
                pass
    return out


def prefix_hit_rate(before: dict | None, after: dict | None) -> dict | None:
    """Engine-reported prefix-cache hit rate over the run, from counters (vLLM exposes hits and queries). None if absent."""
    if not before or not after:
        return None

    def total(d, pat):
        return sum(v for k, v in d.items() if re.search(pat, k.split("{")[0]) and not k.split("{")[0].endswith("_created"))

    dh = total(after, r"prefix_cache_hits") - total(before, r"prefix_cache_hits")
    dq = total(after, r"prefix_cache_queries") - total(before, r"prefix_cache_queries")
    return {"hits": dh, "queries": dq, "rate": dh / dq} if dq > 0 else None


async def _request(sess, url, payload, timeout) -> dict:
    t0 = time.perf_counter()
    rec = {"status": None, "error": None, "ttft": None, "itl": [], "e2e": None, "text": "", "usage": None}
    last = None
    try:
        async with sess.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=timeout)) as r:
            rec["status"] = r.status
            if r.status != 200:
                rec["error"] = (await r.text())[:300]
                return rec
            async for raw in r.content:
                line = raw.decode("utf-8", "ignore").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    continue          # keep reading to EOF: a half-read response leaves the pooled connection in an unknown state
                obj = json.loads(data)
                if obj.get("usage"):
                    rec["usage"] = obj["usage"]
                for ch in obj.get("choices", []):
                    piece = (ch.get("delta") or {}).get("content")
                    if piece:
                        now = time.perf_counter()
                        if rec["ttft"] is None:
                            rec["ttft"] = now - t0
                        else:
                            rec["itl"].append(now - last)
                        last = now
                        rec["text"] += piece
    except Exception as e:  # timeouts, resets, bad JSON: recorded, never raised, so one failure cannot end a run
        rec["error"] = f"{type(e).__name__}: {e}"[:300]
    rec["e2e"] = time.perf_counter() - t0
    return rec


async def run_load(base_url: str, model: str, trace_dir: str | Path, sessions: list[dict], concurrency: int,
                   max_tokens: int = 24, timeout: float = 180.0, think_time: float = 0.0,
                   gpu_index: int | None = None) -> dict:
    """Play ``sessions`` with ``concurrency`` parallel agents. Returns raw per-request records + summary."""
    base_url = base_url.rstrip("/")
    root = Path(trace_dir)
    queue: asyncio.Queue = asyncio.Queue()
    for s in sessions:
        queue.put_nowait(s)
    records: list[dict] = []
    conn = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(connector=conn) as sess:
        before = await scrape_metrics(sess, base_url)
        t_start = time.perf_counter()

        async def worker():
            while True:
                try:
                    s = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                for t, turn in enumerate(s["turns"]):
                    payload = {"model": model, "messages": build_messages(s, t, root), "max_tokens": max_tokens,
                               "temperature": 0, "stream": True, "stream_options": {"include_usage": True}}
                    ts = time.perf_counter() - t_start
                    r = await _request(sess, f"{base_url}/v1/chat/completions", payload, timeout)
                    r.update(session=s["session_id"], turn=t, prefix_id=s["prefix_id"], shared=s["shared"], t_start=ts,
                             box=turn["box"])
                    records.append(r)
                    if think_time:
                        await asyncio.sleep(think_time)

        with GpuMonitor(gpu_index=gpu_index) as gpu:
            await asyncio.gather(*(worker() for _ in range(concurrency)))
            wall = time.perf_counter() - t_start
        after = await scrape_metrics(sess, base_url)
    return {"records": records, "wall_s": wall, "gpu": gpu.summary(), "metrics_before": before, "metrics_after": after,
            "engine_prefix_hit_rate": prefix_hit_rate(before, after), "summary": summarize(records, wall)}


def _usage(r: dict, key: str):
    u = r.get("usage") or {}
    if key == "cached":
        return (u.get("prompt_tokens_details") or {}).get("cached_tokens")
    return u.get(key)


def summarize(records: list[dict], wall: float) -> dict:
    ok = [r for r in records if r["status"] == 200 and not r["error"] and r["ttft"] is not None]
    n = len(records)
    ttft = [r["ttft"] for r in ok]
    e2e = [r["e2e"] for r in ok]
    itl = [g for r in ok for g in r["itl"]]
    comp = [_usage(r, "completion_tokens") for r in ok]
    prompt = [_usage(r, "prompt_tokens") for r in ok]
    cached = [_usage(r, "cached") for r in ok]
    out = {"requests": n, "ok": len(ok), "errors": n - len(ok), "error_rate": (n - len(ok)) / n if n else None,
           "wall_s": wall, "req_per_s": len(ok) / wall if wall else None}
    for name, xs in (("ttft", ttft), ("e2e", e2e), ("itl", itl)):
        out[name] = {"p50": percentile(xs, 50), "p90": percentile(xs, 90), "p99": percentile(xs, 99),
                     "mean": statistics.fmean(xs) if xs else None, "n": len(xs)}
    if all(c is not None for c in comp) and ok:
        out["completion_tokens_per_s"] = sum(comp) / wall
        out["completion_tokens_mean"] = statistics.fmean(comp)
    if all(p is not None for p in prompt) and ok:
        out["prompt_tokens_per_s"] = sum(prompt) / wall
        out["prompt_tokens_mean"] = statistics.fmean(prompt)
        if all(c is not None for c in cached):
            out["cached_token_fraction"] = sum(cached) / sum(prompt)   # as reported by the engine in `usage`
    if parse_click:
        hits = [parse_click(r["text"]) is not None and in_box(parse_click(r["text"]), r["box"]) for r in ok]
        k = sum(hits)
        out["trace_click_accuracy"] = {"hits": k, "n": len(ok), "acc": k / len(ok) if ok else None,
                                       "ci": list(wilson(k, len(ok))),
                                       "valid_format": sum(parse_click(r["text"]) is not None for r in ok) / len(ok) if ok else None}
    return out
