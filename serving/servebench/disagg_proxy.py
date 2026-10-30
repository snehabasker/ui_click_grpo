"""A two-step proxy that makes a prefill instance and a decode instance look like one OpenAI-compatible server.

    python -m servebench.disagg_proxy --prefill http://127.0.0.1:8001 --decode http://127.0.0.1:8002 --port 9100 --store ~/work/kvstore

For every /v1/chat/completions request the proxy (1) replays the request on the PREFILL instance with ``max_tokens=1`` and no
streaming, which makes that instance compute the prompt's KV cache and hand it to its KV connector, then (2) sends the original request
to the DECODE instance, whose connector loads that KV instead of recomputing the prompt, and streams its answer back unchanged.
A client that measures time to first token and inter-token latency therefore measures the whole two-step path.

``POST /reset_prefix_cache`` and ``/reset_mm_cache`` are forwarded to both instances and also empty the KV store directory. The store
has to be emptied between repeats: a producer that finds KV for a prompt on disk loads it instead of computing it, which would turn a
"cold" repeat into a warm one. ``GET /proxy/stats`` returns the prefill-step durations seen so far. Everything else goes to the decode
instance.
"""
from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import aiohttp
from aiohttp import web

HOP = {"transfer-encoding", "connection", "content-length", "keep-alive"}


def prefill_body(body: dict) -> dict:
    """The request as the prefill instance should see it: same prompt, one generated token, no streaming."""
    b = dict(body, max_tokens=1, stream=False)
    for k in ("stream_options", "max_completion_tokens"):
        b.pop(k, None)
    return b


def wipe(store: Path | None) -> int:
    """Delete everything inside the KV store directory (not the directory itself); returns how many entries were removed."""
    if store is None or not store.exists():
        return 0
    n = 0
    for child in store.iterdir():
        shutil.rmtree(child) if child.is_dir() else child.unlink()
        n += 1
    return n


def make_app(prefill_url: str, decode_url: str, store: Path | None = None) -> web.Application:
    state = {"prefill_s": [], "failures": 0}
    app = web.Application(client_max_size=64 * 2**20)

    async def session(app):
        app["http"] = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None, sock_read=600),
                                            connector=aiohttp.TCPConnector(limit=0))
        yield
        await app["http"].close()

    app.cleanup_ctx.append(session)

    async def chat(request: web.Request):
        body = await request.json()
        http: aiohttp.ClientSession = request.app["http"]
        t0 = time.perf_counter()
        async with http.post(f"{prefill_url}/v1/chat/completions", json=prefill_body(body)) as r:
            if r.status != 200:
                state["failures"] += 1
                return web.json_response({"error": "prefill instance failed", "status": r.status, "body": (await r.text())[:500]}, status=502)
            await r.read()
        state["prefill_s"].append(time.perf_counter() - t0)
        async with http.post(f"{decode_url}/v1/chat/completions", json=body) as r:
            if r.status != 200:
                state["failures"] += 1
                return web.json_response({"error": "decode instance failed", "status": r.status, "body": (await r.text())[:500]}, status=502)
            headers = {k: v for k, v in r.headers.items() if k.lower() not in HOP}
            if not body.get("stream"):
                return web.Response(body=await r.read(), status=200, headers=headers)
            resp = web.StreamResponse(status=200, headers=headers)
            await resp.prepare(request)
            async for chunk in r.content.iter_any():
                await resp.write(chunk)
            await resp.write_eof()
            return resp

    async def reset(request: web.Request):
        http: aiohttp.ClientSession = request.app["http"]
        out = {}
        for name, url in (("prefill", prefill_url), ("decode", decode_url)):
            try:
                async with http.post(f"{url}{request.path}") as r:
                    out[name] = r.status
            except aiohttp.ClientError as e:
                out[name] = f"{type(e).__name__}"
        if request.path == "/reset_prefix_cache":
            out["store_entries_removed"] = wipe(store)
        ok = all(v == 200 for k, v in out.items() if k in ("prefill", "decode"))
        return web.json_response(out, status=200 if ok else 502)

    async def stats(request: web.Request):
        return web.json_response({"prefill_s": state["prefill_s"], "failures": state["failures"]})

    async def passthrough(request: web.Request):
        http: aiohttp.ClientSession = request.app["http"]
        async with http.request(request.method, f"{decode_url}{request.path_qs}", data=await request.read()) as r:
            return web.Response(body=await r.read(), status=r.status, content_type=r.content_type)

    app.router.add_post("/v1/chat/completions", chat)
    app.router.add_post("/reset_prefix_cache", reset)
    app.router.add_post("/reset_mm_cache", reset)
    app.router.add_get("/proxy/stats", stats)
    app.router.add_route("*", "/{tail:.*}", passthrough)
    app["state"] = state
    return app


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefill", required=True)
    ap.add_argument("--decode", required=True)
    ap.add_argument("--port", type=int, default=9100)
    ap.add_argument("--store", default=None, help="the KV connector's shared storage directory (emptied on /reset_prefix_cache)")
    a = ap.parse_args()
    web.run_app(make_app(a.prefill.rstrip("/"), a.decode.rstrip("/"), Path(a.store).expanduser() if a.store else None),
                host="127.0.0.1", port=a.port, print=None)
