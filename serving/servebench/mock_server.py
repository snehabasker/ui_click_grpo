"""A fake OpenAI-compatible streaming server used ONLY to test the client and the router.

It does no model inference. Its timings are configured constants, so nothing measured against it is ever a result.
It does simulate a prefix cache (by hashing the system prompt and few-shot turns) so that the client's cached-token and
/metrics logic can be exercised.
"""
from __future__ import annotations

import asyncio
import hashlib
import json

from aiohttp import web

ANSWER = "(500, 500)"


def _prefix_key(messages: list[dict]) -> str:
    """Everything up to (not including) the final user turn, with image data replaced by its length."""
    parts = []
    for m in messages[:-1]:
        c = m["content"]
        parts.append(m["role"] + ":" + (c if isinstance(c, str) else json.dumps(
            [p if p.get("type") != "image_url" else {"img": len(p["image_url"]["url"])} for p in c])))
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()


def make_app(ttft: float = 0.05, itl: float = 0.005, fail_after: int | None = None, name: str = "mock") -> web.Application:
    state = {"seen": set(), "hits": 0, "queries": 0, "n": 0, "inflight": 0}

    async def chat(request: web.Request):
        state["n"] += 1
        if fail_after is not None and state["n"] > fail_after:
            return web.Response(status=503, text="mock replica failure")
        body = await request.json()
        key = _prefix_key(body["messages"])
        hit = key in state["seen"]
        state["seen"].add(key)
        state["queries"] += 1
        state["hits"] += int(hit)
        prompt_tokens = sum(len(json.dumps(m["content"])) for m in body["messages"]) // 8
        cached = int(prompt_tokens * 0.8) if hit else 0
        state["inflight"] += 1
        try:
            await asyncio.sleep(ttft)
            if not body.get("stream"):
                return web.json_response({"choices": [{"message": {"role": "assistant", "content": ANSWER}}],
                                          "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": len(ANSWER)}})
            resp = web.StreamResponse(headers={"Content-Type": "text/event-stream", "X-Mock-Replica": name})
            await resp.prepare(request)
            for i, ch in enumerate(ANSWER):
                if i:
                    await asyncio.sleep(itl)
                await resp.write(f"data: {json.dumps({'choices': [{'delta': {'content': ch}}]})}\n\n".encode())
            usage = {"prompt_tokens": prompt_tokens, "completion_tokens": len(ANSWER),
                     "prompt_tokens_details": {"cached_tokens": cached}}
            await resp.write(f"data: {json.dumps({'choices': [], 'usage': usage})}\n\n".encode())
            await resp.write(b"data: [DONE]\n\n")
            return resp
        finally:
            state["inflight"] -= 1

    async def metrics(_):
        return web.Response(text=(f"# HELP vllm:prefix_cache_queries_total mock\nvllm:prefix_cache_queries_total {state['queries']}\n"
                                  f"vllm:prefix_cache_hits_total {state['hits']}\nvllm:num_requests_running {state['inflight']}\n"))

    async def reset(_):
        state["seen"].clear()
        return web.Response(text="ok")

    app = web.Application()
    app["state"] = state
    app.add_routes([web.post("/v1/chat/completions", chat), web.get("/metrics", metrics), web.get("/health", lambda r: web.Response(text="ok")),
                    web.post("/reset_prefix_cache", reset)])
    return app


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--ttft", type=float, default=0.05)
    ap.add_argument("--itl", type=float, default=0.005)
    ap.add_argument("--duration", type=float, default=None, help="exit by itself after this many seconds")
    a = ap.parse_args()

    async def main():
        runner = web.AppRunner(make_app(a.ttft, a.itl))
        await runner.setup()
        await web.TCPSite(runner, "127.0.0.1", a.port).start()
        await asyncio.sleep(a.duration if a.duration else 10**9)
        await runner.cleanup()

    asyncio.run(main())
