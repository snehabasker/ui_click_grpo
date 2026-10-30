"""The prefill/decode proxy against fake instances (no model, no GPU): the order of the two steps, what each instance is sent,
streaming fidelity, error handling, and the KV-store wipe. Fake timings prove nothing about an engine."""
import asyncio
import json

from aiohttp import ClientSession, web
from aiohttp.test_utils import TestServer

from servebench import disagg_proxy as P

BODY = {"model": "m", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 24, "stream": True,
        "stream_options": {"include_usage": True}}


def fake_instance(name, log, status=200):
    async def chat(request):
        body = await request.json()
        log.append((name, body))
        if status != 200:
            return web.Response(status=status, text="boom")
        if not body.get("stream"):
            return web.json_response({"choices": [{"message": {"content": "(1, 2)"}}], "usage": {"completion_tokens": 1}})
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)
        for part in ("a", "b", "c"):
            await resp.write(f"data: {json.dumps({'choices': [{'delta': {'content': part}}]})}\n\n".encode())
            await asyncio.sleep(0.01)
        await resp.write(b"data: [DONE]\n\n")
        return resp

    async def reset(request):
        log.append((name, request.path))
        return web.Response(text="ok")

    app = web.Application()
    app.router.add_post("/v1/chat/completions", chat)
    app.router.add_post("/reset_prefix_cache", reset)
    app.router.add_post("/reset_mm_cache", reset)
    app.router.add_get("/v1/models", lambda r: web.json_response({"data": [{"id": name}]}))
    return app


async def with_proxy(fn, prefill_status=200, decode_status=200, store=None):
    log = []
    pre, dec = TestServer(fake_instance("prefill", log, prefill_status)), TestServer(fake_instance("decode", log, decode_status))
    await pre.start_server()
    await dec.start_server()
    proxy = TestServer(P.make_app(f"http://127.0.0.1:{pre.port}", f"http://127.0.0.1:{dec.port}", store))
    await proxy.start_server()
    try:
        async with ClientSession() as s:
            return await fn(s, f"http://127.0.0.1:{proxy.port}", log, proxy.app["state"])
    finally:
        for srv in (proxy, pre, dec):
            await srv.close()


def test_prefill_runs_first_with_one_token_then_decode_gets_the_original_request():
    async def go(s, base, log, state):
        async with s.post(f"{base}/v1/chat/completions", json=BODY) as r:
            assert r.status == 200
            text = (await r.read()).decode()
        return text, log, state

    text, log, state = asyncio.run(with_proxy(go))
    assert [n for n, _ in log] == ["prefill", "decode"]
    pre, dec = log[0][1], log[1][1]
    assert pre["max_tokens"] == 1 and pre["stream"] is False and "stream_options" not in pre
    assert pre["messages"] == BODY["messages"]
    assert dec == BODY                                            # the decode instance sees the client's request untouched
    assert text.count("data:") == 4 and "[DONE]" in text         # streamed through, chunks intact
    assert len(state["prefill_s"]) == 1


def test_non_streaming_requests_return_the_decode_body():
    async def go(s, base, log, state):
        async with s.post(f"{base}/v1/chat/completions", json=dict(BODY, stream=False)) as r:
            return r.status, await r.json()

    status, body = asyncio.run(with_proxy(go))
    assert status == 200 and body["choices"][0]["message"]["content"] == "(1, 2)"


def test_a_failing_prefill_instance_is_a_502_and_decode_is_never_called():
    async def go(s, base, log, state):
        async with s.post(f"{base}/v1/chat/completions", json=BODY) as r:
            return r.status, (await r.json())["error"], [n for n, _ in log], state["failures"]

    status, err, called, failures = asyncio.run(with_proxy(go, prefill_status=500))
    assert status == 502 and "prefill" in err and called == ["prefill"] and failures == 1


def test_a_failing_decode_instance_is_a_502():
    async def go(s, base, log, state):
        async with s.post(f"{base}/v1/chat/completions", json=BODY) as r:
            return r.status, (await r.json())["error"]

    status, err = asyncio.run(with_proxy(go, decode_status=503))
    assert status == 502 and "decode" in err


def test_reset_prefix_cache_reaches_both_instances_and_empties_the_store(tmp_path):
    (tmp_path / "abc").mkdir()
    (tmp_path / "abc" / "layer.safetensors").write_bytes(b"x")
    (tmp_path / "loose.bin").write_bytes(b"y")

    async def go(s, base, log, state):
        async with s.post(f"{base}/reset_prefix_cache") as r:
            return r.status, await r.json(), log

    status, body, log = asyncio.run(with_proxy(go, store=tmp_path))
    assert status == 200 and body["prefill"] == 200 and body["decode"] == 200 and body["store_entries_removed"] == 2
    assert sorted(n for n, _ in log) == ["decode", "prefill"]
    assert list(tmp_path.iterdir()) == [] and tmp_path.exists()


def test_reset_mm_cache_does_not_touch_the_store(tmp_path):
    (tmp_path / "keep").mkdir()

    async def go(s, base, log, state):
        async with s.post(f"{base}/reset_mm_cache") as r:
            return r.status

    assert asyncio.run(with_proxy(go, store=tmp_path)) == 200
    assert (tmp_path / "keep").exists()


def test_other_paths_go_to_the_decode_instance():
    async def go(s, base, log, state):
        async with s.get(f"{base}/v1/models") as r:
            return await r.json()

    assert asyncio.run(with_proxy(go))["data"][0]["id"] == "decode"


def test_prefill_body_does_not_mutate_the_original():
    b = dict(BODY, max_completion_tokens=50)
    p = P.prefill_body(b)
    assert b["max_tokens"] == 24 and b["stream"] is True and "max_completion_tokens" in b
    assert p["max_tokens"] == 1 and "max_completion_tokens" not in p
