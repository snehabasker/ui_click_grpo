"""Is the load client's inter-token timing trustworthy?

Serves an SSE stream from a raw asyncio socket with EXACT spacing (busy-wait, no timer granularity) and reads it with
(a) a raw-socket reader that timestamps bytes as they arrive and (b) servebench's aiohttp client. If they agree with each
other and with the true spacing, the client's ITL numbers can be trusted at that spacing. Also records this OS's
asyncio.sleep granularity, which is why the mock server cannot be used to validate small intervals.

    python serving/scripts/validate_client_timing.py        -> results/raw/client_timing_<ts>.json
"""
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "serving"))

import aiohttp

from servebench import envinfo
from servebench.client import _request

N_TOKENS, REPEATS = 10, 8


async def handle(gap):
    async def _h(reader, writer):
        head = await reader.readuntil(b"\r\n\r\n")
        n = [int(l.split(b":")[1]) for l in head.split(b"\r\n") if l.lower().startswith(b"content-length")]
        if n:
            await reader.readexactly(n[0])          # drain the request body, or closing the socket resets the connection
        writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n"
                     b"Connection: close\r\n\r\n")
        await writer.drain()
        for i in range(N_TOKENS):
            data = b"data: " + json.dumps({"choices": [{"delta": {"content": str(i)}}]}).encode() + b"\n\n"
            writer.write(b"%x\r\n" % len(data) + data + b"\r\n")
            await writer.drain()
            end = time.perf_counter() + gap
            while time.perf_counter() < end:
                await asyncio.sleep(0)
        tail = b"data: [DONE]\n\n"
        writer.write(b"%x\r\n" % len(tail) + tail + b"\r\n0\r\n\r\n")
        await writer.drain()
        writer.close()
    return _h


async def raw_gaps(port):
    r, w = await asyncio.open_connection("127.0.0.1", port)
    w.write(b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Length: 2\r\n\r\n{}")
    await w.drain()
    stamps, buf = [], b""
    while b"[DONE]" not in buf:
        chunk = await r.read(65536)
        if not chunk:
            break
        stamps += [time.perf_counter()] * chunk.count(b"data: {")
        buf += chunk
    w.close()
    return [b - a for a, b in zip(stamps, stamps[1:])]


async def measure(gap):
    srv = await asyncio.start_server(await handle(gap), "127.0.0.1", 0)
    port = srv.sockets[0].getsockname()[1]
    raw = []
    for _ in range(REPEATS):
        raw += await raw_gaps(port)
    mine, errors, err_text = [], 0, []
    async with aiohttp.ClientSession() as s:
        for _ in range(REPEATS):
            r = await _request(s, f"http://127.0.0.1:{port}/v1/chat/completions", {"stream": True}, 10)
            errors += bool(r["error"])
            err_text += [r["error"]] if r["error"] else []
            mine += r["itl"]
    srv.close()
    ms = lambda xs: statistics.median(xs) * 1000
    return {"true_spacing_ms": gap * 1000, "raw_socket_median_ms": ms(raw), "raw_n": len(raw),
            "servebench_client_median_ms": ms(mine), "client_n": len(mine), "client_request_errors": errors, "client_error_messages": sorted(set(err_text)), "client_error_messages": sorted(set(err_text)),
            "client_minus_raw_ms": ms(mine) - ms(raw)}


async def sleep_granularity(d, n=50):
    xs = []
    for _ in range(n):
        t = time.perf_counter()
        await asyncio.sleep(d)
        xs.append(time.perf_counter() - t)
    return {"requested_ms": d * 1000, "actual_median_ms": statistics.median(xs) * 1000}


async def main():
    return {"client_vs_exact_spacing": [await measure(g) for g in (0.010, 0.020, 0.040)],
            "asyncio_sleep_granularity": [await sleep_granularity(d) for d in (0.003, 0.010, 0.030)]}


res = asyncio.run(main())
path = ROOT / "serving" / "results" / "raw" / f"client_timing_{time.strftime('%Y%m%d-%H%M%S')}.json"
path.write_text(json.dumps({"kind": "client_timing", "env": envinfo.capture(), "results": res}, indent=1))
for r in res["client_vs_exact_spacing"]:
    print(f"true {r['true_spacing_ms']:.0f} ms | raw {r['raw_socket_median_ms']:.2f} ms (n={r['raw_n']}) | client {r['servebench_client_median_ms']:.2f} ms "
          f"(n={r['client_n']}, request errors {r['client_request_errors']})")
for r in res["asyncio_sleep_granularity"]:
    print(f"asyncio.sleep({r['requested_ms']:.0f} ms) -> {r['actual_median_ms']:.2f} ms")
print("wrote", path.name)
