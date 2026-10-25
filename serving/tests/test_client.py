"""Client tests against the MOCK server. Mock timings are constants, so none of this measures an engine."""
import asyncio

import pytest
from aiohttp.test_utils import TestServer

from servebench.client import prefix_hit_rate, run_load, summarize
from servebench.mock_server import ANSWER, make_app
from servebench.traces import TraceConfig, generate, load_trace
from uiclick.reward import in_box, parse_click


@pytest.fixture(scope="module")
def small_trace(tmp_path_factory):
    out = tmp_path_factory.mktemp("clienttrace")
    generate(TraceConfig(name="c", n_sessions=6, shared_fraction=1.0, n_shared_groups=1, min_turns=3, max_turns=4,
                         warmup_sessions=1, seed=2), out)
    return out, load_trace(out)


def _play(app_kwargs, trace_dir, sessions, concurrency=2):
    async def go():
        server = TestServer(make_app(**app_kwargs))
        await server.start_server()
        try:
            return await run_load(f"http://127.0.0.1:{server.port}", "mock", trace_dir, sessions, concurrency), server.app["state"]
        finally:
            await server.close()
    return asyncio.run(go())


def test_client_measures_latency_tokens_and_accuracy(small_trace):
    out, sessions = small_trace
    res, state = _play(dict(ttft=0.05, itl=0.03), out, sessions)
    s = res["summary"]
    n = sum(len(x["turns"]) for x in sessions)
    assert s["requests"] == s["ok"] == n and s["errors"] == 0
    assert 0.045 <= s["ttft"]["p50"] < 0.5
    # itl is kept >= 30 ms: below ~20 ms the mock server itself (Windows timers, ~10 ms granularity) bunches tokens. The client was
    # validated separately against an exact-spacing raw-socket server (see NOTES.md).
    assert 0.025 <= s["itl"]["p50"] < 0.2
    assert s["completion_tokens_mean"] == len(ANSWER)
    expect = sum(in_box(parse_click(ANSWER), t["box"]) for x in sessions for t in x["turns"])
    assert s["trace_click_accuracy"]["hits"] == expect and s["trace_click_accuracy"]["n"] == n
    assert s["req_per_s"] > 0


def test_cached_tokens_and_engine_metrics(small_trace):
    out, sessions = small_trace
    res, state = _play(dict(ttft=0.01, itl=0.001), out, sessions, concurrency=1)
    s = res["summary"]
    # The mock only calls it a hit when the whole prefix (incl. growing per-session history) was seen before, so exactly the
    # first turn of every session after the first can hit. (Real engines match the longest common prefix; this is a plumbing test.)
    assert state["hits"] == len(sessions) - 1
    assert 0 < s["cached_token_fraction"] < 0.8
    hr = res["engine_prefix_hit_rate"]
    assert hr["queries"] == state["queries"] and hr["hits"] == state["hits"]
    assert hr["rate"] == pytest.approx(state["hits"] / state["queries"])


def test_failures_are_recorded_not_raised(small_trace):
    out, sessions = small_trace
    res, _ = _play(dict(ttft=0.01, itl=0.001, fail_after=5), out, sessions)
    s = res["summary"]
    assert s["errors"] > 0 and s["ok"] == 5 and 0 < s["error_rate"] < 1
    assert all(r["status"] == 503 for r in res["records"] if r["error"])


def test_unreachable_server_yields_errors_not_a_crash(small_trace):
    out, sessions = small_trace

    async def go():
        return await run_load("http://127.0.0.1:9", "mock", out, sessions[:1], 1, timeout=2)
    res = asyncio.run(go())
    assert res["summary"]["ok"] == 0 and res["summary"]["errors"] == res["summary"]["requests"] > 0


def test_prefix_hit_rate_edge_cases():
    assert prefix_hit_rate(None, {}) is None
    b = {"vllm:prefix_cache_queries_total": 10.0, "vllm:prefix_cache_hits_total": 4.0}
    a = {"vllm:prefix_cache_queries_total": 30.0, "vllm:prefix_cache_hits_total": 14.0}
    assert prefix_hit_rate(b, a) == {"hits": 10.0, "queries": 20.0, "rate": 0.5}
    assert prefix_hit_rate(b, b) is None            # no queries in the window


def test_summarize_handles_empty():
    s = summarize([], 1.0)
    assert s["requests"] == 0 and s["ttft"]["p50"] is None
