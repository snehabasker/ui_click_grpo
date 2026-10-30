import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import summarize_disagg as S


def lt(ttft, conc):
    s = {"ttft": {"p50": ttft, "p99": ttft * 2}, "itl": {"p50": 0.015}, "e2e": {"p50": ttft + 0.2, "p99": ttft * 3}, "req_per_s": 10 / ttft,
         "cached_token_fraction": 0.5, "error_rate": 0.0}
    return {"tag": "t", "concurrency": conc, "repeat": 1, "summary": s, "gpu": {"temp_c_max": 70, "power_w_median": 60}}


def make(tmp):
    rounds = []
    for rnd, (order, ttfts) in enumerate([(["colocated", "split"], {"colocated": 0.30, "split": 0.50}),
                                          (["split", "colocated"], {"colocated": 0.34, "split": 0.46}),
                                          (["colocated", "split"], {"colocated": 0.32, "split": 0.52})]):
        for setup in order:
            runs = []
            for conc in (1, 4):
                name = f"loadtest_disagg-{setup}_c{conc}_r1_r{rnd}.json"
                (tmp / name).write_text(json.dumps(lt(ttfts[setup] * conc, conc)))
                n_before = 0 if conc == 1 else 10
                prefill = ([0.2] * (n_before + 10)) if setup == "split" else None
                runs.append({"concurrency": conc, "loadtest_files": [name], "kv_store_bytes_after_run": 10 * 2**20 * 10 if setup == "split" else None,
                             "proxy_prefill_s_cumulative": prefill})
            rounds.append({"round": rnd, "order": order, "setup": setup, "runs": runs})
    (tmp / "disagg_bench_20261030-000000.json").write_text(json.dumps(
        {"settings": {"kv_gib_colocated": 0.8, "kv_gib_split_each": 0.4, "connector": "ExampleConnector", "max_seqs": 8, "max_len": 3072},
         "rounds": rounds}))


def test_cells_are_median_min_max_over_rounds_and_orders_are_listed(tmp_path):
    make(tmp_path)
    md = S.tables(tmp_path)
    assert "0.32 (0.30–0.34)" in md and "0.50 (0.46–0.52)" in md          # c=1 TTFT p50, colocated / split
    assert "round 2: split then colocated" in md and "round 1: colocated then split" in md
    assert "**c=1**" in md and "**c=4**" in md


def test_prefill_step_is_sliced_per_run_and_kv_per_request_is_bytes_over_requests(tmp_path):
    make(tmp_path)
    md = S.tables(tmp_path)
    split_c1 = next(l for l in md.splitlines() if l.startswith("| split |"))
    assert "0.200" in split_c1                      # median of the sliced prefill durations
    assert "10.0" in split_c1                       # 100 MiB / 10 requests
    colo = next(l for l in md.splitlines() if l.startswith("| colocated |"))
    assert colo.rstrip().endswith("– | – |")        # no proxy, no KV store
