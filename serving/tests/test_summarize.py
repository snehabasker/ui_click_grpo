import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import summarize


def fake_run(tag, conc, rep, ttft, mock=False, name=None):
    s = {"ttft": {"p50": ttft, "p99": ttft * 2}, "itl": {"p50": 0.015}, "e2e": {"p50": ttft + 0.2, "p99": ttft * 3},
         "req_per_s": 10 / ttft, "cached_token_fraction": 0.6, "error_rate": 0.0,
         "trace_click_accuracy": {"acc": 0.65}}
    return {"tag": tag, "concurrency": conc, "repeat": rep, "mock": mock, "summary": s, "gpu": {"mem_used_mib_max": 7000.0}}


def write(tmp, runs):
    for i, (fname, r) in enumerate(runs):
        (tmp / fname).write_text(json.dumps(r))


def test_median_and_range_over_repeats(tmp_path):
    write(tmp_path, [(f"loadtest_a_c4_r{i}.json", fake_run("a", 4, i, t)) for i, t in enumerate([1.0, 3.0, 2.0])])
    agg = summarize.aggregate(summarize.load_runs(tmp_path))
    m = agg[("a", 4)]["ttft_p50"]
    assert (m["median"], m["min"], m["max"], m["repeats"]) == (2.0, 1.0, 3.0, 3)
    assert "2.00 (1.00–3.00)" in summarize.markdown(agg)


def test_mock_and_explore_files_are_never_aggregated(tmp_path):
    write(tmp_path, [("MOCK_loadtest_a_c1_r1.json", fake_run("m", 1, 1, 0.1, mock=True)),
                     ("loadtest_explore-x_c1_r1.json", fake_run("explore-x", 1, 1, 0.1)),
                     ("loadtest_flagged_c1_r1.json", fake_run("flagged", 1, 1, 0.1, mock=True)),
                     ("loadtest_real_c1_r1.json", fake_run("real", 1, 1, 0.5))])
    runs = summarize.load_runs(tmp_path)
    assert [r["tag"] for r in runs] == ["real"]


def test_units_and_single_repeat_format(tmp_path):
    write(tmp_path, [("loadtest_a_c1_r1.json", fake_run("a", 1, 1, 0.5))])
    md = summarize.markdown(summarize.aggregate(summarize.load_runs(tmp_path)))
    assert "15.0" in md            # itl 0.015 s shown in ms
    assert "60%" in md and "65.0%" in md
    assert "(0.50–" not in md      # a single repeat has no range


def test_sglang_serve_script_uses_the_unambiguous_graph_flag():
    """SGLang 0.5.22 refuses to start with plain --cuda-graph-max-bs (ambiguous between -decode and -prefill): the first
    follow-up attempt failed for exactly this reason."""
    import re
    text = (Path(__file__).resolve().parents[1] / "scripts" / "wsl" / "serve_sglang.sh").read_text()
    assert "--cuda-graph-max-bs-decode" in text
    assert not re.search(r"--cuda-graph-max-bs(?![-\w])", text.replace("# SGLang 0.5.22 renamed this flag: plain --cuda-graph-max-bs is now", ""))


def test_sglang_serve_script_points_every_compiler_variable_at_gcc15():
    """FlashInfer's JIT overrides nvcc's -ccbin with $CXX; CUDA 13.2 rejects GCC > 15, so CC, CXX and CUDAHOSTCXX must all name the
    GCC 15 that ships with the toolkit."""
    text = (Path(__file__).resolve().parents[1] / "scripts" / "wsl" / "serve_sglang.sh").read_text()
    for var in ("CC", "CXX", "CUDAHOSTCXX"):
        assert f"export {var}=" in text
    assert "x86_64-conda-linux-gnu-g++" in text and "x86_64-conda-linux-gnu-gcc" in text
