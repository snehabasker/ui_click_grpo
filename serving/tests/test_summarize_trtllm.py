import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import summarize_trtllm as S


def lt(ttft):
    s = {"ttft": {"p50": ttft, "p99": ttft * 2}, "itl": {"p50": 0.015}, "e2e": {"p50": ttft + 0.2, "p99": ttft * 3}, "req_per_s": 10 / ttft,
         "cached_token_fraction": 0.0, "error_rate": 0.0}
    return {"summary": s, "gpu": {"temp_c_max": 70, "power_w_median": 60}}


def make(tmp):
    rounds = []
    for rnd, order in enumerate([["vllm", "trtllm"], ["trtllm", "vllm"], ["vllm", "trtllm"]]):
        for eng in order:
            name = f"loadtest_trtllm-single-{eng}_c1_r1_{rnd}.json"
            (tmp / name).write_text(json.dumps(lt((0.2 if eng == "vllm" else 0.4) + 0.01 * rnd)))
            rounds.append({"round": rnd, "engine": eng, "order": order, "runs": [{"concurrency": 1, "loadtest_files": [name]}]})
    (tmp / "trtllm_bench_20261030-000000.json").write_text(json.dumps(
        {"settings": {"trace": "serving/traces/single", "prefix_caching": "off on both engines", "max_seqs": 32, "max_len": 4096, "trtllm_backend": "pytorch"},
         "trtllm_env": {"tensorrt_llm": "1.2.1", "torch": "2.9.1+cu130", "transformers": "4.57.3"}, "env": {"packages": {"vllm": "0.32.0"}},
         "trtllm_local_patch_diff": "< flash_attention_2\n> sdpa", "rounds": rounds}))


def test_latency_cells_are_median_min_max_over_rounds_and_the_patch_is_shown(tmp_path):
    make(tmp_path)
    md = S.tables(tmp_path)
    assert "0.21 (0.20–0.22)" in md and "0.41 (0.40–0.42)" in md
    assert "round 2: trtllm then vllm" in md and "> sdpa" in md and "TensorRT-LLM 1.2.1" in md


def test_accuracy_table_is_a_paired_sign_test(tmp_path):
    def acc(tag, hits):
        return {"tag": tag, "set": "synth300", "score": {"n": len(hits), "acc": sum(hits) / len(hits), "per_item_hit": hits}}
    (tmp_path / "accuracy_vllm-prefix-on_synth300_20261026-000000.json").write_text(json.dumps(acc("vllm-prefix-on", [True] * 8 + [False] * 2)))
    (tmp_path / "accuracy_trtllm_synth300_20261030-000000.json").write_text(json.dumps(acc("trtllm", [True] * 6 + [False] * 4)))
    assert "| synth300 | 10 | 80.00% | 60.00% | -20.00 pts | 2 / 0 | 0.5000 |" in S.accuracy_table(tmp_path)


def test_tokens_table_lists_each_check(tmp_path):
    (tmp_path / "prompt_tokens_check_trtllm_screenspot400_20261030-000000.json").write_text(json.dumps(
        {"tag": "trtllm", "set": "screenspot400", "summary": {"n": 20, "equal": 17, "min_diff": 0, "max_diff": 3}}))
    assert "| trtllm | screenspot400 | 20 | 17 | +0 | +3 |" in S.tokens_table(tmp_path)
