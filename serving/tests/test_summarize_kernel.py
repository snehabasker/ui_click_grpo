import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import summarize_kernel as S


def fake_bench(reps=3, fused_ms=(0.010, 0.012, 0.011), eager_ms=0.30, comp_ms=0.020):
    runs = []
    for r in range(reps):
        for impl, ms in (("triton_fused", fused_ms[r]), ("torch_eager", eager_ms), ("torch_compile", comp_ms)):
            runs.append({"rep": r, "dtype": "fp16", "hidden": 1536, "rows": 1024, "impl": impl, "ms": ms, "ms_cudagraph": ms / 2,
                         "gbps": 4.7e6 / (ms * 1e-3) / 1e9, "gbps_cudagraph": 4.7e6 / (ms / 2 * 1e-3) / 1e9, "min_bytes": 4700000,
                         "cudagraph_error": None})
        runs.append({"rep": r, "ceilings": {"copy_256MiB_GBps": 300.0 + r, "read_only_256MiB_GBps": 310.0},
                     "gpu_during_rep": {"samples": 10, "power_w_median": 60.0, "power_w_max": 70.0, "sm_mhz_median": 2400.0,
                                        "sm_mhz_min": 2100.0, "temp_c_max": 70.0, "throttle_reasons_seen": ["0x0000000000000000"]}})
    return {"kind": "kernel_bench",
            "env": {"gpu": {"gpus": [{"name": "GPU", "driver": "1"}]}, "torch": "2.0", "triton": "3.0", "cuda_runtime": "13",
                    "repo_commit": "abcdef123", "device_props": {"l2_cache_bytes": 32 * 2**20, "mem_clock_rate": 12001000, "mem_bus_width": 128}},
            "correctness": {"fp16/1536/1024": {"max_code_diff": 1, "frac_codes_differing": 1e-6, "max_rel_scale_err": 2e-7,
                                               "compiled_vs_eager_max_code_diff": 1}},
            "runs": runs}


def test_cells_are_median_min_max_and_ratios_are_ratios_of_medians(tmp_path):
    (tmp_path / "kernel_bench_x_20261027-000000.json").write_text(json.dumps(fake_bench()))
    md = S.tables(tmp_path)
    assert "0.0110 (0.0100–0.0120)" in md                 # fused median (min-max)
    assert "27.3x" in md                                  # 0.30 / 0.011
    assert "1.82x" in md                                  # 0.020 / 0.011
    assert "301 (300–302)" in md                          # copy ceiling


def test_percent_of_copy_uses_the_measured_ceiling(tmp_path):
    (tmp_path / "kernel_bench_x_20261027-000000.json").write_text(json.dumps(fake_bench()))
    md = S.tables(tmp_path)
    fused_gbps = 4.7e6 / (0.011e-3) / 1e9               # 427.3 GB/s, median repeat
    assert f"{100 * fused_gbps / 301:.0f}%" in md


def test_spec_peak_from_clocks_is_clock_times_bus_times_two(tmp_path):
    d = fake_bench()
    assert abs(S.spec_peak_from_clocks(d) - 12001e6 * 2 * 16 / 1e9) < 1e-6
    d["env"]["device_props"] = {}
    assert S.spec_peak_from_clocks(d) is None


def test_csv_rows_cover_every_metric(tmp_path):
    (tmp_path / "kernel_bench_x_20261027-000000.json").write_text(json.dumps(fake_bench()))
    rows = list(S.csv_rows(tmp_path))
    assert {r[5] for r in rows} == {"ms", "ms_cudagraph", "gbps", "gbps_cudagraph"}
    assert {r[4] for r in rows} == set(S.IMPLS) and all(r[9] == 3 for r in rows)


def test_roofline_table_only_has_large_cells_and_compares_the_best_repeat_with_the_best_ceiling(tmp_path):
    d = fake_bench()
    for r in d["runs"]:
        if "impl" in r:
            r["min_bytes"] = 100 * 2**20 if r["rows"] == 1024 else r["min_bytes"]
    (tmp_path / "kernel_bench_x_20261027-000000.json").write_text(json.dumps(d))
    md = S.tables(tmp_path)
    best = max(4700000 / (ms * 1e-3) / 1e9 for ms in (0.010, 0.012, 0.011))     # the fastest repeat
    sec = md.split("Roofline at the cells")[1].split("**fp16")[0]
    assert "| fp16 | 1536 | 1024 | 100 |" in sec
    assert f"{100 * best / 302:.0f}%" in sec                                      # best copy ceiling is 302


def test_accuracy_section_reports_delta_pass_and_noise_reference(tmp_path):
    def cond(acc):
        return {"acc": acc, "ci": [acc - 0.05, acc + 0.05], "valid_format": 1.0}
    d = {"what": "fake quantisation", "criteria": {"max_abs_delta_acc": 0.02, "sign_test_alpha": 0.05},
         "env": {"packages": {"torch": "2.13", "transformers": "5.19"}, "repo_commit": "abcdef123"},
         "results": {"synth300": {"n": 300, "baseline": cond(0.64), "kernel": cond(0.6467), "reference": cond(0.63),
                                  "kernel_vs_baseline": {"delta_acc": 0.0067, "only_baseline_correct": 5, "only_kernel_correct": 7,
                                                         "sign_test_p": 0.7744, "identical_output_text": 0.4267, "pass": True},
                                  "kernel_vs_reference_identical_output_text": 0.4333}},
         "activation_error": {"items": 16, "n_norms": 57, "relative_l2_error_mean": 0.0262, "relative_l2_error_max": 0.0547,
                              "worst_norm": "layers.27.post_attention_layernorm", "max_abs_error_any_norm": 1.0}}
    (tmp_path / "kernel_accuracy_20261027-000000.json").write_text(json.dumps(d))
    md = S.accuracy_section(tmp_path)
    assert "| synth300 | 300 | 64.00% (59.00%–69.00%) | 64.67% (59.67%–69.67%) | 63.00% | +0.7 pts | 5 / 7 | 0.774 | 42.67% | 43.33% | PASS |" in md
    assert "relative L2 error mean 0.0262" in md
