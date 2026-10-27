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
