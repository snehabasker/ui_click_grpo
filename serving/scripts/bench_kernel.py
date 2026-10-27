"""Benchmark the fused RMSNorm + INT8 Triton kernel against PyTorch eager and torch.compile, with a measured memory-bandwidth ceiling.

    python serving/scripts/bench_kernel.py --reps 5          (needs a CUDA GPU; run it in the vllm venv via wsl/run_kernel_bench.sh)

Every cell is timed with triton.testing.do_bench (median of many launches, includes the Python/dispatcher launch cost) and with
triton.testing.do_bench_cudagraph (the launches are captured in a CUDA graph, so it approximates GPU-side time). That split matters
at small row counts, where a launch costs more than the work. The whole grid is repeated --reps times; within a repeat the order of
implementations alternates, and odd repeats walk the grid backwards, so a GPU power/clock state cannot favour one implementation.
GB/s is "effective": the minimum bytes the operation must move (read x and weight, write int8 codes and one fp32 scale per row)
divided by time, the same numerator for every implementation.
"""
import argparse
import json
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "serving"))

import torch
import triton
import triton.testing as tt

from servebench import envinfo
from servekernels import rmsnorm_int8 as K

EPS = 1e-6
HIDDEN = [1536, 4096, 8192]                 # 1536 = Qwen2-VL-2B's LLM hidden size; the others are common 7B-class sizes
ROWS = [1, 16, 128, 1024, 4096, 16384]      # 1-16: decode batch; 1024-16384: a prefill chunk (one screenshot is ~1.8k tokens)
DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16}
IMPLS = ["triton_fused", "torch_eager", "torch_compile"]

ap = argparse.ArgumentParser()
ap.add_argument("--reps", type=int, default=5)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--hidden", type=int, nargs="+", default=HIDDEN)
ap.add_argument("--rows", type=int, nargs="+", default=ROWS)
ap.add_argument("--dtypes", nargs="+", default=list(DTYPES), choices=list(DTYPES))
ap.add_argument("--warmup-ms", type=int, default=25)
ap.add_argument("--rep-ms", type=int, default=100)
ap.add_argument("--tag", default="kernel")
a = ap.parse_args()
if a.reps < 3:
    sys.exit("rule 2: at least 3 repeats")


def smi(fields):
    out = subprocess.run(["nvidia-smi", f"--query-gpu={','.join(fields)}", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout.strip().split(", ")
    return dict(zip(fields, out))


STATE_FIELDS = ["clocks.sm", "clocks.mem", "power.draw", "temperature.gpu", "clocks_throttle_reasons.active"]
STATIC_FIELDS = ["name", "clocks.max.sm", "clocks.max.mem", "power.limit", "memory.total"]


def device_props():
    out = {"l2_cache_bytes": getattr(torch.cuda.get_device_properties(0), "L2_cache_size", None)}
    try:
        p = triton.runtime.driver.active.utils.get_device_properties(torch.cuda.current_device())
        out.update({k: p[k] for k in p if k in ("multiprocessor_count", "max_shared_mem", "mem_clock_rate", "mem_bus_width", "sm_clock_rate")})
    except Exception as e:                      # not exposed by every Triton version
        out["unavailable"] = repr(e)
    return out


class Sampler:
    """Polls nvidia-smi once a second while a repeat runs (a reading taken between repeats sees an idle GPU and is useless)."""

    def __init__(self):
        self.rows, self._stop = [], threading.Event()
        self._t = threading.Thread(target=self._loop, daemon=True)

    def _loop(self):
        while not self._stop.is_set():
            try:
                self.rows.append(smi(STATE_FIELDS))
            except Exception:
                pass
            self._stop.wait(1.0)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join()

    def summary(self):
        num = lambda k: [float(r[k]) for r in self.rows if r.get(k, "").replace(".", "", 1).isdigit()]
        pw, sm, tc = num("power.draw"), num("clocks.sm"), num("temperature.gpu")
        return {"samples": len(self.rows), "power_w_median": statistics.median(pw) if pw else None, "power_w_max": max(pw, default=None),
                "sm_mhz_median": statistics.median(sm) if sm else None, "sm_mhz_min": min(sm, default=None),
                "temp_c_max": max(tc, default=None), "throttle_reasons_seen": sorted({r["clocks_throttle_reasons.active"] for r in self.rows})}


def make(m, n, dtype):
    g = torch.Generator(device="cuda").manual_seed(a.seed)
    x = torch.randn((m, n), device="cuda", generator=g).to(dtype)
    w = (torch.rand(n, device="cuda", generator=g) + 0.5).to(dtype)
    return x, w


def min_bytes(m, n, esize):
    return m * n * esize + n * esize + m * n + 4 * m


def timed(fn):
    """(do_bench median ms, do_bench_cudagraph median ms or None + error text)"""
    ms = tt.do_bench(fn, warmup=a.warmup_ms, rep=a.rep_ms, return_mode="median")
    try:
        g = tt.do_bench_cudagraph(fn, rep=a.rep_ms, return_mode="median")
        return ms, g, None
    except Exception as e:
        return ms, None, f"{type(e).__name__}: {str(e)[:200]}"


def busy(seconds):
    x = torch.randn(8192, 8192, device="cuda", dtype=torch.float16)
    t = time.time()
    while time.time() - t < seconds:
        x @ x
    torch.cuda.synchronize()


# ---- cells, correctness, compiled reference -------------------------------------------------------------------------------------
import torch._dynamo
torch._dynamo.config.cache_size_limit = 256
compiled = torch.compile(K.reference)
cells = [(dn, n, m) for dn in a.dtypes for n in a.hidden for m in a.rows]
fns, correctness = {}, {}
for dn, n, m in cells:
    x, w = make(m, n, DTYPES[dn])
    q, s = torch.ops.serving.rmsnorm_int8(x, w, EPS)
    qr, sr = K.reference(x, w, EPS)
    d = (q.int() - qr.int()).abs()
    correctness[f"{dn}/{n}/{m}"] = {"max_code_diff": int(d.max()), "frac_codes_differing": float((d > 0).float().mean()),
                                    "max_rel_scale_err": float(((s - sr).abs() / sr.abs().clamp_min(1e-30)).max())}
    compiled(x, w, EPS)                          # compile outside the timed region
    fns[(dn, n, m)] = {"triton_fused": lambda x=x, w=w: torch.ops.serving.rmsnorm_int8(x, w, EPS),
                       "torch_eager": lambda x=x, w=w: K.reference(x, w, EPS),
                       "torch_compile": lambda x=x, w=w: compiled(x, w, EPS)}
    # does the compiled reference agree with the eager one? (a speed number for a wrong function would be meaningless)
    qc, sc = compiled(x, w, EPS)
    correctness[f"{dn}/{n}/{m}"]["compiled_vs_eager_max_code_diff"] = int((qc.int() - qr.int()).abs().max())
print(f"{len(cells)} cells compiled and checked", flush=True)

# ---- ceilings: measured copy and read-only bandwidth ----------------------------------------------------------------------------
ceil_sizes_mib = [64, 256]


def ceilings():
    out = {}
    for mib in ceil_sizes_mib:
        src = torch.empty(mib * 2**20 // 2, dtype=torch.float16, device="cuda").normal_()
        dst = torch.empty_like(src)
        t_copy = tt.do_bench(lambda: dst.copy_(src), warmup=a.warmup_ms, rep=a.rep_ms, return_mode="median")
        t_read = tt.do_bench(lambda: src.sum(), warmup=a.warmup_ms, rep=a.rep_ms, return_mode="median")
        out[f"copy_{mib}MiB_GBps"] = 2 * src.numel() * 2 / (t_copy * 1e-3) / 1e9           # read + write
        out[f"read_only_{mib}MiB_GBps"] = src.numel() * 2 / (t_read * 1e-3) / 1e9           # sum() reads once
    return out


# ---- run ------------------------------------------------------------------------------------------------------------------------
busy(5)
runs = []
for rep in range(a.reps):
    order = cells if rep % 2 == 0 else cells[::-1]
    impls = IMPLS if rep % 2 == 0 else IMPLS[::-1]
    with Sampler() as sampler:
        ceil = ceilings()
        for dn, n, m in order:
            for impl in impls:
                ms, ms_graph, err = timed(fns[(dn, n, m)][impl])
                b = min_bytes(m, n, torch.finfo(DTYPES[dn]).bits // 8)
                runs.append({"rep": rep, "dtype": dn, "hidden": n, "rows": m, "impl": impl, "ms": ms, "ms_cudagraph": ms_graph,
                             "gbps": b / (ms * 1e-3) / 1e9, "gbps_cudagraph": None if ms_graph is None else b / (ms_graph * 1e-3) / 1e9,
                             "min_bytes": b, "cudagraph_error": err})
    gs = sampler.summary()
    runs.append({"rep": rep, "ceilings": ceil, "gpu_during_rep": gs})
    print(f"rep {rep + 1}/{a.reps} done: copy {ceil['copy_256MiB_GBps']:.0f} GB/s, read-only {ceil['read_only_256MiB_GBps']:.0f} GB/s, "
          f"{gs['power_w_median']} W median, sm {gs['sm_mhz_median']} MHz median", flush=True)

rec = {"kind": "kernel_bench", "tag": a.tag,
       "env": envinfo.capture(None, {"args": vars(a), "torch": torch.__version__, "triton": triton.__version__,
                                     "cuda_runtime": torch.version.cuda, "gpu_static": smi(STATIC_FIELDS),
                                     "device_props": device_props(), "seed": a.seed}),
       "correctness": correctness, "runs": runs}
out = ROOT / "serving" / "results" / "raw" / f"kernel_bench_{a.tag}_{time.strftime('%Y%m%d-%H%M%S')}.json"
out.write_text(envinfo.dumps_scrubbed(rec, indent=1))
print("wrote", out)
