"""Phase 4 (one-GPU attempt): prefill/decode disaggregation with vLLM's ExampleConnector vs one colocated instance.

    python serving/scripts/run_disagg_bench.py --mode smoke      # start the pair, send 2 sessions, check the KV store; not a result
    python serving/scripts/run_disagg_bench.py --mode bench      # alternating rounds: colocated, then split (and the reverse)
    python serving/scripts/run_disagg_bench.py --mode accuracy   # click accuracy through both setups on both fixed sets (rule 3)

WHAT THIS IS NOT. Both instances share the one 8 GB GPU, so prefill still competes with decode for the same SMs and memory bandwidth,
and the KV cache moves through FILES ON DISK (vLLM's ExampleConnector, an example connector), not over NVLink / PCIe / RDMA. It tests
that the disaggregated path works end to end and what that path costs on this machine; it is not a model of a two-GPU deployment.

Setups (all vLLM, FP8 weights as in Phase 2, same max_model_len / max_num_seqs):
  colocated  ONE instance, 0.80 GiB KV cache (the same total KV memory as the pair)
  split      a PREFILL instance (kv_producer) + a DECODE instance (kv_consumer), 0.40 GiB KV each, behind servebench.disagg_proxy
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "serving"))

from servebench import envinfo
from servebench.procs import Proc, dir_bytes, gpu_mem_mib, http, wait_ready

HOME = Path.home()
RAW = ROOT / "serving" / "results" / "raw"
LOGS = HOME / "work" / "logs"
MODEL_DIR = ROOT / "serving" / "models" / "merged-clean"
SERVE = ROOT / "serving" / "scripts" / "wsl" / "serve_vllm.sh"
STORE = HOME / "work" / "kvstore"
PY = sys.executable
PROXY_PORT, P_PORT, D_PORT = 9100, 8001, 8002

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["smoke", "bench", "accuracy"], required=True)
ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 16])
ap.add_argument("--sessions-by-conc", default="1:16,4:48,16:96")
ap.add_argument("--rounds", type=int, default=3, help="bench: repeats of the whole colocated/split pair; order alternates each round")
ap.add_argument("--trace", default="serving/traces/default")
ap.add_argument("--kv-gib-split", type=float, default=0.40, help="KV cache per instance in the split setup")
ap.add_argument("--kv-gib-colocated", type=float, default=0.80)
ap.add_argument("--gpu-util-split", type=float, default=0.40)
ap.add_argument("--gpu-util-colocated", type=float, default=0.55)
ap.add_argument("--setups", nargs="+", default=["colocated", "split"], choices=["colocated", "split"])
ap.add_argument("--producer-extra", default="", help="extra vllm flags for the PREFILL instance only (no spaces inside a flag's value)")
ap.add_argument("--tag-prefix", default="disagg", help="load-test tag prefix; use explore-disagg for diagnostic runs that are not results")
ap.add_argument("--max-seqs", type=int, default=8)
ap.add_argument("--max-len", type=int, default=3072)
a = ap.parse_args()

ts = time.strftime("%Y%m%d-%H%M%S")
LOGS.mkdir(parents=True, exist_ok=True)
sessions_by_conc = {int(k): int(v) for k, v in (x.split(":") for x in a.sessions_by_conc.split(","))}


def connector_flag(role: str) -> str:
    cfg = {"kv_connector": "ExampleConnector", "kv_role": role, "kv_connector_extra_config": {"shared_storage_path": str(STORE)}}
    return "--kv-transfer-config=" + json.dumps(cfg, separators=(",", ":"))      # no spaces: serve_vllm.sh word-splits $EXTRA


def start_instance(name: str, port: int, kv_gib: float, util: float, role: str | None) -> Proc:
    extra = f"--quantization fp8 --kv-cache-memory {int(kv_gib * 2**30)}" + (f" {connector_flag(role)}" if role else "")
    if role == "kv_producer" and a.producer_extra:
        extra += " " + a.producer_extra
    env = dict(os.environ, PORT=str(port), GPU_UTIL=str(util), MAX_SEQS=str(a.max_seqs), MAX_LEN=str(a.max_len), EXTRA=extra)
    return Proc(name, ["bash", str(SERVE)], env, LOGS / f"disagg-{name}-{ts}.log")


class Setup:
    """Starts and stops one configuration; ``base`` is where the load generator sends requests."""

    def __init__(self, kind: str):
        self.kind, self.procs, self.facts = kind, [], {}

    def __enter__(self):
        STORE.mkdir(parents=True, exist_ok=True)
        if self.kind == "colocated":
            self.procs = [start_instance("colocated", D_PORT, a.kv_gib_colocated, a.gpu_util_colocated, None)]
            wait_ready(D_PORT, self.procs[0])
            self.base, self.flush = f"http://127.0.0.1:{D_PORT}", [f"http://127.0.0.1:{D_PORT}/reset_prefix_cache",
                                                                    f"http://127.0.0.1:{D_PORT}/reset_mm_cache"]
        else:
            prefill = start_instance("prefill", P_PORT, a.kv_gib_split, a.gpu_util_split, "kv_producer")
            self.procs.append(prefill)
            wait_ready(P_PORT, prefill)
            decode = start_instance("decode", D_PORT, a.kv_gib_split, a.gpu_util_split, "kv_consumer")
            self.procs.append(decode)
            wait_ready(D_PORT, decode)
            proxy = Proc("proxy", [PY, "-m", "servebench.disagg_proxy", "--prefill", f"http://127.0.0.1:{P_PORT}",
                                   "--decode", f"http://127.0.0.1:{D_PORT}", "--port", str(PROXY_PORT), "--store", str(STORE)],
                         dict(os.environ, PYTHONPATH=str(ROOT / "serving")), LOGS / f"disagg-proxy-{ts}.log")
            self.procs.append(proxy)
            wait_ready(PROXY_PORT, proxy)
            self.base = f"http://127.0.0.1:{PROXY_PORT}"
            self.flush = [f"{self.base}/reset_prefix_cache", f"{self.base}/reset_mm_cache"]
        self.facts = {"gpu_mem_loaded_mib": gpu_mem_mib(), "logs": [Path(p.log).name for p in self.procs]}
        return self

    def __exit__(self, *exc):
        for p in reversed(self.procs):
            p.stop()
        time.sleep(5)                                    # let the driver release GPU memory before the next setup starts

    def proxy_stats(self):
        return json.loads(http(f"{self.base}/proxy/stats")[1]) if self.kind == "split" else None


def loadtest(setup: Setup, conc: int, tag: str):
    before = set(os.listdir(RAW))
    cmd = [PY, "-u", str(ROOT / "serving/scripts/loadtest.py"), "--base-url", setup.base, "--model", "merged-clean", "--trace", a.trace,
           "--concurrency", str(conc), "--repeats", "1", "--engine", tag, "--tag", tag, "--model-dir", str(MODEL_DIR), "--gpu-index", "0",
           "--max-sessions", str(sessions_by_conc[conc]), "--server-cmd", f"vllm fp8, {setup.kind}"]
    for u in setup.flush:
        cmd += ["--flush-url", u]
    r = subprocess.run(cmd, capture_output=True, text=True)
    print((r.stdout.strip().splitlines() or [r.stderr[-300:]])[-1], flush=True)
    return sorted(set(os.listdir(RAW)) - before)


def accuracy(setup: Setup, tag: str):
    out = []
    for s in ("synth300", "screenspot400"):
        before = set(os.listdir(RAW))
        r = subprocess.run([PY, str(ROOT / "serving/scripts/accuracy_check.py"), "--backend", "openai", "--base-url", setup.base, "--model",
                            "merged-clean", "--set", s, "--tag", tag, "--model-dir", str(MODEL_DIR)], capture_output=True, text=True)
        print((r.stdout.strip().splitlines() or [r.stderr[-300:]])[-1], flush=True)
        out += sorted(set(os.listdir(RAW)) - before)
    return out


def write(name: str, rec: dict, where: Path = RAW):
    out = where / f"{name}_{ts}.json"
    out.write_text(envinfo.dumps_scrubbed(rec, indent=1))
    print("wrote", out.name, flush=True)


def record(extra: dict) -> dict:
    return {"env": envinfo.capture(MODEL_DIR, {"args": vars(a)}),
            "settings": {"quantization": "fp8", "max_seqs": a.max_seqs, "max_len": a.max_len, "kv_gib_split_each": a.kv_gib_split,
                         "kv_gib_colocated": a.kv_gib_colocated, "connector": "ExampleConnector (KV through files on disk)",
                         "kv_store_dir": "~/work/kvstore (WSL ext4)", "gpus": 1}, **extra}


def main_smoke():
    with Setup("split") as s:
        files = loadtest(s, 1, "explore-disagg-smoke")
        size, n = dir_bytes(STORE)
        print(json.dumps({"proxy": s.proxy_stats(), "store_bytes": size, "store_files": n, "loadtest_files": files})[:600], flush=True)


def main_bench():
    rounds = []
    for rnd in range(a.rounds):
        order = [k for k in (["colocated", "split"] if rnd % 2 == 0 else ["split", "colocated"]) if k in a.setups]
        for kind in order:
            with Setup(kind) as s:
                runs = []
                for conc in a.concurrency:
                    files = loadtest(s, conc, f"{a.tag_prefix}-{kind}")
                    size, n = dir_bytes(STORE) if kind == "split" else (None, None)
                    st = s.proxy_stats()
                    runs.append({"concurrency": conc, "loadtest_files": files, "kv_store_bytes_after_run": size, "kv_store_files_after_run": n,
                                 "proxy_prefill_s_cumulative": st["prefill_s"] if st else None,
                                 "proxy_failures": st["failures"] if st else None})
                rounds.append({"round": rnd, "order": order, "setup": kind, "facts": s.facts, "runs": runs})
        write("disagg_bench_partial", record({"kind": "disagg_bench_partial", "rounds": rounds}), LOGS)   # progress only, not a result
    write("disagg_bench" if a.tag_prefix == "disagg" else f"{a.tag_prefix}_bench", record({"kind": "disagg_bench", "rounds": rounds}))


def main_accuracy():
    res = {}
    for kind in ("split", "colocated"):
        with Setup(kind) as s:
            res[kind] = {"tag": f"{a.tag_prefix}-{kind}", "files": accuracy(s, f"{a.tag_prefix}-{kind}"), "proxy": s.proxy_stats()}
    write("disagg_accuracy" if a.tag_prefix == "disagg" else f"{a.tag_prefix}_accuracy", record({"kind": "disagg_accuracy", "results": res}))


{"smoke": main_smoke, "bench": main_bench, "accuracy": main_accuracy}[a.mode]()
