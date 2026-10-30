"""Phase 5: TensorRT-LLM (PyTorch backend) vs vLLM on a one-image-per-request trace, plus click accuracy through TensorRT-LLM.

    python serving/scripts/run_trtllm_bench.py --mode accuracy   # synth300 + ScreenSpot-400 through the TensorRT-LLM server (rule 3)
    python serving/scripts/run_trtllm_bench.py --mode bench      # alternating rounds: vllm then trtllm, and the reverse

WHY THE `single` TRACE. The agent trace of Phases 1-4 puts three images in one request (two few-shot screenshots and the current one). TensorRT-LLM
1.2.1 with transformers 4.57 failed on those prompts inside the Hugging Face Qwen2-VL processor ("index 3 is out of bounds for dimension 0 with
size 3": 4 image placeholders, 3 images); its single-image requests are what this script measures. So this is NOT the Phase 1 workload: every
request has one image, no shared few-shot prefix, so prefix caching is switched OFF on both engines (vLLM --no-enable-prefix-caching, TensorRT-LLM
enable_block_reuse false) and there is nothing for it to do.

Both engines: bf16 weights (the merged checkpoint; TensorRT-LLM reads a view of it, see make_trtllm_view.py), max 32 sequences, max length 4096,
image pixel budget 1003520. TensorRT-LLM runs with ONE local patch to the installed package (vision-tower attention flash_attention_2 -> sdpa,
13_trtllm_sdpa_patch.sh), recorded in the result.
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
from servebench.procs import Proc, gpu_mem_mib, http, wait_ready

HOME = Path.home()
RAW = ROOT / "serving" / "results" / "raw"
LOGS = HOME / "work" / "logs"
MODEL_DIR = ROOT / "serving" / "models" / "merged-clean"
VIEW_DIR = HOME / "work" / "merged-clean-trt"
SERVE_VLLM = ROOT / "serving" / "scripts" / "wsl" / "serve_vllm.sh"
SERVE_TRT = ROOT / "serving" / "scripts" / "wsl" / "serve_trtllm.sh"
PY = sys.executable
PORT = {"vllm": 8000, "trtllm": 8003}

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["bench", "accuracy"], required=True)
ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 16])
ap.add_argument("--rounds", type=int, default=3)
ap.add_argument("--trace", default="serving/traces/single")
ap.add_argument("--max-seqs", type=int, default=32)
ap.add_argument("--max-len", type=int, default=4096)
ap.add_argument("--ready-timeout", type=float, default=900.0)
ap.add_argument("--tag-prefix", default="trtllm-single")
a = ap.parse_args()

ts = time.strftime("%Y%m%d-%H%M%S")
LOGS.mkdir(parents=True, exist_ok=True)


def trt_versions() -> dict:
    code = ("import importlib.metadata as m;print({p: m.version(p) for p in ('tensorrt_llm','torch','transformers','flashinfer-python') "
            "if True})")
    try:
        out = subprocess.run(["bash", "-c", f"source {ROOT}/serving/scripts/wsl/trtllm_env.sh; $TRT_PY -c \"{code}\""], capture_output=True, text=True, timeout=120).stdout
        return eval(out.strip().splitlines()[-1])        # a dict literal printed by the line above
    except Exception as e:
        return {"error": repr(e)}


def patch_diff() -> str:
    f = HOME / "venvs/trtllm/lib/python3.12/site-packages/tensorrt_llm/_torch/models/modeling_qwen2vl.py"
    r = subprocess.run(["diff", str(f) + ".orig", str(f)], capture_output=True, text=True)
    return r.stdout.strip()


class Server:
    def __init__(self, engine: str):
        self.engine, self.proc, self.facts = engine, None, {}

    def __enter__(self):
        log = LOGS / f"{a.tag_prefix}-{self.engine}-{ts}.log"
        if self.engine == "vllm":
            env = dict(os.environ, PORT=str(PORT["vllm"]), PREFIX_CACHING="0", GPU_UTIL="0.85", MAX_SEQS=str(a.max_seqs), MAX_LEN=str(a.max_len))
            self.proc = Proc("vllm", ["bash", str(SERVE_VLLM)], env, log)
            self.flush = [f"http://127.0.0.1:{PORT['vllm']}/reset_prefix_cache", f"http://127.0.0.1:{PORT['vllm']}/reset_mm_cache"]
            self.model = "merged-clean"
        else:
            subprocess.run([PY, str(ROOT / "serving/scripts/make_trtllm_view.py"), "--src", str(MODEL_DIR), "--dst", str(VIEW_DIR)], check=True)
            env = dict(os.environ, PORT=str(PORT["trtllm"]), MODEL_DIR=str(VIEW_DIR), MAX_SEQS=str(a.max_seqs), MAX_LEN=str(a.max_len), REUSE="false")
            self.proc = Proc("trtllm", ["bash", str(SERVE_TRT)], env, log)
            self.flush = []
        t = wait_ready(PORT[self.engine], self.proc, timeout=a.ready_timeout)
        if self.engine == "trtllm":
            self.model = json.loads(http(f"http://127.0.0.1:{PORT['trtllm']}/v1/models")[1])["data"][0]["id"]
        self.base = f"http://127.0.0.1:{PORT[self.engine]}"
        self.facts = {"startup_s": t, "gpu_mem_loaded_mib": gpu_mem_mib(), "log": log.name, "model_name": self.model}
        return self

    def __exit__(self, *exc):
        self.proc.stop()
        subprocess.run(["pkill", "-f", "[t]rtllm-serve"], capture_output=True)
        time.sleep(6)


def loadtest(s: Server, conc: int):
    before = set(os.listdir(RAW))
    tag = f"{a.tag_prefix}-{s.engine}"
    cmd = [PY, "-u", str(ROOT / "serving/scripts/loadtest.py"), "--base-url", s.base, "--model", s.model, "--trace", a.trace, "--concurrency", str(conc),
           "--repeats", "1", "--engine", tag, "--tag", tag, "--model-dir", str(MODEL_DIR), "--gpu-index", "0", "--server-cmd", f"{s.engine} bf16, no prefix caching"]
    for u in s.flush:
        cmd += ["--flush-url", u]
    r = subprocess.run(cmd, capture_output=True, text=True)
    print((r.stdout.strip().splitlines() or [r.stderr[-300:]])[-1], flush=True)
    return sorted(set(os.listdir(RAW)) - before)


def accuracy(s: Server, tag: str):
    out = []
    for name in ("synth300", "screenspot400"):
        before = set(os.listdir(RAW))
        r = subprocess.run([PY, str(ROOT / "serving/scripts/accuracy_check.py"), "--backend", "openai", "--base-url", s.base, "--model", s.model, "--set", name,
                            "--tag", tag, "--model-dir", str(MODEL_DIR)], capture_output=True, text=True)
        print((r.stdout.strip().splitlines() or [r.stderr[-300:]])[-1], flush=True)
        out += sorted(set(os.listdir(RAW)) - before)
    return out


def record(kind: str, extra: dict) -> dict:
    return {"kind": kind, "env": envinfo.capture(MODEL_DIR, {"args": vars(a)}), "trtllm_env": trt_versions(), "trtllm_local_patch_diff": patch_diff(),
            "settings": {"dtype": "bf16", "max_seqs": a.max_seqs, "max_len": a.max_len, "prefix_caching": "off on both engines", "trace": a.trace,
                         "trtllm_backend": "pytorch", "trtllm_model_dir": "view of merged-clean (same weights, 4.x-format configs)"}, **extra}


def write(name: str, rec: dict):
    out = RAW / f"{name}_{ts}.json"
    out.write_text(envinfo.dumps_scrubbed(rec, indent=1))
    print("wrote", out.name, flush=True)


if a.mode == "accuracy":
    with Server("trtllm") as s:
        files = accuracy(s, "trtllm")
    write("trtllm_accuracy", record("trtllm_accuracy", {"files": files, "server": s.facts}))
else:
    rounds = []
    for rnd in range(a.rounds):
        for engine in (["vllm", "trtllm"] if rnd % 2 == 0 else ["trtllm", "vllm"]):
            with Server(engine) as s:
                runs = [{"concurrency": c, "loadtest_files": loadtest(s, c)} for c in a.concurrency]
            rounds.append({"round": rnd, "engine": engine, "order": ["vllm", "trtllm"] if rnd % 2 == 0 else ["trtllm", "vllm"], "server": s.facts, "runs": runs})
    write("trtllm_bench", record("trtllm_bench", {"rounds": rounds}))
