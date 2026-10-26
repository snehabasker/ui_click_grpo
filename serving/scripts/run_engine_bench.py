"""Start an inference engine, verify click accuracy through it, replay the trace at several concurrency levels, stop it.

Runs inside WSL2 with the client venv; the engine runs from its own venv via scripts/wsl/serve_<engine>.sh.

    python serving/scripts/run_engine_bench.py --engine vllm --label vllm-prefix-on --prefix-caching 1 \
        --concurrency 1 4 16 32 --repeats 3

Everything that identifies the run (server command, engine/package versions, startup time, KV-cache size, GPU memory, attention
backend, deviations from the engine defaults) is written to results/raw/engine_<label>_<ts>.json next to the per-run files.
"""
import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "serving"))

from servebench import envinfo

HOME = Path.home()
ENGINE_PY = {"vllm": HOME / "venvs/vllm/bin/python", "sglang": HOME / "venvs/sglang/bin/python"}
FLUSH = {"vllm": ["/reset_prefix_cache", "/reset_mm_cache"], "sglang": ["/flush_cache"]}
MODEL_DIR = ROOT / "serving" / "models" / "merged-clean"
RAW = ROOT / "serving" / "results" / "raw"

ap = argparse.ArgumentParser()
ap.add_argument("--engine", choices=["vllm", "sglang"], required=True)
ap.add_argument("--label", required=True, help="e.g. vllm-prefix-on; used in every file name")
ap.add_argument("--prefix-caching", type=int, choices=[0, 1], default=1)
ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 16, 32])
ap.add_argument("--repeats", type=int, default=3)
ap.add_argument("--trace", default="serving/traces/default")
ap.add_argument("--max-sessions", type=int, default=None)
ap.add_argument("--port", type=int, default=8000)
ap.add_argument("--gpu-util", type=float, default=0.85)
ap.add_argument("--max-seqs", type=int, default=32)
ap.add_argument("--max-len", type=int, default=4096)
ap.add_argument("--startup-timeout", type=int, default=900)
ap.add_argument("--skip-accuracy", action="store_true")
ap.add_argument("--accuracy-only", action="store_true")
ap.add_argument("--extra", default="", help="extra flags appended to the engine command (recorded)")
a = ap.parse_args()

PY = sys.executable
ts = time.strftime("%Y%m%d-%H%M%S")
base = f"http://127.0.0.1:{a.port}"
log_path = HOME / "work" / "logs" / f"{a.label}-{ts}.log"
log_path.parent.mkdir(parents=True, exist_ok=True)


def http_ok(path: str, timeout: float = 3.0) -> bool:
    try:
        with urllib.request.urlopen(base + path, timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def gpu_mem_mib() -> int | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout.strip().splitlines()
        return int(out[0])
    except Exception:
        return None


def engine_versions() -> dict:
    code = ("import importlib.metadata as m\nimport json\nout={}\nfor p in ('vllm','sglang','sgl-kernel','sglang-kernel','torch','transformers',"
            "'triton','flashinfer-python','flashinfer-cubin','flash-attn'):\n    try: out[p]=m.version(p)\n    except Exception: pass\nprint(json.dumps(out))")
    try:
        return json.loads(subprocess.run([str(ENGINE_PY[a.engine]), "-c", code], capture_output=True, text=True, timeout=60).stdout)
    except Exception:
        return {}


def parse_log(text: str) -> dict:
    info = {}
    m = re.search(r"GPU KV cache size: ([\d,]+) tokens", text)
    if m:
        info["kv_cache_tokens"] = int(m.group(1).replace(",", ""))
    m = re.search(r"Using (\w+) attention backend", text)
    if m:
        info["attention_backend"] = m.group(1)
    m = re.search(r"Model loading took ([\d.]+) GiB", text)
    if m:
        info["model_memory_gib"] = float(m.group(1))
    m = re.search(r"(?:max_total_num_tokens|#tokens)[=:]\s*(\d+)", text)
    if m:
        info["kv_cache_tokens_sglang"] = int(m.group(1))
    m = re.search(r"^\+ exec (.*)$", text, re.M)
    if m:
        info["server_cmd"] = m.group(1).strip()
    return info


env = dict(os.environ, PORT=str(a.port), PREFIX_CACHING=str(a.prefix_caching), GPU_UTIL=str(a.gpu_util),
           MAX_SEQS=str(a.max_seqs), MAX_LEN=str(a.max_len), EXTRA=a.extra)
serve = ROOT / "serving" / "scripts" / "wsl" / f"serve_{a.engine}.sh"
print(f"[{a.label}] starting {a.engine}; log: {log_path}", flush=True)
mem_before = gpu_mem_mib()
t0 = time.time()
with open(log_path, "w") as lf:
    proc = subprocess.Popen(["bash", str(serve)], stdout=lf, stderr=subprocess.STDOUT, env=env, start_new_session=True)

result = {"kind": "engine_run", "label": a.label, "engine": a.engine, "prefix_caching": bool(a.prefix_caching),
          "settings": {k: v for k, v in vars(a).items()}, "gpu_mem_before_mib": mem_before}
ok = False
try:
    while time.time() - t0 < a.startup_timeout:
        if proc.poll() is not None:
            break
        if http_ok("/v1/models"):
            ok = True
            break
        time.sleep(3)
    startup_s = time.time() - t0
    text = log_path.read_text(errors="ignore")
    result.update(started=ok, startup_s=startup_s, log_file=str(log_path.name), **parse_log(text))
    if not ok:
        result["failure_tail"] = text.splitlines()[-25:]
        print(f"[{a.label}] FAILED to start after {startup_s:.0f}s", flush=True)
    else:
        time.sleep(5)
        result["gpu_mem_loaded_mib"] = gpu_mem_mib()
        result["versions"] = engine_versions()
        print(f"[{a.label}] ready after {startup_s:.0f}s; gpu {result['gpu_mem_loaded_mib']} MiB", flush=True)
        # Caches reset before every repeat so repeats are independent: the prefix cache, and for vLLM also the multimodal
        # (image) cache, otherwise repeats 2 and 3 would start with warm image preprocessing. With prefix caching switched
        # off there is no prefix cache to flush, but the image cache is still reset.
        flush_urls = [base + p for p in FLUSH[a.engine]]
        if not a.prefix_caching:
            flush_urls = [u for u in flush_urls if "mm_cache" in u]
        common = [PY, "-u"]
        if not a.skip_accuracy:
            for s in ("synth300", "screenspot400"):
                subprocess.run(common + [str(ROOT / "serving/scripts/accuracy_check.py"), "--backend", "openai", "--base-url", base,
                                         "--model", "merged-clean", "--set", s, "--tag", a.label, "--concurrency", "4",
                                         "--model-dir", str(MODEL_DIR)], check=False)
        if not a.accuracy_only:
            for c in a.concurrency:
                cmd = common + [str(ROOT / "serving/scripts/loadtest.py"), "--base-url", base, "--model", "merged-clean",
                                "--trace", a.trace, "--concurrency", str(c), "--repeats", str(a.repeats), "--engine", a.label,
                                "--tag", a.label, "--model-dir", str(MODEL_DIR), "--gpu-index", "0",
                                "--server-cmd", result.get("server_cmd", "")]
                for u in flush_urls:
                    cmd += ["--flush-url", u]
                if a.max_sessions:
                    cmd += ["--max-sessions", str(a.max_sessions)]
                subprocess.run(cmd, check=False)
finally:
    if proc.poll() is None:
        os.killpg(proc.pid, signal.SIGTERM)
        for _ in range(40):
            if proc.poll() is not None:
                break
            time.sleep(1)
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
    time.sleep(3)
    result["gpu_mem_after_stop_mib"] = gpu_mem_mib()
    result["env"] = envinfo.capture(MODEL_DIR, {"engine_python": str(ENGINE_PY[a.engine])})
    RAW.mkdir(parents=True, exist_ok=True)
    out = RAW / f"engine_{a.label}_{ts}.json"
    out.write_text(json.dumps(result, indent=1))
    print(f"[{a.label}] wrote {out.name}; gpu after stop: {result['gpu_mem_after_stop_mib']} MiB", flush=True)
