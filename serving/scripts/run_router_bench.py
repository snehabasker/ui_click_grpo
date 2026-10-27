"""Phase 2 benchmark: the Rust router in front of two vLLM replicas (FP8 weights, sharing one 8 GB GPU).

    python serving/scripts/run_router_bench.py --mode bench     # prefix-aware vs round-robin, same trace, 3 repeats
    python serving/scripts/run_router_bench.py --mode failure   # kill one replica mid-run, restart it, measure

Runs inside WSL2 with the client venv. Replicas are started with scripts/wsl/serve_vllm.sh; the router is the release binary built from
serving/router. Every result is written to results/raw/ with the environment record.
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "serving"))

from servebench import envinfo

HOME = Path.home()
RAW = ROOT / "serving" / "results" / "raw"
LOGS = HOME / "work" / "logs"
ROUTER_BIN = HOME / "work" / "cargo-target" / "release" / "prefix-router"
MODEL_DIR = ROOT / "serving" / "models" / "merged-clean"
SERVE = ROOT / "serving" / "scripts" / "wsl" / "serve_vllm.sh"
PY = sys.executable

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["bench", "failure"], required=True)
ap.add_argument("--concurrency", type=int, nargs="+", default=[4, 16, 32])
ap.add_argument("--repeats", type=int, default=3)
ap.add_argument("--policies", nargs="+", default=["prefix", "round-robin"])
ap.add_argument("--trace", default="serving/traces/default")
ap.add_argument("--sessions-by-conc", default="4:48,16:96,32:96")
ap.add_argument("--gpu-util", type=float, default=0.46, help="per replica, as a fraction of the whole GPU")
ap.add_argument("--max-seqs", type=int, default=8)
ap.add_argument("--max-len", type=int, default=3072)
ap.add_argument("--router-port", type=int, default=9000)
ap.add_argument("--kill-after", type=float, default=60.0, help="failure mode: seconds into the run before the replica is killed")
ap.add_argument("--restart-after", type=float, default=30.0, help="failure mode: seconds after the kill before the restart is launched")
ap.add_argument("--failure-concurrency", type=int, default=4)
ap.add_argument("--skip-accuracy", action="store_true")
a = ap.parse_args()

PORTS = [8001, 8002]
ts = time.strftime("%Y%m%d-%H%M%S")
LOGS.mkdir(parents=True, exist_ok=True)


def http(url: str, method: str = "GET", timeout: float = 3.0):
    req = urllib.request.Request(url, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, r.read().decode()


def gpu_mem() -> int | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout.split()
        return int(out[0])
    except Exception:
        return None


class Proc:
    """A child process in its own process group, so killing it also kills vLLM's engine-core child and frees the GPU."""

    def __init__(self, name, cmd, env, log):
        self.name, self.log = name, log
        self.f = open(log, "w")
        self.p = subprocess.Popen(cmd, stdout=self.f, stderr=subprocess.STDOUT, env=env, start_new_session=True)

    def alive(self):
        return self.p.poll() is None

    def kill(self, sig=signal.SIGKILL):
        if self.alive():
            os.killpg(self.p.pid, sig)
            try:
                self.p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(self.p.pid, signal.SIGKILL)

    def stop(self):
        self.kill(signal.SIGTERM)
        self.kill(signal.SIGKILL)


def start_replica(i: int, tag: str) -> Proc:
    env = dict(os.environ, PORT=str(PORTS[i]), GPU_UTIL=str(a.gpu_util), MAX_SEQS=str(a.max_seqs), MAX_LEN=str(a.max_len),
               EXTRA="--quantization fp8")
    return Proc(f"replica{i}", ["bash", str(SERVE)], env, LOGS / f"router-{tag}-replica{i}-{ts}.log")


def wait_ready(port: int, proc: Proc, timeout: float = 600.0) -> float:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if not proc.alive():
            raise RuntimeError(f"{proc.name} exited before becoming ready; see {proc.log}")
        try:
            if http(f"http://127.0.0.1:{port}/v1/models")[0] == 200:
                return time.time() - t0
        except Exception:
            pass
        time.sleep(2)
    raise RuntimeError(f"{proc.name} not ready after {timeout}s")


def start_router(policy: str) -> Proc:
    cmd = [str(ROUTER_BIN), "--listen", f"127.0.0.1:{a.router_port}", "--policy", policy]
    for p in PORTS:
        cmd += ["--backend", f"http://127.0.0.1:{p}"]
    p = Proc("router", cmd, dict(os.environ, RUST_LOG="info"), LOGS / f"router-{policy}-{ts}.log")
    t0 = time.time()
    while time.time() - t0 < 15:
        try:
            if http(f"http://127.0.0.1:{a.router_port}/router/health")[0] == 200:
                return p
        except Exception:
            time.sleep(0.2)
    raise RuntimeError("router did not come up")


def router_metrics() -> dict:
    """The router's own counters as {series: value}."""
    out = {}
    for line in http(f"http://127.0.0.1:{a.router_port}/metrics")[1].splitlines():
        if line and not line.startswith("#"):
            k, _, v = line.rpartition(" ")
            try:
                out[k] = float(v)
            except ValueError:
                pass
    return out


def flush_urls():
    return [f"http://127.0.0.1:{p}{path}" for p in PORTS for path in ("/reset_prefix_cache", "/reset_mm_cache")]


def run_loadtest(tag: str, conc: int, repeats: int, sessions: int | None, extra=()):
    base = f"http://127.0.0.1:{a.router_port}"
    cmd = [PY, "-u", str(ROOT / "serving/scripts/loadtest.py"), "--base-url", base, "--model", "merged-clean", "--trace", a.trace,
           "--concurrency", str(conc), "--repeats", str(repeats), "--engine", tag, "--tag", tag, "--model-dir", str(MODEL_DIR),
           "--gpu-index", "0", "--server-cmd", f"prefix-router in front of 2x vllm fp8 ({tag})"]
    for u in flush_urls():
        cmd += ["--flush-url", u]
    if sessions:
        cmd += ["--max-sessions", str(sessions)]
    cmd += list(extra)
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr[-300:], flush=True)
    return r


def replica_facts(procs) -> list:
    import re
    facts = []
    for p in procs:
        t = Path(p.log).read_text(errors="ignore")
        m = re.search(r"GPU KV cache size: ([\d,]+) tokens", t)
        m2 = re.search(r"Model loading took ([\d.]+) GiB", t)
        facts.append({"log": Path(p.log).name, "kv_cache_tokens": int(m.group(1).replace(",", "")) if m else None,
                      "model_gib": float(m2.group(1)) if m2 else None})
    return facts


def common_record(kind: str, procs, extra: dict) -> dict:
    sha = subprocess.run(["sha256sum", str(ROUTER_BIN)], capture_output=True, text=True).stdout.split()[0]
    return {"kind": kind, "env": envinfo.capture(MODEL_DIR, {"args": vars(a)}), "router_binary_sha256": sha,
            "router_version": subprocess.run([str(ROUTER_BIN), "--version"], capture_output=True, text=True).stdout.strip(),
            "replicas": replica_facts(procs), "replica_settings": {"quantization": "fp8", "gpu_util_each": a.gpu_util,
                                                                  "max_seqs": a.max_seqs, "max_len": a.max_len},
            "gpu_mem_loaded_mib": gpu_mem(), **extra}


def write(name: str, rec: dict):
    out = RAW / f"{name}_{ts}.json"
    out.write_text(envinfo.dumps_scrubbed(rec, indent=1))
    print("wrote", out.name, flush=True)


def main_bench():
    procs = []
    try:
        for i in range(2):
            procs.append(start_replica(i, "bench"))
            print(f"replica {i} ready after {wait_ready(PORTS[i], procs[i]):.0f}s", flush=True)
        time.sleep(5)
        info = common_record("router_bench", procs, {"policies": a.policies, "gpu_mem_two_replicas_mib": gpu_mem()})
        sessions = dict(kv.split(":") for kv in a.sessions_by_conc.split(","))
        snapshots = []
        for policy in a.policies:
            tag = f"router-{policy}"
            router = start_router(policy)
            try:
                if not a.skip_accuracy:
                    subprocess.run([PY, "-u", str(ROOT / "serving/scripts/accuracy_check.py"), "--backend", "openai", "--base-url",
                                    f"http://127.0.0.1:{a.router_port}", "--model", "merged-clean", "--set", "synth300", "--tag", tag,
                                    "--concurrency", "4", "--model-dir", str(MODEL_DIR)])
                for c in a.concurrency:
                    for rep in range(a.repeats):
                        before = router_metrics()
                        run_loadtest(tag, c, 1, int(sessions[str(c)]) if str(c) in sessions else None)
                        snapshots.append({"policy": policy, "concurrency": c, "repeat": rep + 1, "before": before, "after": router_metrics()})
            finally:
                router.stop()
        info["router_metric_snapshots"] = snapshots
        write("router_bench", info)
    finally:
        for p in procs:
            p.stop()


def main_failure():
    procs = [None, None]
    timeline, events = [], {}
    stop = threading.Event()
    t_start = time.time()

    def rel():
        return time.time() - t_start

    def monitor():
        while not stop.is_set():
            try:
                h = json.loads(http(f"http://127.0.0.1:{a.router_port}/router/health", timeout=1.0)[1])
            except urllib.error.HTTPError as e:  # 503 (nothing healthy) still carries a JSON body
                h = json.loads(e.read().decode())
            except Exception:
                h = None
            try:
                m = router_metrics()
            except Exception:
                m = {}
            row = {"t": round(rel(), 3)}
            if h:
                for i, b in enumerate(h["backends"]):
                    row[f"healthy{i}"] = b["healthy"]
                    row[f"breaker{i}"] = b["breaker"]
            for i, p in enumerate(PORTS):
                row[f"ok{i}"] = m.get(f'router_backend_ok_total{{backend="http://127.0.0.1:{p}"}}')
                row[f"err{i}"] = m.get(f'router_backend_errors_total{{backend="http://127.0.0.1:{p}"}}')
            row["retries"] = m.get("router_retries_total")
            timeline.append(row)
            time.sleep(0.1)

    try:
        for i in range(2):
            procs[i] = start_replica(i, "failure")
            print(f"replica {i} ready after {wait_ready(PORTS[i], procs[i]):.0f}s", flush=True)
        time.sleep(5)
        router = start_router("prefix")
        t_start = time.time()
        th = threading.Thread(target=monitor, daemon=True)
        th.start()
        lt = {}

        def load():
            lt["result"] = run_loadtest("router-failure", a.failure_concurrency, 1, None)

        lth = threading.Thread(target=load)
        lth.start()

        time.sleep(a.kill_after)
        events["kill_replica1_at"] = round(rel(), 3)
        procs[1].kill(signal.SIGKILL)
        print(f"[{rel():.1f}s] killed replica 1", flush=True)
        time.sleep(a.restart_after)
        events["restart_launched_at"] = round(rel(), 3)
        procs[1] = start_replica(1, "failure-restart")
        wait_ready(PORTS[1], procs[1])
        events["restarted_replica_ready_at"] = round(rel(), 3)
        print(f"[{rel():.1f}s] replica 1 serving again", flush=True)
        lth.join(timeout=900)
        time.sleep(3)  # let the monitor see the end state
        stop.set()
        th.join(timeout=5)

        def first(pred, after=0.0):
            for r in timeline:
                if r["t"] >= after and pred(r):
                    return r["t"]
            return None

        k = events["kill_replica1_at"]
        down = first(lambda r: r.get("healthy1") is False, k)
        ready = events["restarted_replica_ready_at"]
        up = first(lambda r: r.get("healthy1") is True, ready)
        ok_at_ready = next((r["ok1"] for r in timeline if r["t"] >= ready and r.get("ok1") is not None), None)
        first_ok = first(lambda r: r.get("ok1") is not None and ok_at_ready is not None and r["ok1"] > ok_at_ready, ready)
        events.update(router_marked_unhealthy_at=down, router_marked_healthy_again_at=up, first_request_served_by_restarted_replica_at=first_ok,
                      detect_seconds=None if down is None else round(down - k, 3),
                      healthy_again_seconds_after_ready=None if up is None else round(up - ready, 3),
                      first_request_seconds_after_ready=None if first_ok is None else round(first_ok - ready, 3))
        print(json.dumps(events, indent=1), flush=True)
        info = common_record("router_failure", [p for p in procs if p], {"events": events, "timeline": timeline,
                                                                          "loadtest_stdout_last_line": lt.get("result").stdout.strip().splitlines()[-1] if lt.get("result") and lt["result"].stdout.strip() else None})
        write("router_failure", info)
        router.stop()
    finally:
        stop.set()
        for p in procs:
            if p:
                p.stop()


if not ROUTER_BIN.exists():
    sys.exit(f"build the router first: {ROUTER_BIN} is missing")
(main_bench if a.mode == "bench" else main_failure)()
