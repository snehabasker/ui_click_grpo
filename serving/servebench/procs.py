"""Small helpers shared by the benchmark drivers that start engines: child processes in their own group, readiness polling, GPU memory.
(run_router_bench.py has its own older copies; they are not refactored so that Phase 2's committed code stays as it was run.)"""
from __future__ import annotations

import os
import signal
import subprocess
import time
import urllib.request


def http(url: str, method: str = "GET", timeout: float = 3.0):
    req = urllib.request.Request(url, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, r.read().decode()


def gpu_mem_mib() -> int | None:
    try:
        return int(subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                  capture_output=True, text=True, timeout=10).stdout.split()[0])
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
        self.f.close()


def wait_ready(port: int, proc: Proc, path: str = "/v1/models", timeout: float = 600.0) -> float:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if not proc.alive():
            raise RuntimeError(f"{proc.name} exited before becoming ready; see {proc.log}")
        try:
            if http(f"http://127.0.0.1:{port}{path}")[0] == 200:
                return time.time() - t0
        except Exception:
            pass
        time.sleep(2)
    raise RuntimeError(f"{proc.name} not ready after {timeout}s")


def dir_bytes(path) -> tuple[int, int]:
    """(total bytes, number of files) under a directory; (0, 0) if it does not exist."""
    total = n = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
                n += 1
            except OSError:
                pass
    return total, n
