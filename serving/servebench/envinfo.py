"""Environment capture, attached to every raw result so each number traces back to an exact setup."""
from __future__ import annotations

import datetime
import hashlib
import importlib.metadata as md
import platform
import subprocess
import sys
from pathlib import Path

PACKAGES = ["torch", "transformers", "peft", "vllm", "sglang", "sgl-kernel", "triton", "triton-windows", "tensorrt_llm",
            "flashinfer-python", "flash-attn", "aiohttp"]


def _run(cmd: list[str], timeout: int = 15) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout.strip()
    except Exception:
        return ""


def gpu_info() -> dict:
    rows = _run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,compute_cap", "--format=csv,noheader"])
    gpus = []
    for line in rows.splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) >= 4:
            gpus.append({"name": p[0], "driver": p[1], "memory_total": p[2], "compute_cap": p[3]})
    cuda = None
    for line in _run(["nvidia-smi"]).splitlines():
        if "CUDA Version:" in line:
            cuda = line.split("CUDA Version:")[1].split()[0]
            break
    return {"gpus": gpus, "driver_cuda_version": cuda}


def package_versions() -> dict:
    out = {}
    for p in PACKAGES:
        try:
            out[p] = md.version(p)
        except md.PackageNotFoundError:
            pass
    return out


def file_sha256(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def model_hash(model_dir: str | Path) -> str:
    """sha256 over (name, sha256) of every weight file plus config.json. Cached in <dir>/MODEL_HASH."""
    d = Path(model_dir)
    cache = d / "MODEL_HASH"
    if cache.exists():
        return cache.read_text().strip()
    files = sorted(d.glob("*.safetensors")) + [d / "config.json"]
    h = hashlib.sha256()
    for f in files:
        if f.exists():
            h.update(f.name.encode())
            h.update(file_sha256(f).encode())
    digest = h.hexdigest()
    try:
        cache.write_text(digest)
    except OSError:
        pass
    return digest


def repo_commit(repo: str | Path | None = None) -> str | None:
    cmd = ["git"] + (["-C", str(repo)] if repo else []) + ["rev-parse", "HEAD"]
    return _run(cmd) or None


def capture(model_dir: str | Path | None = None, extra: dict | None = None) -> dict:
    """GPU, driver, CUDA, package versions, model hash, command line, date. ``extra`` carries seed, concurrency, etc."""
    return {
        "date": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "command": " ".join(sys.argv),
        "gpu": gpu_info(),
        "packages": package_versions(),
        "model_dir": str(model_dir) if model_dir else None,
        "model_hash": model_hash(model_dir) if model_dir and Path(model_dir).exists() else None,
        "repo_commit": repo_commit(Path(__file__).resolve().parents[2]),
        **(extra or {}),
    }
