"""Image-set manifests. Images are rendered once (on the machine that has the fonts the policy was trained on) and shipped as
PNGs with sha256 checksums, so every engine, on any OS, is fed byte-identical inputs."""
from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Item:
    id: str
    image: str                 # path relative to the manifest directory
    sha256: str
    instruction: str
    box: list                  # x0, y0, x1, y1 in the model's 0-1000 coordinates
    meta: dict = field(default_factory=dict)


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def save_png(img, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, format="PNG")
    return sha256_bytes(path.read_bytes())


def write_manifest(root: Path, items: list[Item], meta: dict) -> None:
    (root / "manifest.json").write_text(json.dumps({"meta": meta, "items": [asdict(i) for i in items]}, indent=1))


def load_manifest(root: str | Path) -> tuple[dict, list[Item]]:
    root = Path(root)
    d = json.loads((root / "manifest.json").read_text())
    return d["meta"], [Item(**i) for i in d["items"]]


def verify_manifest(root: str | Path) -> list[str]:
    """Return the ids whose image is missing or whose checksum no longer matches."""
    root = Path(root)
    _, items = load_manifest(root)
    bad = []
    for it in items:
        p = root / it.image
        if not p.exists() or sha256_bytes(p.read_bytes()) != it.sha256:
            bad.append(it.id)
    return bad


def png_data_uri(path: str | Path) -> str:
    return "data:image/png;base64," + base64.b64encode(Path(path).read_bytes()).decode()
