"""Agent-style multi-turn trace generator built on uiclick.synth.

A session = shared system prompt + few-shot screenshots, then 3-8 turns of (screenshot, instruction) -> click.
Controls: how many sessions share an identical prefix (``shared_fraction``), over how many distinct shared prefixes
(``n_shared_groups``, with a skewed popularity), and how much history each turn carries.

Everything is deterministic in the config. Seeds are disjoint from training (< 1e6) and from the held-out test splits
(1e7 .. 1.07e7).  Images are rendered once and stored by content hash; trace.jsonl references them.
"""
from __future__ import annotations

import hashlib
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from uiclick.synth import FONTS, make_sample

from .data import save_png
from .prompts import CONTEXTS, build_system

FEWSHOT_SHARED_BASE = 2_000_000
FEWSHOT_UNIQUE_BASE = 3_000_000
SERVE_BASE = 20_000_000
WARMUP_BASE = 30_000_000


@dataclass
class TraceConfig:
    name: str = "default"
    n_sessions: int = 96
    min_turns: int = 3
    max_turns: int = 8
    shared_fraction: float = 0.75   # share of sessions that use one of the shared prefixes (the rest get a unique prefix)
    n_shared_groups: int = 4
    n_fewshot: int = 2
    history: str = "text"           # none | text (previous instructions + answers, no old images) | full (with old images)
    warmup_sessions: int = 4
    seed: int = 0


def _answer(box) -> str:
    return f"({(box[0] + box[2]) / 2:.0f}, {(box[1] + box[3]) / 2:.0f})"


def _render(seed: int, out: Path) -> dict:
    s = make_sample(seed)
    tmp = out / "images" / "_tmp.png"
    h = save_png(s.image, tmp)
    rel = f"images/{h[:16]}.png"
    final = out / rel
    if final.exists():
        tmp.unlink()
    else:
        tmp.replace(final)
    return {"image": rel, "sha256": h, "instruction": s.instruction, "box": list(s.box), "answer": _answer(s.box),
            "kind": s.kind, "seed": seed}


def _fewshot(base: int, n: int, out: Path) -> list[dict]:
    return [_render(base + j, out) for j in range(n)]


def generate(cfg: TraceConfig, out_dir: str | Path) -> dict:
    out = Path(out_dir)
    (out / "images").mkdir(parents=True, exist_ok=True)
    rng = random.Random(cfg.seed)
    n_shared = round(cfg.n_sessions * cfg.shared_fraction)
    flags = [True] * n_shared + [False] * (cfg.n_sessions - n_shared)
    rng.shuffle(flags)
    weights = [1 / (g + 1) for g in range(cfg.n_shared_groups)]   # skewed: group 0 is the most popular prefix
    group_cache: dict[int, tuple[str, list[dict]]] = {}
    counter = 0

    def make_session(sid: int, shared: bool, warm: bool) -> dict:
        nonlocal counter
        if shared:
            g = rng.choices(range(cfg.n_shared_groups), weights)[0]
            if g not in group_cache:
                group_cache[g] = (build_system(CONTEXTS[g % len(CONTEXTS)]),
                                  _fewshot(FEWSHOT_SHARED_BASE + g * 10, cfg.n_fewshot, out))
            system, fewshot = group_cache[g]
            prefix_id = f"shared-{g}"
        else:
            ref = hashlib.sha256(f"{cfg.seed}-{sid}-{warm}".encode()).hexdigest()[:10]
            system = build_system(rng.choice(CONTEXTS), ref)
            fewshot = _fewshot((WARMUP_BASE if warm else FEWSHOT_UNIQUE_BASE) + sid * 10, cfg.n_fewshot, out)
            prefix_id = f"unique-{sid}{'-warm' if warm else ''}"
        turns = []
        for _ in range(rng.randint(cfg.min_turns, cfg.max_turns)):
            seed = (WARMUP_BASE + 5_000_000 if warm else SERVE_BASE) + cfg.seed * 1_000_000 + counter
            counter += 1
            turns.append(_render(seed, out))
        return {"session_id": f"{'w' if warm else 's'}{sid:04d}", "prefix_id": prefix_id, "shared": shared,
                "history": cfg.history, "system": system, "fewshot": fewshot, "turns": turns}

    sessions = [make_session(i, f, False) for i, f in enumerate(flags)]
    warm = [make_session(i, False, True) for i in range(cfg.warmup_sessions)]
    for name, rows in (("trace.jsonl", sessions), ("warmup.jsonl", warm)):
        (out / name).write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    meta = {"config": asdict(cfg), "n_sessions": len(sessions), "n_requests": sum(len(s["turns"]) for s in sessions),
            "n_shared_sessions": n_shared,
            "sessions_per_prefix": {k: sum(s["prefix_id"] == k for s in sessions)
                                    for k in sorted({s["prefix_id"] for s in sessions if s["shared"]})},
            "turns_per_session": {"min": min(len(s["turns"]) for s in sessions), "max": max(len(s["turns"]) for s in sessions)},
            "n_images": len(list((out / "images").glob("*.png"))), "fonts": FONTS,
            "trace_sha256": hashlib.sha256((out / "trace.jsonl").read_bytes()).hexdigest()}
    (out / "trace_meta.json").write_text(json.dumps(meta, indent=1))
    return meta


def load_trace(trace_dir: str | Path, name: str = "trace.jsonl") -> list[dict]:
    return [json.loads(l) for l in (Path(trace_dir) / name).read_text().splitlines() if l.strip()]
