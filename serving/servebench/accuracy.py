"""Click accuracy on a fixed image set, through either a local HF model or an OpenAI-compatible endpoint.

Used after every serving change (rule 3). Scoring is the training reward's own parser, so a served model is judged
exactly like the trained one.
"""
from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "serving"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from .prompts import PROMPT
from uiclick.reward import in_box, parse_click

from .data import Item, png_data_uri
from .stats import wilson


def score(items: list[Item], texts: list[str]) -> dict:
    pts = [parse_click(t) for t in texts]
    hits = [p is not None and in_box(p, it.box) for p, it in zip(pts, items)]
    k = sum(hits)
    out = {"n": len(items), "hits": k, "acc": k / len(items), "ci": list(wilson(k, len(items))),
           "valid_format": sum(p is not None for p in pts) / len(items), "per_item_hit": hits}
    for key in ("data_type", "kind"):
        groups = {}
        for it, h in zip(items, hits):
            if key in it.meta:
                groups.setdefault(it.meta[key], []).append(h)
        if groups:
            out[f"by_{key}"] = {g: {"n": len(v), "acc": sum(v) / len(v)} for g, v in sorted(groups.items())}
    return out


def zero_shot_messages(root: Path, item: Item) -> list[dict]:
    """The exact single-image prompt the policy was trained and evaluated with (no system prompt, no few-shot)."""
    return [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": png_data_uri(root / item.image)}},
        {"type": "text", "text": PROMPT.format(instruction=item.instruction)}]}]


def predict_hf(model, processor, root: Path, items: list[Item], batch_size: int = 8) -> list[str]:
    from PIL import Image

    from uiclick.evaluate import predict

    images = [Image.open(root / it.image).convert("RGB") for it in items]
    return predict(model, processor, images, [it.instruction for it in items], batch_size=batch_size)


async def predict_openai(root: Path, items: list[Item], base_url: str, model: str, concurrency: int = 4,
                         max_tokens: int = 24, timeout: float = 300.0) -> tuple[list[str], list[float]]:
    import aiohttp

    sem = asyncio.Semaphore(concurrency)
    texts = [""] * len(items)
    lat = [0.0] * len(items)

    async def one(i: int, it: Item, sess: aiohttp.ClientSession):
        payload = {"model": model, "messages": zero_shot_messages(root, it), "max_tokens": max_tokens, "temperature": 0}
        async with sem:
            t0 = time.perf_counter()
            async with sess.post(f"{base_url}/v1/chat/completions", json=payload) as r:
                body = await r.json()
            lat[i] = time.perf_counter() - t0
        try:
            texts[i] = body["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError):
            texts[i] = f"<error: {str(body)[:200]}>"

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as sess:
        await asyncio.gather(*(one(i, it, sess) for i, it in enumerate(items)))
    return texts, lat
