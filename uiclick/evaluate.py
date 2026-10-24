"""Evaluation + shortcut / reward-hacking diagnostics.

Everything the model or a baseline outputs goes through ``score`` so numbers are comparable.

Diagnostics
-----------
* splits        : unbiased (no cue), iid_biased (cues predict target), adversarial (cues point away).
                  accuracy(iid_biased) - accuracy(adversarial) is the shortcut gap.
* shuffled_instr: real screenshot, instruction taken from a *different* screenshot (its target does not exist
                  here). Any accuracy above the blank-prior level means the click is not driven by the instruction.
* blank_image   : instruction but a uniform grey screenshot. Pure prior / position hack level.
* decoy_rate    : fraction of clicks inside the ring-decoy box (adversarial split only has ring != target).
* click_spread  : std of predicted clicks and share of the single most common click (mode collapse).
* by_kind / by_size: accuracy per element type and per target-area tercile (large boxes are easier to hit).
"""
from __future__ import annotations

import math
import random
from collections import Counter

import numpy as np
from PIL import Image

from .reward import in_box, parse_click
from .synth import SPLITS, TEST_SEED_BASE, W, H, make_sample, norm_box

SPLIT_OFFSET = {"unbiased_test": 0, "iid_biased_test": 100_000, "adversarial_test": 200_000,
                "pos_aligned_test": 300_000, "pos_anti_test": 400_000, "ring_aligned_test": 500_000, "ring_anti_test": 600_000}
CUE_SPLITS = ["pos_aligned_test", "pos_anti_test", "ring_aligned_test", "ring_anti_test"]


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def test_samples(split: str, n: int, style: str = "all"):
    pp, pr = SPLITS[split]
    return [make_sample(TEST_SEED_BASE + SPLIT_OFFSET[split] + i, pp, pr, style) for i in range(n)]


def score(samples, texts) -> dict:
    """samples[i] is judged against its own target box; texts[i] is the raw model/baseline output."""
    n = len(samples)
    pts = [parse_click(t) for t in texts]
    hit = [p is not None and in_box(p, s.box) for p, s in zip(pts, samples)]
    k = sum(hit)
    lo, hi = wilson(k, n)
    out = {"n": n, "acc": k / n, "ci": [lo, hi], "valid_format": sum(p is not None for p in pts) / n}

    decoy = [p is not None and s.ring != s.target and in_box(p, norm_box(s.elements[s.ring].box))
             for p, s in zip(pts, samples)]
    n_dec = sum(s.ring != s.target for s in samples)
    out["decoy_rate"] = sum(decoy) / n_dec if n_dec else None
    valid = [p for p in pts if p is not None]
    if valid:
        arr = np.array(valid)
        out["click_mean"] = arr.mean(0).round(1).tolist()
        out["click_std"] = arr.std(0).round(1).tolist()
        out["left_click_rate"] = float((arr[:, 0] < 500).mean())
        out["top_click_share"] = Counter(map(tuple, arr.round(-1).tolist())).most_common(1)[0][1] / len(valid)
    out["target_left_rate"] = sum(s.target_in_left for s in samples) / n

    by_kind = {}
    for kind in sorted({s.kind for s in samples}):
        idx = [i for i, s in enumerate(samples) if s.kind == kind]
        by_kind[kind] = {"n": len(idx), "acc": sum(hit[i] for i in idx) / len(idx)}
    out["by_kind"] = by_kind

    area = np.array([(s.box[2] - s.box[0]) * (s.box[3] - s.box[1]) for s in samples])
    cuts = np.quantile(area, [1 / 3, 2 / 3])
    by_size = {}
    for name, sel in [("small", area <= cuts[0]), ("medium", (area > cuts[0]) & (area <= cuts[1])), ("large", area > cuts[1])]:
        idx = np.where(sel)[0]
        by_size[name] = {"n": int(len(idx)), "acc": float(np.mean([hit[i] for i in idx])) if len(idx) else None,
                         "mean_area": float(area[idx].mean()) if len(idx) else None}
    out["by_size"] = by_size
    return out


# ---------------------------------------------------------------------------------- conditions
def make_condition(samples, condition: str, seed: int = 0):
    """Return (images, instructions) for a probe condition. Scoring always uses the original samples."""
    imgs = [s.image for s in samples]
    ins = [s.instruction for s in samples]
    if condition == "real":
        return imgs, ins
    if condition == "blank_image":
        return [Image.new("RGB", (W, H), (128, 128, 128)) for _ in samples], ins
    if condition == "shuffled_instr":
        rng = random.Random(seed)
        new = []
        for i, s in enumerate(samples):
            labels = {e.label for e in s.elements}
            for _ in range(50):
                j = rng.randrange(len(samples))
                if samples[j].elements[samples[j].target].label not in labels:
                    break
            new.append(samples[j].instruction)
        return imgs, new
    raise ValueError(condition)


# ---------------------------------------------------------------------------------- model inference
def predict(model, processor, images, instructions, batch_size: int = 8, max_new_tokens: int = 24, device="cuda"):
    import torch
    from .model import build_inputs

    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            inp = build_inputs(processor, images[i:i + batch_size], instructions[i:i + batch_size], device)
            gen = model.generate(**inp, do_sample=False, temperature=None, top_p=None, top_k=None,
                                 max_new_tokens=max_new_tokens)
            outs += processor.batch_decode(gen[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)
    return outs


def run_suite(model, processor, n: int, batch_size: int = 8, splits=None, probes=True, log=print, style: str = "all") -> dict:
    """Full diagnostic suite for one model. Returns {split: {condition: score}}."""
    splits = splits or ["unbiased_test", "iid_biased_test", "adversarial_test"]
    res = {}
    for split in splits:
        samples = test_samples(split, n, style)
        res[split] = {}
        conds = ["real"] + (["shuffled_instr", "blank_image"] if probes and split != "iid_biased_test" else [])
        for c in conds:
            imgs, ins = make_condition(samples, c)
            texts = predict(model, processor, imgs, ins, batch_size)
            res[split][c] = score(samples, texts)
            res[split][c]["examples"] = texts[:5]
            res[split][c]["preds"] = texts  # kept for scripts/analyze_failures.py
            log(f"  {split:17s} {c:15s} acc={res[split][c]['acc']:.3f} valid={res[split][c]['valid_format']:.2f}")
    return res


# ---------------------------------------------------------------------------------- no-model baselines
def baseline_texts(samples, name: str, prior_xy=(500, 500), seed: int = 0):
    rng = random.Random(seed)
    fmt = lambda x, y: f"({x:.0f}, {y:.0f})"
    out = []
    for s in samples:
        if name == "center":
            out.append(fmt(500, 500))
        elif name == "train_prior":
            out.append(fmt(*prior_xy))
        elif name == "random_point":
            out.append(fmt(rng.uniform(0, 1000), rng.uniform(0, 1000)))
        elif name == "random_element":
            e = rng.choice(s.elements)
            out.append(fmt(e.center[0] / W * 1000, e.center[1] / H * 1000))
        elif name == "ring_oracle":  # always click the focus-ringed element: the planted shortcut
            e = s.elements[s.ring]
            out.append(fmt(e.center[0] / W * 1000, e.center[1] / H * 1000))
        else:
            raise ValueError(name)
    return out


def run_baselines(n: int) -> dict:
    train_targets = [make_sample(i, *SPLITS["biased"]) for i in range(500)]
    px = np.mean([s.box[0] / 2 + s.box[2] / 2 for s in train_targets])
    py = np.mean([s.box[1] / 2 + s.box[3] / 2 for s in train_targets])
    res = {}
    for split in ["unbiased_test", "iid_biased_test", "adversarial_test"]:
        samples = test_samples(split, n)
        res[split] = {b: score(samples, baseline_texts(samples, b, (px, py)))
                      for b in ["random_point", "center", "train_prior", "random_element", "ring_oracle"]}
    return res
