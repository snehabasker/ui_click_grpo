"""Zero-shot transfer check on REAL UIs (ScreenSpot, test split only - never used for training).

    python scripts/eval_screenspot.py --name base
    python scripts/eval_screenspot.py --name clean --adapter runs/clean/adapter --draw 6

Convention: ScreenSpot bbox = [x0, y0, x1, y1] normalised to 0-1. The model answers in 0-1000 normalised coords of
the image it was shown. Both are resolution-independent, so a resize cannot shift the click as long as the aspect
ratio is preserved (Qwen's smart-resize only rounds to a multiple of 28 px, <= 2% distortion). ``--draw K`` saves K
screenshots with the ground-truth box (green) and the predicted click (red) at the ORIGINAL resolution so the
convention can be checked by eye instead of trusted.
"""
import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from uiclick.evaluate import wilson
from uiclick.reward import in_box, parse_click

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
ap.add_argument("--adapter", default=None)
ap.add_argument("--n", type=int, default=400, help="random subset size (0 = all 1272)")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--bs", type=int, default=2)
ap.add_argument("--max_pixels", type=int, default=1280 * 28 * 28)
ap.add_argument("--draw", type=int, default=0)
ap.add_argument("--mode", default="point", choices=["point", "box"],
                help="box: control that asks the model for its NATIVE bounding-box output and clicks the box centre")
ap.add_argument("--dataset", default="bevaya/ScreenSpot")
a = ap.parse_args()


def summarize(rows, texts, boxes):
    pts = [parse_click(t) for t in texts]
    hit = np.array([p is not None and in_box(p, b) for p, b in zip(pts, boxes)])
    area = np.array([(b[2] - b[0]) * (b[3] - b[1]) for b in boxes])
    cuts = np.quantile(area, [1 / 3, 2 / 3])
    out = {"n": len(rows), "acc": float(hit.mean()), "ci": list(wilson(int(hit.sum()), len(hit))),
           "valid_format": float(np.mean([p is not None for p in pts]))}
    for key in ["data_source", "data_type"]:
        g = defaultdict(list)
        for r, h in zip(rows, hit):
            g[r[key]].append(h)
        out[f"by_{key}"] = {k: {"n": len(v), "acc": float(np.mean(v))} for k, v in sorted(g.items())}
    out["by_size"] = {nm: {"n": int(m.sum()), "acc": float(hit[m].mean()) if m.any() else None, "mean_area": float(area[m].mean())}
                      for nm, m in [("small", area <= cuts[0]), ("medium", (area > cuts[0]) & (area <= cuts[1])), ("large", area > cuts[1])]}
    return out


from datasets import load_dataset

ds = load_dataset(a.dataset, split="test")
idx = list(range(len(ds)))
if a.n and a.n < len(ds):
    random.Random(a.seed).shuffle(idx)
    idx = sorted(idx[:a.n])
rows = [{k: ds[i][k] for k in ["file_name", "instruction", "data_type", "data_source"]} for i in idx]
boxes = [tuple(1000 * v for v in ds[i]["bbox"]) for i in idx]
print(f"{len(idx)} samples; bbox sample (0-1000): {[round(v) for v in boxes[0]]}", flush=True)

res = {}
rng = random.Random(a.seed)
res["baseline_center"] = summarize(rows, ["(500, 500)"] * len(rows), boxes)
res["baseline_random_point"] = summarize(rows, [f"({rng.uniform(0, 1000):.0f}, {rng.uniform(0, 1000):.0f})" for _ in rows], boxes)
res["baseline_oracle_box_centre"] = summarize(rows, [f"({(b[0]+b[2])/2:.0f}, {(b[1]+b[3])/2:.0f})" for b in boxes], boxes)  # sanity: must be ~100%

if a.name != "baselines":
    from uiclick.evaluate import predict
    from uiclick.meta import run_meta
    from uiclick.model import load_model, load_processor

    import uiclick.model as um
    if a.mode == "box":
        um.PROMPT = ("{instruction}\nOutput the bounding box of the target element as (x1, y1), (x2, y2), "
                     "with coordinates normalised to 0-1000 relative to the image width and height.")
    model, proc = load_model(adapter=a.adapter), load_processor(max_pixels=a.max_pixels)
    imgs = [ds[i]["image"].convert("RGB") for i in idx]
    texts = predict(model, proc, imgs, [r["instruction"] for r in rows], batch_size=a.bs)
    if a.mode == "box":  # click the centre of the predicted box; anything that is not exactly 4 numbers is invalid
        import re
        def centre(t):
            v = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", re.sub(r"<\|[^|>]*\|>", " ", t))]
            return f"({(v[0] + v[2]) / 2:.0f}, {(v[1] + v[3]) / 2:.0f})" if len(v) == 4 else "invalid"
        raw, texts = texts, [centre(t) for t in texts]
    res["model"] = summarize(rows, texts, boxes)
    res["model"]["examples"] = list(zip([r["instruction"] for r in rows[:8]], (raw if a.mode == "box" else texts)[:8]))
    print(f"{a.name}: acc={res['model']['acc']:.3f} valid={res['model']['valid_format']:.2f}  "
          f"center-baseline={res['baseline_center']['acc']:.3f}", flush=True)

    if a.draw:
        from PIL import ImageDraw
        for k in range(min(a.draw, len(imgs))):
            im = imgs[k].copy()
            w, h = im.size
            d = ImageDraw.Draw(im)
            b = boxes[k]
            d.rectangle((b[0] / 1000 * w, b[1] / 1000 * h, b[2] / 1000 * w, b[3] / 1000 * h), outline=(0, 200, 0), width=max(2, w // 300))
            p = parse_click(texts[k])
            if p:
                x, y, r = p[0] / 1000 * w, p[1] / 1000 * h, max(6, w // 120)
                d.line((x - r, y, x + r, y), fill=(255, 0, 0), width=3); d.line((x, y - r, x, y + r), fill=(255, 0, 0), width=3)
            d.text((6, 4), f"{rows[k]['instruction']} -> {texts[k]}", fill=(255, 0, 255))
            im.thumbnail((900, 900))
            im.save(ROOT / "results" / f"convention_screenspot_{a.name}_{k}.png")
    res["meta"] = run_meta({"eval_args": vars(a), "dataset": a.dataset, "n_dataset": len(ds)})

(ROOT / "results" / f"screenspot_{a.name}.json").write_text(json.dumps(res, indent=1))
print("wrote results/screenspot_%s.json" % a.name)
