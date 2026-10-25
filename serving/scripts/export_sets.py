"""Render the two fixed accuracy sets to PNG + manifest (rule 3 uses these after every serving change).

    synth300      the exact 300 held-out 'unbiased_test' screens used in results/eval_*.json
    screenspot400 the exact random 400-sample ScreenSpot subset (seed 0) used in results/screenspot_*.json
"""
import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "serving"))

from servebench import envinfo
from servebench.data import Item, save_png, write_manifest

ap = argparse.ArgumentParser()
ap.add_argument("--only", choices=["synth300", "screenspot400"])
ap.add_argument("--out", default=str(ROOT / "serving" / "data"))
a = ap.parse_args()
out = Path(a.out)

if a.only in (None, "synth300"):
    from uiclick.evaluate import test_samples
    root = out / "synth300"
    items = []
    for i, s in enumerate(test_samples("unbiased_test", 300)):
        rel = f"images/{i:04d}.png"
        items.append(Item(f"synth-{i:04d}", rel, save_png(s.image, root / rel), s.instruction, list(s.box),
                          {"kind": s.kind, "seed": s.seed}))
    write_manifest(root, items, {"name": "synth300", "split": "unbiased_test", "n": 300, "source": "uiclick.synth",
                                 "fonts": __import__("uiclick.synth", fromlist=["FONTS"]).FONTS, "env": envinfo.capture()})
    print("synth300:", len(items), "items")

if a.only in (None, "screenspot400"):
    from datasets import load_dataset
    ds = load_dataset("bevaya/ScreenSpot", split="test")
    idx = list(range(len(ds)))
    random.Random(0).shuffle(idx)          # same selection as scripts/eval_screenspot.py --n 400 --seed 0
    idx = sorted(idx[:400])
    root = out / "screenspot400"
    items = []
    for k, i in enumerate(idx):
        r = ds[i]
        rel = f"images/{k:04d}.png"
        items.append(Item(f"ssp-{i:05d}", rel, save_png(r["image"].convert("RGB"), root / rel), r["instruction"],
                          [1000 * v for v in r["bbox"]], {"data_type": r["data_type"], "data_source": r["data_source"],
                                                          "file_name": r["file_name"], "dataset_index": i}))
    # sanity: must be the same subset the original evaluation used
    ref = json.loads((ROOT / "results" / "screenspot_clean.json").read_text())["model"]["examples"]
    assert [x[0] for x in ref[:8]] == [it.instruction for it in items[:8]], "subset differs from results/screenspot_clean.json"
    write_manifest(root, items, {"name": "screenspot400", "dataset": "bevaya/ScreenSpot", "n": 400, "seed": 0,
                                 "note": "third-party screenshots: not committed", "env": envinfo.capture()})
    print("screenspot400:", len(items), "items; subset matches results/screenspot_clean.json")
