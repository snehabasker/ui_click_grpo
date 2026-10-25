"""Does the merged checkpoint click like the LoRA-adapter model it was merged from?

    python serving/scripts/verify_merge.py --adapter runs/clean/adapter --merged serving/models/merged-clean

Both models run through the same HF code path (uiclick.evaluate.predict, greedy) on both fixed sets. Pass criteria, fixed
BEFORE looking at results:  |acc_merged - acc_adapter| <= 2.0 points  AND  exact two-sided sign test on the paired
discordant items has p > 0.05.  A failed check is reported as failed, not tuned away.
"""
import argparse
import gc
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "serving"))

import torch

from servebench import accuracy, envinfo
from servebench.data import load_manifest
from servebench.stats import sign_test_p
from uiclick.model import MODEL_ID, load_model, load_processor

MAX_DELTA, ALPHA = 0.02, 0.05
SETS = {"synth300": 8, "screenspot400": 2}   # name -> batch size (same as the original evaluations)

ap = argparse.ArgumentParser()
ap.add_argument("--adapter", required=True)
ap.add_argument("--merged", required=True)
ap.add_argument("--sets", nargs="+", default=list(SETS), choices=list(SETS))
a = ap.parse_args()

preds = {}
for name, kw in [("adapter", dict(model_id=MODEL_ID, adapter=a.adapter)), ("merged", dict(model_id=a.merged))]:
    model = load_model(**kw)
    proc = load_processor(kw["model_id"], max_pixels=1280 * 28 * 28)
    for s in a.sets:
        root = ROOT / "serving" / "data" / s
        _, items = load_manifest(root)
        t0 = time.time()
        preds[(name, s)] = accuracy.predict_hf(model, proc, root, items, SETS[s])
        sc = accuracy.score(items, preds[(name, s)])
        print(f"{name:8s} {s:14s} acc={sc['acc']:.4f} valid={sc['valid_format']:.3f} ({time.time() - t0:.0f}s)", flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()

report = {}
for s in a.sets:
    _, items = load_manifest(ROOT / "serving" / "data" / s)
    A, M = accuracy.score(items, preds[("adapter", s)]), accuracy.score(items, preds[("merged", s)])
    ha, hm = A["per_item_hit"], M["per_item_hit"]
    b, c = sum(x and not y for x, y in zip(ha, hm)), sum(y and not x for x, y in zip(ha, hm))
    p = sign_test_p(b, c)
    delta = M["acc"] - A["acc"]
    report[s] = {"n": len(items), "adapter": {k: A[k] for k in ("acc", "ci", "valid_format")},
                 "merged": {k: M[k] for k in ("acc", "ci", "valid_format")}, "delta_acc": delta,
                 "only_adapter_correct": b, "only_merged_correct": c, "sign_test_p": p,
                 "identical_output_text": sum(x.strip() == y.strip() for x, y in zip(preds[("adapter", s)], preds[("merged", s)])) / len(items),
                 "pass": abs(delta) <= MAX_DELTA and p > ALPHA}
    print(f"{s}: adapter {A['acc']:.4f} vs merged {M['acc']:.4f} (delta {100*delta:+.1f} pts), discordant {b}/{c}, "
          f"sign-test p={p:.3f}, identical text {report[s]['identical_output_text']:.1%} -> {'PASS' if report[s]['pass'] else 'FAIL'}")

out = ROOT / "serving" / "results" / "raw" / f"merge_verify_{time.strftime('%Y%m%d-%H%M%S')}.json"
out.write_text(json.dumps({"kind": "merge_verify", "criteria": {"max_abs_delta_acc": MAX_DELTA, "sign_test_alpha": ALPHA},
                           "env": envinfo.capture(a.merged, {"args": vars(a)}), "results": report,
                           "predictions": {f"{n}/{s}": t for (n, s), t in preds.items()}}, indent=1))
print("wrote", out)
